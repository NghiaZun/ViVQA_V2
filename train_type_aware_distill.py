"""
Type-Aware Hybrid Distillation

Architecture:
1. Main task: Generate answer + reasoning
2. Auxiliary task: Classify reasoning type (COUNTING, CAUSAL, etc.)
3. Loss: 0.2*GT + 0.5*Teacher + 0.2*Embedding + 0.1*Type_Classification

Key innovation: Model learns reasoning TYPE → better structured reasoning
"""

import os
import json
import gc
import math
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    BlipProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup
)
from model import VQAGenModel

# =====================
# REPRODUCIBILITY
# =====================
def set_seed(seed: int = 42):
    """Set seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Enable cuDNN auto-tuner for better performance
torch.backends.cudnn.benchmark = True

# =====================
# MEMORY MANAGEMENT
# =====================
def clear_memory():
    """Clear GPU cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# =====================
# CONFIG WITH DATACLASS
# =====================
@dataclass
class TrainConfig:
    # Paths
    train_csv: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_dir: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    teacher_jsonl: str = "/kaggle/input/teacher-5-12/teacher_outputs_train.jsonl"
    teacher_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    checkpoint_dir: str = "/kaggle/input/checkpoints/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working"
    
    # Training hyperparameters
    batch_size: int = 4
    accum_steps: int = 8                # Effective batch = 4 * 8 = 32
    num_epochs: int = 20
    val_ratio: float = 0.1
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Learning rates
    base_lr: float = 3e-5               # For fusion/decoder/text
    vision_lr: float = 1e-5             # Smaller for vision encoder
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_ratio: float = 0.1
    use_amp: bool = True
    resume_epoch: int = 0
    
    # Progressive Training Strategy
    stage1_epochs: int = 8              # Focus on teacher + type
    stage2_epochs: int = 8              # Balance all losses
    # remaining epochs -> stage3: Fine-tune with GT
    
    # Early stopping
    es_patience: int = 6
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log.csv"
    curve_png: str = "training_curve.png"
    clear_cache_every_n_steps: int = 20
    
    # Teacher inference
    temperature: float = 2.5
    max_output_len: int = 128

# Loss weights - DYNAMIC per stage - GT-GUIDED (NO separate GT loss!)
def get_loss_weights(stage: int, temperature: float = 2.0):
    """Get loss weights based on training stage with temperature scaling for embeddings
    
    GT-GUIDED STRATEGY (teacher answer = GT + reasoning):
    - Teacher loss = main optimization target (contains GT answer + reasoning)
    - Embedding alignment = match teacher's representations
    - Type classification = structured reasoning awareness
    
    Args:
        stage: Training stage (1, 2, or 3)
        temperature: Temperature for scaling embedding alignment losses (higher = softer alignment)
        
    Returns:
        Dictionary with loss weights and temperature (NO 'gt' key - not used!)
    """
    if stage == 1:
        # Stage 1: Learn reasoning patterns + type classification
        return {
            'teacher': 0.7,     # HIGH - teacher = GT + reasoning
            'vision': 0.15,     # Vision-teacher alignment
            'text': 0.05,       # Text-teacher alignment
            'type': 0.1,        # Type classification
            'temperature': temperature
        }
    elif stage == 2:
        # Stage 2: REASONING MASTERY - Strong teacher guidance
        return {
            'teacher': 0.75,    # VERY HIGH - master reasoning generation
            'vision': 0.12,     # Slightly reduce embeddings
            'text': 0.03,
            'type': 0.1,
            'temperature': temperature
        }
    else:
        # Stage 3: POLISH - Focus on teacher output quality
        return {
            'teacher': 0.8,     # DOMINANT - full output quality
            'vision': 0.08,     # Lower embeddings
            'text': 0.02,
            'type': 0.1,
            'temperature': temperature
        }

# Reasoning types (matching teacher_outputs.jsonl)
REASONING_TYPES = [
    "DESCRIPTIVE",      # Most common - visual descriptions
    "SPATIAL",          # Location/position questions
    "COMMONSENSE",      # Common sense reasoning
    "COUNTING",         # Count objects
    "CAUSAL",           # Cause-effect relationships
    "OBJECT",           # Object identification
    "INTENT",           # Intent/purpose questions
    "VISUAL_RECOGNITION" # Default fallback
]
TYPE_TO_IDX = {t: i for i, t in enumerate(REASONING_TYPES)}
NUM_TYPES = len(REASONING_TYPES)

# =====================
# TYPE-AWARE MODEL
# =====================
class TypeAwareVQAModel(nn.Module):
    """
    Wrapper around VQAGenModel with reasoning type classification head
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Type classification head
        hidden_dim = 768
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_TYPES)
        )
        
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Base model forward
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Get fusion embeddings for type classification
        # IMPORTANT: Allow gradient flow for better type classification learning
        v_out = self.base_model.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = v_out.mean(dim=1)
        t_out = self.base_model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        t_feat = t_out[:, 0, :]
        fused = torch.cat([v_feat, t_feat], dim=-1)
        fused = self.base_model.fusion(fused)  # [batch, hidden_dim]
        
        # Predict reasoning type
        type_logits = self.type_classifier(fused.squeeze(1))  # [batch, NUM_TYPES]
        
        return outputs, type_logits
    
    def generate(self, pixel_values, input_ids, attention_mask, **kwargs):
        return self.base_model.generate(pixel_values, input_ids, attention_mask, **kwargs)

# =====================
# PROGRESSIVE UNFREEZING
# =====================
def set_training_stage(model: TypeAwareVQAModel, stage: int):
    """
    Progressive unfreezing strategy:
    Stage 1: Train fusion + decoder + type_classifier only
    Stage 2: + Unfreeze text encoder
    Stage 3: + Unfreeze vision encoder (last block)
    """
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False
    
    # Always train decoder, fusion, and type classifier
    for p in model.base_model.decoder.parameters():
        p.requires_grad = True
    if hasattr(model.base_model, "fusion"):
        for p in model.base_model.fusion.parameters():
            p.requires_grad = True
    for p in model.type_classifier.parameters():
        p.requires_grad = True
    
    # Cross-attention and projections if exist
    if hasattr(model.base_model, "cross_attn"):
        for p in model.base_model.cross_attn.parameters():
            p.requires_grad = True
    if hasattr(model.base_model, "vision_proj"):
        for p in model.base_model.vision_proj.parameters():
            p.requires_grad = True
    if hasattr(model.base_model, "text_proj"):
        for p in model.base_model.text_proj.parameters():
            p.requires_grad = True
    
    # Stage 2+: Unfreeze text encoder
    if stage >= 2:
        for p in model.base_model.text_encoder.parameters():
            p.requires_grad = True
    
    # Stage 3: Unfreeze vision encoder (last block)
    if stage >= 3:
        try:
            # Try to unfreeze last transformer block
            last_block = model.base_model.vision_encoder.encoder.layers[-1]
            for p in last_block.parameters():
                p.requires_grad = True
        except Exception:
            # Fallback: unfreeze entire vision encoder
            for p in model.base_model.vision_encoder.parameters():
                p.requires_grad = True

def build_optimizer(model: TypeAwareVQAModel, cfg: TrainConfig):
    """Build optimizer with different LR for vision vs other components"""
    vision_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vision_encoder" in n:
            vision_params.append(p)
        else:
            other_params.append(p)
    
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": cfg.base_lr})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": cfg.vision_lr})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer

def count_trainable_params(model: nn.Module):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =====================
# DATASET WITH REASONING TYPE
# =====================
class TypeAwareDataset(Dataset):
    def __init__(self, csv_path, image_dir, teacher_jsonl, student_processor, 
                 student_text_tokenizer, student_decoder_tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.student_processor = student_processor
        self.student_text_tokenizer = student_text_tokenizer
        self.student_decoder_tokenizer = student_decoder_tokenizer
        self.max_len = max_len
        
        # Load teacher outputs - tự động tìm file merged
        self.teacher_outputs = {}
        teacher_file = self._find_teacher_file(teacher_jsonl)
        if teacher_file and os.path.exists(teacher_file):
            with open(teacher_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    img_id = data.get('img_id', data.get('image_id'))
                    self.teacher_outputs[str(img_id)] = data
            print(f"[INFO] Loaded {len(self.teacher_outputs)} teacher outputs from {teacher_file}")
        else:
            print(f"[WARN] No teacher outputs found. Training with GT only.")
    
    def _find_teacher_file(self, default_path):
        """Tìm file teacher_outputs_merged trong /kaggle/input"""
        # Thử path mặc định trước
        if os.path.exists(default_path):
            return default_path
        
        # Tìm trong kaggle input
        kaggle_input = "/kaggle/input"
        if not os.path.exists(kaggle_input):
            return None
        
        print(f"[INFO] 🔍 Searching for teacher_outputs_merged* in {kaggle_input}...")
        for root, dirs, files in os.walk(kaggle_input):
            for file in files:
                if file.startswith("teacher_outputs_merged") and file.endswith(".jsonl"):
                    found_path = os.path.join(root, file)
                    print(f"[INFO] ✅ Found merged teacher file: {found_path}")
                    return found_path
        
        return None
        
    def __len__(self):
        return len(self.df)
    
    def _parse_reasoning_type(self, text, log_failures=False):
        """Extract reasoning type from teacher output with better error handling
        
        Args:
            text: Raw teacher output text
            log_failures: Whether to log failed parsing attempts
            
        Returns:
            Reasoning type index (defaults to DESCRIPTIVE if parsing fails)
        """
        import re
        
        if not text or not isinstance(text, str):
            return TYPE_TO_IDX["DESCRIPTIVE"]
        
        # Try to find [TYPE] pattern
        match = re.search(r'\[([A-Z_]+)\]', text)
        if match:
            rtype = match.group(1)
            if rtype in TYPE_TO_IDX:
                return TYPE_TO_IDX[rtype]
        
        # Try to find Reasoning (TYPE): pattern
        match = re.search(r'Reasoning\s*\(([^)]+)\):', text)
        if match:
            rtype = match.group(1).upper().replace(' ', '_').replace('-', '_')
            if rtype in TYPE_TO_IDX:
                return TYPE_TO_IDX[rtype]
        
        # Try to find Type: pattern
        match = re.search(r'Type:\s*([A-Z_]+)', text)
        if match:
            rtype = match.group(1)
            if rtype in TYPE_TO_IDX:
                return TYPE_TO_IDX[rtype]
        
        # Log failure if requested
        if log_failures and not hasattr(self, '_logged_parse_warning'):
            print(f"[WARN] Failed to parse reasoning type from: {text[:100]}...")
            self._logged_parse_warning = True
        
        # Default to DESCRIPTIVE
        return TYPE_TO_IDX["DESCRIPTIVE"]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['img_id'])
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        question = str(row["question"])
        gt_answer = str(row["answer"])
        
        # Get teacher output - GT-GUIDED (answer = GT, adds reasoning)
        teacher_data = self.teacher_outputs.get(img_id, {})
        teacher_answer = teacher_data.get("teacher_answer", gt_answer)
        teacher_reasoning = teacher_data.get("teacher_reasoning", "")
        teacher_raw = teacher_data.get("teacher_raw", "")
        
        # GT-GUIDED VALIDATION: Verify teacher_answer matches gt_answer
        if teacher_answer and teacher_answer.strip().lower() != gt_answer.strip().lower():
            if not hasattr(self, '_gt_mismatch_warned'):
                print(f"[WARN] ⚠️  GT-guided mismatch detected!")
                print(f"  Image: {img_id}")
                print(f"  GT: '{gt_answer}' vs Teacher: '{teacher_answer}'")
                print(f"  This should NOT happen in GT-guided mode!")
                self._gt_mismatch_warned = True
            # Force teacher_answer = gt_answer to maintain GT-guided guarantee
            teacher_answer = gt_answer
        
        # LIGHTWEIGHT QUALITY CHECK for GT-guided teacher
        # Accept fallback entries (they have _fallback flag)
        is_fallback = teacher_data.get('_fallback', False)
        use_teacher = True
        
        if not is_fallback and teacher_reasoning:
            # Only check for obvious failures in teacher-generated (not fallback)
            bad_patterns = [
                len(teacher_reasoning) < 5,   # Too short (likely parsing error)
                len(teacher_reasoning) > 250,  # Suspiciously long
                "..." in teacher_reasoning and len(teacher_reasoning) < 20,  # Incomplete
            ]
            
            if any(bad_patterns):
                use_teacher = False
                if not hasattr(self, '_quality_warning_shown'):
                    print(f"[INFO] ⚠️  Filtering {sum(bad_patterns)} low-quality teacher outputs")
                    self._quality_warning_shown = True
        # Fallback entries are ALWAYS accepted (simple but valid)
        
        # Construct full teacher output: Answer: X\nReasoning: Y
        # Note: teacher_answer should equal gt_answer (GT-guided)
        if use_teacher and teacher_reasoning:
            teacher_full = f"Answer: {teacher_answer}\nReasoning: {teacher_reasoning}"
        else:
            # Fallback to GT if teacher reasoning is broken
            teacher_full = f"Answer: {gt_answer}"
        
        # Extract reasoning type
        reasoning_type_idx = self._parse_reasoning_type(teacher_raw if teacher_raw else teacher_reasoning)
        
        # Student inputs
        student_pixel_values = self.student_processor(image, return_tensors="pt").pixel_values[0]
        student_q_enc = self.student_text_tokenizer(
            question, truncation=True, padding="max_length",
            max_length=64, return_tensors="pt"
        )
        
        # Labels: Ground truth (simple answer)
        gt_enc = self.student_decoder_tokenizer(
            gt_answer, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        
        # Teacher target (answer + reasoning)
        teacher_enc = self.student_decoder_tokenizer(
            teacher_full, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        
        return {
            "image": image,
            "question": question,
            "student_pixel_values": student_pixel_values,
            "student_input_ids": student_q_enc.input_ids[0],
            "student_attention_mask": student_q_enc.attention_mask[0],
            "gt_labels": gt_enc.input_ids[0],
            "teacher_labels": teacher_enc.input_ids[0],
            "reasoning_type": reasoning_type_idx
        }

# =====================
# TEACHER FORWARD
# =====================
@torch.no_grad()
def get_teacher_embeddings(images, questions, teacher, teacher_processor):
    """Get teacher embeddings for alignment
    
    Args:
        images: List of PIL Images
        questions: List of question strings
        teacher: Teacher VLM model (Qwen2-VL)
        teacher_processor: Teacher's processor for tokenization
        
    Returns:
        teacher_embeddings: [batch_size, hidden_dim] tensor with normalized embeddings
    """
    messages_batch = []
    for img, q in zip(images, questions):
        messages_batch.append([
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q}
            ]}
        ])
    
    texts = [teacher_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
             for msg in messages_batch]
    
    inputs = teacher_processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(teacher.device)
    
    outputs = teacher(**inputs, output_hidden_states=True, return_dict=True)
    teacher_embeddings = outputs.hidden_states[-1].mean(dim=1)
    
    return teacher_embeddings

# =====================
# TYPE-AWARE HYBRID LOSS
# =====================
def compute_type_aware_loss(student, batch, device, teacher_embeddings=None, loss_weights=None):
    """
    Multi-task loss - OPTIMIZED for GT-guided teacher:
    1. CE with teacher output (teacher = GT + reasoning)
    2. Vision embedding alignment
    3. Text embedding alignment
    4. Reasoning type classification
    
    NOTE: GT loss NOT computed (weight=0, redundant with teacher)
    
    Args:
        student: Student model (TypeAwareVQAModel)
        batch: Batch dictionary with inputs and labels
        device: torch.device for tensor placement
        teacher_embeddings: Optional teacher embeddings for alignment loss
        loss_weights: dict with keys ['gt', 'teacher', 'vision', 'text', 'type']
        
    Returns:
        Tuple of (total_loss, loss_gt, loss_teacher, loss_type, loss_vision, loss_text)
    """
    if loss_weights is None:
        loss_weights = get_loss_weights(2)  # Default to stage 2 weights
    
    pixel_values = batch["student_pixel_values"].to(device)
    input_ids = batch["student_input_ids"].to(device)
    attention_mask = batch["student_attention_mask"].to(device)
    gt_labels = batch["gt_labels"].to(device)
    teacher_labels = batch["teacher_labels"].to(device)
    reasoning_type = batch["reasoning_type"].to(device)
    
    # Student forward pass
    outputs, type_logits = student(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=teacher_labels  # Use teacher labels (GT + reasoning)
    )
    
    student_logits = outputs.logits
    
    # Loss 1: CE with teacher output (MAIN loss - teacher = GT + reasoning)
    loss_teacher = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_labels.view(-1),
        ignore_index=-100
    )
    
    # Loss 2: GT loss - ONLY for logging (not used in optimization)
    # GT-GUIDED: Teacher already contains GT answer, so GT loss is for monitoring only
    # Must compute in no_grad with SEPARATE forward pass to avoid gradient contamination
    with torch.no_grad():
        # Separate forward pass with GT labels for fair comparison
        gt_outputs, _ = student(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=gt_labels  # Use GT labels for this pass
        )
        loss_gt = F.cross_entropy(
            gt_outputs.logits.view(-1, gt_outputs.logits.size(-1)),
            gt_labels.view(-1),
            ignore_index=-100
        ).item()
    
    # Loss 3: Reasoning type classification
    loss_type = F.cross_entropy(type_logits, reasoning_type)
    
    # Loss 4 & 5: Embedding alignment
    loss_vision = torch.tensor(0.0, device=device)
    loss_text = torch.tensor(0.0, device=device)
    
    if teacher_embeddings is not None:
        with torch.no_grad():
            v_out = student.base_model.vision_encoder(pixel_values=pixel_values).last_hidden_state
            v_feat = v_out.mean(dim=1)
            t_out = student.base_model.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            t_feat = t_out[:, 0, :]
        
        student_vision_emb = F.normalize(v_feat, dim=-1)
        student_text_emb = F.normalize(t_feat, dim=-1)
        teacher_emb_norm = F.normalize(teacher_embeddings.to(device), dim=-1)
        
        # Apply temperature scaling to embedding losses for softer alignment
        temperature = loss_weights.get('temperature', 1.0)
        loss_vision = F.mse_loss(student_vision_emb, teacher_emb_norm) / temperature
        loss_text = F.mse_loss(student_text_emb, teacher_emb_norm) / temperature
    
    # Combined loss - OPTIMIZED: No GT loss (weight=0, already in teacher)
    total_loss = (
        # loss_weights['gt'] * loss_gt  # REMOVED - redundant
        loss_weights['teacher'] * loss_teacher +
        loss_weights['vision'] * loss_vision +
        loss_weights['text'] * loss_text +
        loss_weights['type'] * loss_type
    )
    
    return total_loss, loss_gt, loss_teacher.item(), loss_type.item(), loss_vision.item(), loss_text.item()

# =====================
# PLOT TRAINING CURVES
# =====================
def plot_curves(csv_path, out_png):
    """Plot training/validation curves"""
    try:
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(12, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(df["epoch"], df["train_loss"], label="train_loss", marker='o')
        plt.plot(df["epoch"], df["val_loss"], label="val_loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training/Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        plt.plot(df["epoch"], df["lr"], label="learning_rate", color='green', marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[INFO] Training curves saved to {out_png}")
    except Exception as e:
        print(f"[WARN] Failed to plot curves: {e}")

# =====================
# TRAINING LOOP WITH GRADIENT ACCUMULATION
# =====================
def train():
    # Initialize
    set_seed(42)
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Effective batch size: {cfg.batch_size} * {cfg.accum_steps} = {cfg.batch_size * cfg.accum_steps}")
    print(f"[CONFIG] Reasoning types: {REASONING_TYPES}")
    
    # Load models
    print("\n[INFO] Initializing Student model from scratch...")
    print("[INFO] Using pretrained components:")
    print("  - Vision: BLIP ViT (pretrained on image-text)")
    print("  - Text: PhoBERT (pretrained on Vietnamese)")
    print("  - Decoder: ViT5 (pretrained on Vietnamese T5)")
    print("  - Fusion + Type Classifier: RANDOM INIT (train from scratch)")
    
    base_student = VQAGenModel(
        vision_model_name="Salesforce/blip-vqa-base",
        phobert_dir=os.path.join(cfg.checkpoint_dir, "phobert_tokenizer"),
        vit5_dir=os.path.join(cfg.checkpoint_dir, "vit5_tokenizer")
    )
    
    print("\n[INFO] Creating Type-Aware wrapper...")
    student = TypeAwareVQAModel(base_student).to(device)
    
    print("[INFO] Model architecture:")
    print(f"  - Total parameters: {sum(p.numel() for p in student.parameters())/1e6:.1f}M")
    print(f"  - Trainable parameters: {sum(p.numel() for p in student.parameters() if p.requires_grad)/1e6:.1f}M")
    
    print("[INFO] Loading Teacher model (Qwen2-VL)...")
    teacher_processor = AutoProcessor.from_pretrained(cfg.teacher_model, trust_remote_code=True)
    teacher = AutoModelForVision2Seq.from_pretrained(
        cfg.teacher_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    student_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    
    print("[INFO] Models loaded successfully!")
    
    # Dataset
    dataset = TypeAwareDataset(
        cfg.train_csv, cfg.image_dir, cfg.teacher_jsonl,
        student_processor,
        student.base_model.text_tokenizer,
        student.base_model.decoder_tokenizer,
        max_len=cfg.max_output_len
    )
    
    # Train/Val split
    n_val = int(len(dataset) * cfg.val_ratio)
    train_size = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, n_val])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    
    print(f"\n[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Logging setup
    log_path = os.path.join(cfg.save_dir, cfg.log_csv)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "epoch", "stage", "trainable_params", "lr",
            "train_loss", "train_gt", "train_teacher", "train_type",
            "val_loss", "best_val", "es_counter"
        ]).to_csv(log_path, index=False)
    
    # Training state
    scaler = GradScaler(enabled=cfg.use_amp)
    best_val_loss = float('inf')
    start_epoch = cfg.resume_epoch
    es_counter = 0
    
    # Resume from checkpoint
    auto_checkpoint = os.path.join(cfg.save_dir, "latest_checkpoint.pt")
    if cfg.resume_epoch > 0 and os.path.exists(auto_checkpoint):
        print(f"\n[RESUME] Loading checkpoint from: {auto_checkpoint}")
        checkpoint = torch.load(auto_checkpoint, map_location='cpu')
        student.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        es_counter = checkpoint.get('es_counter', 0)
        print(f"[RESUME] Continuing from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        del checkpoint
        clear_memory()
    
    print(f"\n{'='*70}")
    print("TYPE-AWARE HYBRID DISTILLATION - GT-GUIDED TEACHER")
    print(f"{'='*70}")
    print(f"[TEACHER STRATEGY] GT-Guided: Answer = Ground Truth + Reasoning")
    print(f"  ✅ Answer: 100% correct (forced to GT)")
    print(f"  🧠 Reasoning: Generated by Qwen-7B")
    print(f"  📊 NO separate GT loss (redundant - teacher already contains GT!)")
    print(f"\n[LOSS COMPOSITION]")
    print(f"  🎯 Teacher loss (0.7-0.8): Learn answer + reasoning together")
    print(f"  👁️  Vision/Text alignment (0.15-0.1): Match teacher embeddings")
    print(f"  🏷️  Type classification (0.1): Structured reasoning awareness")
    print(f"\n[TRAINING STRATEGY]")
    print(f"  Stage 1 (Epochs 1-{cfg.stage1_epochs}): Learn reasoning patterns (teacher=0.7)")
    print(f"  Stage 2 (Epochs {cfg.stage1_epochs+1}-{cfg.stage1_epochs+cfg.stage2_epochs}): Master reasoning (teacher=0.75)")
    print(f"  Stage 3 (Epochs {cfg.stage1_epochs+cfg.stage2_epochs+1}-{cfg.num_epochs}): Polish (teacher=0.8)")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, cfg.num_epochs):
        clear_memory()
        
        # Determine training stage
        if epoch < cfg.stage1_epochs:
            current_stage = 1
        elif epoch < cfg.stage1_epochs + cfg.stage2_epochs:
            current_stage = 2
        else:
            current_stage = 3
        
        # Set progressive unfreezing
        set_training_stage(student, current_stage)
        
        # Build optimizer for current stage
        optimizer = build_optimizer(student, cfg)
        
        # Calculate steps for scheduler
        train_steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
        total_train_steps = train_steps_per_epoch * (cfg.num_epochs - epoch)
        warmup_steps = max(1, int(total_train_steps * cfg.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps
        )
        
        # Get loss weights for current stage with temperature
        loss_weights = get_loss_weights(current_stage, temperature=cfg.temperature)
        trainable_params = count_trainable_params(student)
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs} - Stage {current_stage}")
        print(f"Trainable params: {trainable_params/1e6:.2f}M | LR: {current_lr:.2e}")
        print(f"Loss weights: Teacher={loss_weights['teacher']}, Vision={loss_weights['vision']}, "
              f"Text={loss_weights['text']}, Type={loss_weights['type']}")
        print_gpu_memory()
        print(f"{'='*70}")
        
        # TRAINING with gradient accumulation
        student.train()
        total_loss = 0
        total_gt = 0
        total_teacher = 0
        total_type = 0
        total_vision = 0
        total_text = 0
        steps = 0
        
        optimizer.zero_grad(set_to_none=True)
        loop = tqdm(train_loader, desc="Train", leave=False)
        
        for step, batch in enumerate(loop):
            # Get teacher embeddings
            images = batch["image"]
            questions = batch["question"]
            
            try:
                teacher_emb = get_teacher_embeddings(images, questions, teacher, teacher_processor)
            except Exception as e:
                if step == 0:  # Log only first failure to avoid spam
                    print(f"[WARN] Teacher embedding extraction failed: {e}")
                teacher_emb = None
            
            # Forward pass with AMP
            with autocast(dtype=torch.float16, enabled=cfg.use_amp):
                loss, gt, teach_loss, typ, vision, text = compute_type_aware_loss(
                    student, batch, device, teacher_emb, loss_weights
                )
                loss = loss / cfg.accum_steps  # Scale loss for accumulation
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step after accumulation
            if (step + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                steps += 1
            
            # Clear memory periodically
            if (step + 1) % cfg.clear_cache_every_n_steps == 0:
                clear_memory()
            
            # Accumulate metrics (unscaled)
            total_loss += loss.item() * cfg.accum_steps
            total_gt += gt
            total_teacher += teach_loss
            total_type += typ
            total_vision += vision
            total_text += text
            
            loop.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "teacher": f"{teach_loss:.4f}",
                "type": f"{typ:.4f}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        
        # VALIDATION
        student.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                with autocast(dtype=torch.float16, enabled=cfg.use_amp):
                    loss, _, _, _, _, _ = compute_type_aware_loss(student, batch, device, None, loss_weights)
                val_loss += loss.item()
                val_steps += 1
        
        clear_memory()
        avg_val_loss = val_loss / max(val_steps, 1)
        
        # Logging - SIMPLIFIED (no GT loss since teacher=GT+reasoning)
        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"  🎯 Teacher: {total_teacher/len(train_loader):.4f} (answer+reasoning)")
        print(f"  🏷️  Type: {total_type/len(train_loader):.4f}")
        print(f"  👁️  Vision: {total_vision/len(train_loader):.4f}")
        print(f"  📝 Text: {total_text/len(train_loader):.4f}")
        print(f"  [GT loss: {total_gt/len(train_loader):.4f} - NOT USED (weight=0)]")
        
        # QUALITY INDICATOR: Check if teacher loss is reasonable
        if total_teacher/len(train_loader) > 10.0:
            print(f"  ⚠️  WARNING: Teacher loss very high - check data quality!")
        
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        improved = (best_val_loss - avg_val_loss) > cfg.es_min_delta
        if improved:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(student.state_dict(), os.path.join(cfg.save_dir, "vqa_type_aware_best.pt"))
            print(f"[SAVE] ✅ NEW BEST MODEL! Val Loss: {best_val_loss:.4f}")
        else:
            es_counter += 1
        
        # Save checkpoints
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'current_stage': current_stage,
            'es_counter': es_counter
        }
        torch.save(checkpoint_state, os.path.join(cfg.save_dir, "latest_checkpoint.pt"))
        
        # Save last model (lightweight)
        torch.save(student.state_dict(), os.path.join(cfg.save_dir, "vqa_type_aware_last.pt"))
        
        # Stage-wise checkpoints
        if (epoch + 1) == cfg.stage1_epochs:
            torch.save(student.state_dict(), os.path.join(cfg.save_dir, "vqa_stage1_complete.pt"))
            print(f"[CHECKPOINT] Stage 1 complete saved")
        elif (epoch + 1) == cfg.stage1_epochs + cfg.stage2_epochs:
            torch.save(student.state_dict(), os.path.join(cfg.save_dir, "vqa_stage2_complete.pt"))
            print(f"[CHECKPOINT] Stage 2 complete saved")
        
        # Periodic save
        if (epoch + 1) % 5 == 0:
            torch.save(student.state_dict(), os.path.join(cfg.save_dir, f"vqa_type_aware_epoch{epoch+1}.pt"))
            print(f"[CHECKPOINT] Epoch {epoch+1} saved")
        
        # Log to CSV
        row = {
            "epoch": epoch + 1,
            "stage": current_stage,
            "trainable_params": trainable_params,
            "lr": current_lr,
            "train_loss": avg_train_loss,
            "train_gt": total_gt/len(train_loader),
            "train_teacher": total_teacher/len(train_loader),
            "train_type": total_type/len(train_loader),
            "val_loss": avg_val_loss,
            "best_val": best_val_loss,
            "es_counter": es_counter
        }
        df_log = pd.read_csv(log_path)
        df_log.loc[len(df_log)] = row
        df_log.to_csv(log_path, index=False)
        
        print(f"[PROGRESS] Params={trainable_params/1e6:.2f}M | LR={current_lr:.2e} | "
              f"Best={best_val_loss:.4f} | ES={es_counter}/{cfg.es_patience}")
        
        # Early stopping check
        if es_counter >= cfg.es_patience:
            print(f"[INFO] Early stopping triggered after {epoch+1} epochs")
            break
        
        clear_memory()
    
    # Final save
    torch.save(student.state_dict(), os.path.join(cfg.save_dir, "vqa_type_aware_final.pt"))
    
    # Save tokenizers
    try:
        student.base_model.text_tokenizer.save_pretrained(os.path.join(cfg.save_dir, "phobert_tokenizer"))
        student.base_model.decoder_tokenizer.save_pretrained(os.path.join(cfg.save_dir, "vit5_tokenizer"))
        print("[INFO] Tokenizers saved")
    except Exception as e:
        print(f"[WARN] Failed to save tokenizers: {e}")
    
    # Plot training curves
    plot_curves(log_path, os.path.join(cfg.save_dir, cfg.curve_png))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total epochs: {epoch+1}/{cfg.num_epochs}")
    print(f"Logs saved to: {log_path}")

if __name__ == "__main__":
    train()
