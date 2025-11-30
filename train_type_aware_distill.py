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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import (
    BlipProcessor,
    AutoProcessor,
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup
)
from model import VQAGenModel

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
# CONFIG
# =====================
TRAIN_CSV = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
IMAGE_DIR = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
TEACHER_JSONL = "/kaggle/input/teacher/teacher_outputs.jsonl"  # Teacher outputs with reasoning
TEACHER_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
SAVE_DIR = "/kaggle/working"

# Hyperparameters
BATCH_SIZE = 4
EPOCHS = 20
LR = 3e-5
WARMUP_RATIO = 0.1
VAL_RATIO = 0.1

# Progressive Training Strategy
STAGE1_EPOCHS = 8       # Focus on teacher + type (epochs 1-8)
STAGE2_EPOCHS = 8       # Balance all losses (epochs 9-16)
STAGE3_EPOCHS = 4       # Fine-tune with GT (epochs 17-20)

# Loss weights - DYNAMIC per stage
# Stage 1: Learn from teacher + type classification
STAGE1_WEIGHTS = {
    'gt': 0.1,
    'teacher': 0.6,
    'vision': 0.15,
    'text': 0.05,
    'type': 0.1
}

# Stage 2: Balance all
STAGE2_WEIGHTS = {
    'gt': 0.2,
    'teacher': 0.5,
    'vision': 0.15,
    'text': 0.05,
    'type': 0.1
}

# Stage 3: Trust ground truth more
STAGE3_WEIGHTS = {
    'gt': 0.3,
    'teacher': 0.4,
    'vision': 0.15,
    'text': 0.05,
    'type': 0.1
}

# Checkpoint & Resume
RESUME_FROM = None  # Set to checkpoint path to resume
AUTO_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pt")
CLEAR_CACHE_EVERY_N_STEPS = 20

TEMPERATURE = 2.5
MAX_OUTPUT_LEN = 128    # Enough for answer + reasoning

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

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[CONFIG] Device: {device}")
print(f"[CONFIG] Loss weights: GT={ALPHA_GT}, Teacher={ALPHA_TEACHER}, Vision={ALPHA_VISION}, Text={ALPHA_TEXT}, Type={ALPHA_TYPE}")
print(f"[CONFIG] Reasoning types: {REASONING_TYPES}")

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
        with torch.no_grad():
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
# LOAD MODELS
# =====================
print("\n[INFO] Initializing Student model from scratch...")
print("[INFO] Using pretrained components:")
print("  - Vision: BLIP ViT (pretrained on image-text)")
print("  - Text: PhoBERT (pretrained on Vietnamese)")
print("  - Decoder: ViT5 (pretrained on Vietnamese T5)")
print("  - Fusion + Type Classifier: RANDOM INIT (train from scratch)")

base_student = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)

print("\n[INFO] Creating Type-Aware wrapper...")
student = TypeAwareVQAModel(base_student).to(device)

print("[INFO] Model architecture:")
print(f"  - Total parameters: {sum(p.numel() for p in student.parameters())/1e6:.1f}M")
print(f"  - Trainable parameters: {sum(p.numel() for p in student.parameters() if p.requires_grad)/1e6:.1f}M")
print("[INFO] Training from SCRATCH (no base model checkpoint loaded)")

print("[INFO] Loading Teacher model (Qwen2-VL)...")
teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
teacher = AutoModelForVision2Seq.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

student_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

print("[INFO] Models loaded successfully!")

# =====================
# DATASET WITH REASONING TYPE
# =====================
class TypeAwareDataset(Dataset):
    def __init__(self, csv_path, image_dir, teacher_jsonl, student_processor, 
                 student_text_tokenizer, student_decoder_tokenizer, max_len=MAX_OUTPUT_LEN):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.student_processor = student_processor
        self.student_text_tokenizer = student_text_tokenizer
        self.student_decoder_tokenizer = student_decoder_tokenizer
        self.max_len = max_len
        
        # Load teacher outputs
        self.teacher_outputs = {}
        if os.path.exists(teacher_jsonl):
            with open(teacher_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    img_id = data.get('img_id', data.get('image_id'))
                    self.teacher_outputs[str(img_id)] = data
            print(f"[INFO] Loaded {len(self.teacher_outputs)} teacher outputs")
        
    def __len__(self):
        return len(self.df)
    
    def _parse_reasoning_type(self, text):
        """Extract reasoning type from teacher output"""
        import re
        
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
        
        # Get teacher output
        teacher_data = self.teacher_outputs.get(img_id, {})
        teacher_answer = teacher_data.get("teacher_answer", gt_answer)
        teacher_reasoning = teacher_data.get("teacher_reasoning", "")
        teacher_raw = teacher_data.get("teacher_raw", "")
        
        # Construct full teacher output: Answer: X\nReasoning: Y
        if teacher_reasoning:
            teacher_full = f"Answer: {teacher_answer}\nReasoning: {teacher_reasoning}"
        else:
            teacher_full = f"Answer: {teacher_answer}"
        
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
def get_teacher_embeddings(images, questions):
    """Get teacher embeddings for alignment"""
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
def compute_type_aware_loss(student, batch, teacher_embeddings=None, loss_weights=None):
    """
    Multi-task loss with dynamic weights:
    1. CE with ground truth
    2. CE with teacher output
    3. Vision embedding alignment
    4. Text embedding alignment
    5. Reasoning type classification
    
    loss_weights: dict with keys ['gt', 'teacher', 'vision', 'text', 'type']
    """
    if loss_weights is None:
        loss_weights = STAGE2_WEIGHTS
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
        labels=teacher_labels  # Use teacher labels for main task
    )
    
    student_logits = outputs.logits
    
    # Loss 1: CE with teacher output (HIGH weight)
    loss_teacher = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_labels.view(-1),
        ignore_index=-100
    )
    
    # Loss 2: CE with ground truth (LOW weight)
    loss_gt = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        gt_labels.view(-1),
        ignore_index=-100
    )
    
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
        
        loss_vision = F.mse_loss(student_vision_emb, teacher_emb_norm)
        loss_text = F.mse_loss(student_text_emb, teacher_emb_norm)
    
    # Combined loss with dynamic weights
    total_loss = (
        loss_weights['gt'] * loss_gt +
        loss_weights['teacher'] * loss_teacher +
        loss_weights['vision'] * loss_vision +
        loss_weights['text'] * loss_text +
        loss_weights['type'] * loss_type
    )
    
    return total_loss, loss_gt.item(), loss_teacher.item(), loss_type.item(), loss_vision.item(), loss_text.item()

# =====================
# TRAINING LOOP
# =====================
def train():
    # Dataset
    dataset = TypeAwareDataset(
        TRAIN_CSV, IMAGE_DIR, TEACHER_JSONL,
        student_processor,
        student.base_model.text_tokenizer,
        student.base_model.decoder_tokenizer
    )
    
    # Train/Val split
    n_val = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"\n[RESUME] Loading checkpoint from: {RESUME_FROM}")
        checkpoint = torch.load(RESUME_FROM, map_location='cpu')
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[RESUME] Continuing from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        del checkpoint
        clear_memory()
    
    print(f"\n{'='*70}")
    print("TYPE-AWARE HYBRID DISTILLATION - PROGRESSIVE TRAINING")
    print(f"{'='*70}")
    print(f"[STRATEGY]")
    print(f"  Stage 1 (Epochs 1-{STAGE1_EPOCHS}): Focus on Teacher + Type (weights: teacher=0.6)")
    print(f"  Stage 2 (Epochs {STAGE1_EPOCHS+1}-{STAGE1_EPOCHS+STAGE2_EPOCHS}): Balance all losses")
    print(f"  Stage 3 (Epochs {STAGE1_EPOCHS+STAGE2_EPOCHS+1}-{EPOCHS}): Trust GT more (weights: gt=0.3)")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, EPOCHS):
        clear_memory()
        
        # Determine training stage and weights
        if epoch < STAGE1_EPOCHS:
            current_stage = 1
            loss_weights = STAGE1_WEIGHTS
        elif epoch < STAGE1_EPOCHS + STAGE2_EPOCHS:
            current_stage = 2
            loss_weights = STAGE2_WEIGHTS
        else:
            current_stage = 3
            loss_weights = STAGE3_WEIGHTS
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{EPOCHS} - Stage {current_stage}")
        print(f"Loss weights: GT={loss_weights['gt']}, Teacher={loss_weights['teacher']}, Type={loss_weights['type']}")
        print_gpu_memory()
        print(f"{'='*70}")
        
        # TRAINING
        student.train()
        total_loss = 0
        total_gt = 0
        total_teacher = 0
        total_type = 0
        total_vision = 0
        total_text = 0
        
        loop = tqdm(train_loader, desc="Train")
        for batch in loop:
            optimizer.zero_grad()
            
            # Get teacher embeddings
            images = batch["image"]
            questions = batch["question"]
            
            try:
                teacher_emb = get_teacher_embeddings(images, questions)
            except Exception as e:
                print(f"[WARN] Teacher forward failed: {e}")
                teacher_emb = None
            
            # Compute loss with current stage weights
            with autocast():
                loss, gt, teacher, typ, vision, text = compute_type_aware_loss(
                    student, batch, teacher_emb, loss_weights
                )
            
            # Clear memory periodically
            if (loop.n + 1) % CLEAR_CACHE_EVERY_N_STEPS == 0:
                clear_memory()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_gt += gt
            total_teacher += teacher
            total_type += typ
            total_vision += vision
            total_text += text
            
            loop.set_postfix({
                "loss": f"{total_loss/(loop.n+1):.4f}",
                "teacher": f"{teacher:.4f}",
                "type": f"{typ:.4f}"
            })
        
        avg_train_loss = total_loss / len(train_loader)
        
        # VALIDATION
        student.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                loss, _, _, _, _, _ = compute_type_aware_loss(student, batch, None, loss_weights)
                val_loss += loss.item()
        
        clear_memory()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"  - GT: {total_gt/len(train_loader):.4f}")
        print(f"  - Teacher: {total_teacher/len(train_loader):.4f}")
        print(f"  - Type: {total_type/len(train_loader):.4f}")
        print(f"  - Vision: {total_vision/len(train_loader):.4f}")
        print(f"  - Text: {total_text/len(train_loader):.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Auto-save checkpoint for resume
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'current_stage': current_stage
        }
        torch.save(checkpoint_state, AUTO_CHECKPOINT_PATH)
        print(f"[CHECKPOINT] Auto-saved to: latest_checkpoint.pt")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, "vqa_type_aware_best.pt"))
            print(f"[SAVE] ✅ NEW BEST MODEL! Val Loss: {best_val_loss:.4f}")
        
        # Stage-wise checkpoints
        if (epoch + 1) == STAGE1_EPOCHS:
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, "vqa_stage1_complete.pt"))
            print(f"[CHECKPOINT] Stage 1 complete saved")
        elif (epoch + 1) == STAGE1_EPOCHS + STAGE2_EPOCHS:
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, "vqa_stage2_complete.pt"))
            print(f"[CHECKPOINT] Stage 2 complete saved")
        
        # Periodic save
        if (epoch + 1) % 5 == 0:
            torch.save(student.state_dict(), os.path.join(SAVE_DIR, f"vqa_type_aware_epoch{epoch+1}.pt"))
            print(f"[CHECKPOINT] Epoch {epoch+1} saved")
        
        clear_memory()
    
    # Final save
    torch.save(student.state_dict(), os.path.join(SAVE_DIR, "vqa_type_aware_final.pt"))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
