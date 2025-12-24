"""
ENHANCED THREE-STAGE VQA TRAINING
==================================
Improvements:
1. Dynamic Loss Weighting (adaptive λ per epoch)
2. Label Smoothing + Focal Loss for Type Classification
3. Comprehensive Validation Metrics (BLEU, per-type accuracy, difficulty analysis)
4. Stochastic Weight Averaging (SWA)
5. Better Early Stopping (multi-metric)
6. Data Augmentation (question paraphrasing)
7. Gradient checkpointing for memory efficiency
8. Learning rate warmup + cosine decay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from PIL import Image
from transformers import (
    CLIPProcessor,
    get_cosine_schedule_with_warmup
)
from model_optimal import OptimalVQAModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import os
import gc
import json
import random
import pandas as pd
import numpy as np
from collections import defaultdict

# =====================
# REPRODUCIBILITY
# =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

torch.backends.cudnn.benchmark = True

# =====================
# CONFIG
# =====================
@dataclass
class EnhancedThreeStageConfig:
    # Paths
    train_csv: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_dir: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    teacher_jsonl: str = "/home/nghia-duong/ViVQA_V2/data/teacher_outputs_train.jsonl"
    checkpoint_dir: str = "/kaggle/input/model-base/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working"
    
    # Model
    vision_model: str = "openai/clip-vit-large-patch14"
    hidden_dim: int = 768
    num_fusion_layers: int = 4
    num_heads: int = 12
    dropout: float = 0.1
    use_lora: bool = True
    use_type_routing: bool = True
    use_gradient_checkpointing: bool = True
    
    # Training
    batch_size: int = 2
    accum_steps: int = 16
    num_epochs: int = 120
    val_ratio: float = 0.1
    num_workers: int = 2
    
    # Learning rates
    base_lr: float = 2e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    
    # Progressive unfreezing
    stage1_epochs: int = 60
    stage2_epochs: int = 30
    stage3_epochs: int = 30
    
    # Dynamic loss weights (initial values)
    lambda_type_start: float = 1.0
    lambda_type_end: float = 0.5
    lambda_reasoning_start: float = 0.3
    lambda_reasoning_end: float = 0.2
    lambda_answer_start: float = 3.0
    lambda_answer_end: float = 4.5
    
    # Loss improvements
    label_smoothing: float = 0.1
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_focal_loss_for_type: bool = True
    
    # SWA (Stochastic Weight Averaging)
    use_swa: bool = True
    swa_start_epoch: int = 90
    swa_lr: float = 1e-6
    
    # Generation
    num_beams: int = 4
    length_penalty: float = 1.2
    max_length: int = 256
    
    # Enhanced early stopping
    es_patience: int = 15
    es_min_delta: float = 1e-4
    es_monitor_metrics: list = None  # ['val_loss', 'val_accuracy', 'val_type_accuracy']
    
    # Logging
    log_csv: str = "train_log_enhanced_three_stage.csv"
    clear_cache_every_n_steps: int = 20
    
    # Data augmentation
    use_augmentation: bool = False  # Set True if you have augmentation tools
    augmentation_prob: float = 0.3

    def __post_init__(self):
        if self.es_monitor_metrics is None:
            self.es_monitor_metrics = ['val_loss', 'val_accuracy', 'val_type_accuracy']

# =====================
# FOCAL LOSS
# =====================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [N, C] or [B, L, C]
            labels: [N] or [B, L]
        """
        original_shape = logits.shape
        if len(original_shape) == 3:
            B, L, C = original_shape
            logits = logits.view(-1, C)
            labels = labels.view(-1)
        
        ce_loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Mask ignore_index
        mask = (labels != self.ignore_index)
        focal_loss = focal_loss[mask]
        
        return focal_loss.mean() if focal_loss.numel() > 0 else torch.tensor(0.0, device=logits.device)

# =====================
# ENHANCED THREE-STAGE LOSS
# =====================
class EnhancedThreeStageLoss(nn.Module):
    """
    Enhanced combined loss with:
    - Label smoothing
    - Focal loss for type classification
    - Dynamic weighting
    """
    def __init__(self, 
                 lambda_type=1.0, 
                 lambda_reasoning=0.3, 
                 lambda_answer=3.0,
                 label_smoothing=0.1,
                 use_focal_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 ignore_index=-100):
        super().__init__()
        self.lambda_type = lambda_type
        self.lambda_reasoning = lambda_reasoning
        self.lambda_answer = lambda_answer
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.ignore_index = ignore_index
        
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
    
    def update_weights(self, lambda_type, lambda_reasoning, lambda_answer):
        """Update loss weights dynamically"""
        self.lambda_type = lambda_type
        self.lambda_reasoning = lambda_reasoning
        self.lambda_answer = lambda_answer
    
    def forward(self, logits, labels, tokenizer):
        """
        Args:
            logits: [B, L, V]
            labels: [B, L]
            tokenizer: decoder tokenizer
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Compute per-token loss with label smoothing
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        if self.label_smoothing > 0:
            # Label smoothing (FIXED: Apply masking BEFORE smoothing)
            confidence = 1.0 - self.label_smoothing
            smoothing_value = self.label_smoothing / (vocab_size - 1)
            
            # Mask FIRST to avoid smoothing on padding tokens
            mask = (labels_flat != self.ignore_index)
            
            # Create smoothed one-hot only for valid tokens
            one_hot = torch.zeros_like(logits_flat)
            valid_labels = labels_flat.clone()
            valid_labels[~mask] = 0  # Set invalid to 0 to avoid scatter errors
            
            one_hot.scatter_(1, valid_labels.unsqueeze(1), 1)
            one_hot = one_hot * confidence + smoothing_value
            one_hot[~mask] = 0  # Zero out padding positions
            
            log_probs = F.log_softmax(logits_flat, dim=1)
            loss_per_token = -(one_hot * log_probs).sum(dim=1)
        else:
            loss_per_token = F.cross_entropy(
                logits_flat, labels_flat,
                ignore_index=self.ignore_index,
                reduction='none'
            )
        
        loss_per_token = loss_per_token.view(batch_size, seq_len)
        
        type_losses = []
        reasoning_losses = []
        answer_losses = []
        type_counts = 0
        reasoning_counts = 0
        answer_counts = 0
        
        for b in range(batch_size):
            try:
                valid_mask = (labels[b] != self.ignore_index)
                valid_ids = labels[b][valid_mask]
                valid_loss = loss_per_token[b][valid_mask]
                
                if len(valid_ids) == 0:
                    continue
                
                # Encode delimiters
                reasoning_ids = tokenizer.encode("Reasoning:", add_special_tokens=False)
                answer_ids = tokenizer.encode("\nAnswer:", add_special_tokens=False)
                
                reasoning_pos = None
                answer_pos = None
                
                # Search for delimiters
                for i in range(len(valid_ids) - len(reasoning_ids) + 1):
                    if all(valid_ids[i + j].item() == reasoning_ids[j] for j in range(len(reasoning_ids))):
                        reasoning_pos = i
                        break
                
                for i in range(len(valid_ids) - len(answer_ids) + 1):
                    if all(valid_ids[i + j].item() == answer_ids[j] for j in range(len(answer_ids))):
                        answer_pos = i
                        break
                
                # Split losses
                if reasoning_pos is not None and answer_pos is not None and answer_pos > reasoning_pos:
                    # Type
                    if reasoning_pos > 0:
                        type_loss_val = valid_loss[:reasoning_pos].sum()
                        if self.use_focal_loss:
                            type_logits = logits[b][valid_mask][:reasoning_pos]
                            type_labels = valid_ids[:reasoning_pos]
                            type_loss_val = self.focal_loss(type_logits, type_labels) * reasoning_pos
                        type_losses.append(type_loss_val)
                        type_counts += reasoning_pos
                    
                    # Reasoning
                    reas_len = answer_pos - reasoning_pos
                    if reas_len > 0:
                        reasoning_losses.append(valid_loss[reasoning_pos:answer_pos].sum())
                        reasoning_counts += reas_len
                    
                    # Answer
                    ans_len = len(valid_loss) - answer_pos
                    if ans_len > 0:
                        answer_losses.append(valid_loss[answer_pos:].sum())
                        answer_counts += ans_len
                
                elif answer_pos is not None:
                    # Type + Answer
                    if answer_pos > 0:
                        type_loss_val = valid_loss[:answer_pos].sum()
                        if self.use_focal_loss:
                            type_logits = logits[b][valid_mask][:answer_pos]
                            type_labels = valid_ids[:answer_pos]
                            type_loss_val = self.focal_loss(type_logits, type_labels) * answer_pos
                        type_losses.append(type_loss_val)
                        type_counts += answer_pos
                    
                    ans_len = len(valid_loss) - answer_pos
                    if ans_len > 0:
                        answer_losses.append(valid_loss[answer_pos:].sum())
                        answer_counts += ans_len
                else:
                    # Fallback
                    answer_losses.append(valid_loss.sum())
                    answer_counts += len(valid_loss)
            
            except Exception:
                valid_mask = (labels[b] != self.ignore_index)
                if valid_mask.any():
                    answer_losses.append(loss_per_token[b][valid_mask].sum())
                    answer_counts += valid_mask.sum().item()
        
        # Aggregate
        type_loss = torch.tensor(0.0, device=device)
        reasoning_loss = torch.tensor(0.0, device=device)
        answer_loss = torch.tensor(0.0, device=device)
        
        if type_counts > 0:
            type_loss = sum(type_losses) / type_counts
        if reasoning_counts > 0:
            reasoning_loss = sum(reasoning_losses) / reasoning_counts
        if answer_counts > 0:
            answer_loss = sum(answer_losses) / answer_counts
        
        # Weighted combination
        combined = (
            self.lambda_type * type_loss +
            self.lambda_reasoning * reasoning_loss +
            self.lambda_answer * answer_loss
        )
        
        return combined, type_loss, reasoning_loss, answer_loss

# =====================
# DYNAMIC LOSS WEIGHTING
# =====================
def get_dynamic_loss_weights(epoch, total_epochs, cfg):
    """
    Compute dynamic loss weights based on training progress.
    Early: focus on type + reasoning structure
    Late: focus on answer accuracy
    """
    progress = epoch / total_epochs
    
    lambda_type = cfg.lambda_type_start + (cfg.lambda_type_end - cfg.lambda_type_start) * progress
    lambda_reasoning = cfg.lambda_reasoning_start + (cfg.lambda_reasoning_end - cfg.lambda_reasoning_start) * progress
    lambda_answer = cfg.lambda_answer_start + (cfg.lambda_answer_end - cfg.lambda_answer_start) * progress
    
    return lambda_type, lambda_reasoning, lambda_answer

# =====================
# ENHANCED DATASET
# =====================
class EnhancedThreeStageDataset(Dataset):
    """Enhanced dataset with optional augmentation"""
    def __init__(self, csv_path, image_dir, teacher_jsonl, vision_processor,
                 text_tokenizer, decoder_tokenizer, max_len=256, 
                 use_augmentation=False, augmentation_prob=0.3):
        
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        
        # Load teacher outputs
        self.teacher_outputs = {}
        if os.path.exists(teacher_jsonl):
            with open(teacher_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    key = (str(data['img_id']), str(data['question']))
                    self.teacher_outputs[key] = data
            print(f"[INFO] Loaded {len(self.teacher_outputs)} teacher outputs")
    
    def augment_question(self, question):
        """Simple question augmentation (placeholder)"""
        if not self.use_augmentation or random.random() > self.augmentation_prob:
            return question
        
        # Simple augmentation: add/remove punctuation
        if random.random() > 0.5:
            question = question.rstrip('?') + '?'
        
        return question
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['img_id'])
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        question_original = str(row["question"])
        gt_answer = str(row["answer"])
        
        # Augment question (for input only)
        question = self.augment_question(question_original)

        # Lookup with ORIGINAL question (FIXED: use original for lookup)
        key = (img_id, question_original)
        teacher_data = self.teacher_outputs.get(key, {})
        teacher_answer = teacher_data.get("teacher_answer", gt_answer)
        teacher_reasoning = teacher_data.get("teacher_reasoning", "")
        reasoning_type = teacher_data.get("reasoning_type", "OBJECT")
        
        # GT-guided correction (FIXED: discard reasoning when correcting answer)
        if teacher_answer and teacher_answer.strip().lower() != gt_answer.strip().lower():
            teacher_answer = gt_answer
            teacher_reasoning = ""  # CRITICAL: Discard reasoning when answer is corrected
        
        # Three-stage format (reasoning only if valid)
        if teacher_reasoning and teacher_answer:
            target = f"Type: {reasoning_type}\nReasoning: {teacher_reasoning}\nAnswer: {teacher_answer}"
        elif teacher_answer:
            target = f"Type: {reasoning_type}\nAnswer: {teacher_answer}"
        else:
            target = f"Answer: {gt_answer}"
        
        # Process image
        vision_inputs = self.vision_processor(images=image, return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].squeeze(0)
        
        # Tokenize question
        text_inputs = self.text_tokenizer(
            question, max_length=64, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        # Tokenize target
        target_inputs = self.decoder_tokenizer(
            target, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        labels = target_inputs["input_ids"].squeeze(0)
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "img_id": img_id,
            "question": question,
            "gt_answer": gt_answer,
            "reasoning_type": reasoning_type,
            "teacher_reasoning": teacher_reasoning,
            "teacher_answer": teacher_answer,
        }

# =====================
# EVALUATION METRICS
# =====================
def extract_answer_from_three_stage(prediction: str) -> str:
    """Extract answer from three-stage format"""
    prediction = prediction.strip()
    if "Answer:" in prediction:
        answer_part = prediction.split("Answer:")[-1].strip()
        answer = answer_part.split("\n")[0].strip()
        return answer
    lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    return lines[-1] if lines else prediction

def extract_type_from_three_stage(prediction: str) -> str:
    """Extract type from three-stage format"""
    prediction = prediction.strip()
    if "Type:" in prediction:
        type_part = prediction.split("Type:")[-1].strip()
        type_line = type_part.split("\n")[0].strip()
        for delimiter in ["Reasoning:", "Answer:"]:
            if delimiter in type_line:
                type_line = type_line.split(delimiter)[0].strip()
        return type_line
    return ""

def extract_reasoning_from_three_stage(prediction: str) -> str:
    """Extract reasoning from three-stage format"""
    prediction = prediction.strip()
    if "Reasoning:" in prediction and "Answer:" in prediction:
        reasoning_part = prediction.split("Reasoning:")[1].split("Answer:")[0].strip()
        return reasoning_part
    return ""

def compute_bleu_score(prediction: str, reference: str) -> float:
    """Compute BLEU score for reasoning quality"""
    if not prediction or not reference:
        return 0.0
    
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    try:
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0

def analyze_by_difficulty(predictions, ground_truths):
    """Analyze accuracy by answer length (difficulty proxy)"""
    easy_correct = 0
    medium_correct = 0
    hard_correct = 0
    easy_total = 0
    medium_total = 0
    hard_total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        gt_words = len(gt.split())
        correct = (pred.lower().strip() == gt.lower().strip())
        
        if gt_words <= 3:
            easy_total += 1
            if correct:
                easy_correct += 1
        elif gt_words <= 8:
            medium_total += 1
            if correct:
                medium_correct += 1
        else:
            hard_total += 1
            if correct:
                hard_correct += 1
    
    results = {}
    if easy_total > 0:
        results['easy_acc'] = 100.0 * easy_correct / easy_total
    if medium_total > 0:
        results['medium_acc'] = 100.0 * medium_correct / medium_total
    if hard_total > 0:
        results['hard_acc'] = 100.0 * hard_correct / hard_total
    
    return results

def compute_per_type_accuracy(predictions, types, ground_truths):
    """Compute accuracy per reasoning type"""
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    
    for pred, typ, gt in zip(predictions, types, ground_truths):
        type_total[typ] += 1
        if pred.lower().strip() == gt.lower().strip():
            type_correct[typ] += 1
    
    type_acc = {}
    for typ in type_total:
        type_acc[typ] = 100.0 * type_correct[typ] / type_total[typ]
    
    return type_acc

# =====================
# TRAINING FUNCTIONS
# =====================
def set_training_stage(model, stage: int):
    """Progressive unfreezing"""
    for p in model.parameters():
        p.requires_grad = False
    
    # Always train decoder, fusion, projections
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = True
    if hasattr(model, "decoder_input_proj"):
        for p in model.decoder_input_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "type_router"):
        for p in model.type_router.parameters():
            p.requires_grad = True
    
    # Stage 2: Unfreeze text encoder last 2 layers
    if stage >= 2:
        try:
            if hasattr(model.text_encoder, "encoder"):
                last_layers = model.text_encoder.encoder.layer[-2:]
                for layer in last_layers:
                    for p in layer.parameters():
                        p.requires_grad = True
        except:
            pass
    
    # Stage 3: Unfreeze vision encoder last 2 layers
    if stage >= 3:
        try:
            if hasattr(model.vision_encoder, "clip"):
                if hasattr(model.vision_encoder.clip, "vision_model"):
                    last_layers = model.vision_encoder.clip.vision_model.encoder.layers[-2:]
                    for layer in last_layers:
                        for p in layer.parameters():
                            p.requires_grad = True
        except:
            pass

def compute_loss(model, batch, device, criterion):
    """Compute loss"""
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    logits = outputs.logits
    combined_loss, type_loss, reasoning_loss, answer_loss = criterion(
        logits, labels, model.decoder_tokenizer
    )
    
    return combined_loss, type_loss, reasoning_loss, answer_loss

class MultiMetricEarlyStopping:
    """Early stopping based on multiple metrics"""
    def __init__(self, patience=15, min_delta=1e-4, monitor_metrics=None):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metrics = monitor_metrics or ['val_loss']
        self.counters = {m: 0 for m in self.monitor_metrics}
        self.best_values = {m: float('inf') if 'loss' in m else 0.0 for m in self.monitor_metrics}
    
    def step(self, metrics):
        """
        Args:
            metrics: dict with keys matching monitor_metrics
        Returns:
            True if should stop, False otherwise
        """
        improved = False
        
        for metric_name in self.monitor_metrics:
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            best_value = self.best_values[metric_name]
            
            if 'loss' in metric_name:
                # Lower is better
                if current_value < best_value - self.min_delta:
                    self.best_values[metric_name] = current_value
                    self.counters[metric_name] = 0
                    improved = True
                else:
                    self.counters[metric_name] += 1
            else:
                # Higher is better
                if current_value > best_value + self.min_delta:
                    self.best_values[metric_name] = current_value
                    self.counters[metric_name] = 0
                    improved = True
                else:
                    self.counters[metric_name] += 1
        
        # Stop if ALL metrics have not improved
        all_stagnant = all(c >= self.patience for c in self.counters.values())
        
        return all_stagnant, improved

# =====================
# MAIN TRAINING
# =====================
def train():
    set_seed(42)
    cfg = EnhancedThreeStageConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("="*70)
    print("ENHANCED THREE-STAGE VQA TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Vision: {cfg.vision_model}")
    print(f"\n🚀 ENHANCEMENTS:")
    print(f"  ✓ Dynamic Loss Weighting")
    print(f"  ✓ Label Smoothing ({cfg.label_smoothing})")
    print(f"  ✓ Focal Loss for Type (α={cfg.focal_loss_alpha}, γ={cfg.focal_loss_gamma})")
    print(f"  ✓ SWA from epoch {cfg.swa_start_epoch}")
    print(f"  ✓ Multi-Metric Early Stopping")
    print(f"  ✓ Comprehensive Validation Metrics")
    print(f"\n📊 INITIAL LOSS WEIGHTS:")
    print(f"  λ_type: {cfg.lambda_type_start} → {cfg.lambda_type_end}")
    print(f"  λ_reasoning: {cfg.lambda_reasoning_start} → {cfg.lambda_reasoning_end}")
    print(f"  λ_answer: {cfg.lambda_answer_start} → {cfg.lambda_answer_end}")
    print("="*70 + "\n")
    
    # Load model
    model = OptimalVQAModel(
        vision_model_name=cfg.vision_model,
        phobert_dir=os.path.join(cfg.checkpoint_dir, "phobert_tokenizer"),
        vit5_dir=os.path.join(cfg.checkpoint_dir, "vit5_tokenizer"),
        hidden_dim=cfg.hidden_dim,
        num_fusion_layers=cfg.num_fusion_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_lora=cfg.use_lora,
        use_type_routing=cfg.use_type_routing
    ).to(device)
    
    # Gradient checkpointing
    if cfg.use_gradient_checkpointing:
        try:
            if hasattr(model.vision_encoder, 'gradient_checkpointing_enable'):
                model.vision_encoder.gradient_checkpointing_enable()
            if hasattr(model.text_encoder, 'gradient_checkpointing_enable'):
                model.text_encoder.gradient_checkpointing_enable()
            if hasattr(model.decoder, 'gradient_checkpointing_enable'):
                model.decoder.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        except:
            print("⚠ Gradient checkpointing not available")
    
    vision_processor = CLIPProcessor.from_pretrained(cfg.vision_model)
    
    # Loss function
    enhanced_loss = EnhancedThreeStageLoss(
        lambda_type=cfg.lambda_type_start,
        lambda_reasoning=cfg.lambda_reasoning_start,
        lambda_answer=cfg.lambda_answer_start,
        label_smoothing=cfg.label_smoothing,
        use_focal_loss=cfg.use_focal_loss_for_type,
        focal_alpha=cfg.focal_loss_alpha,
        focal_gamma=cfg.focal_loss_gamma
    )
    
    # Dataset
    full_dataset = EnhancedThreeStageDataset(
        cfg.train_csv, cfg.image_dir, cfg.teacher_jsonl,
        vision_processor, model.text_tokenizer, model.decoder_tokenizer,
        max_len=cfg.max_length,
        use_augmentation=cfg.use_augmentation,
        augmentation_prob=cfg.augmentation_prob
    )
    
    # Split
    n_val = int(len(full_dataset) * cfg.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - n_val, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
    
    # Optimizer
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    
    # SWA
    swa_model = None
    swa_scheduler = None
    if cfg.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.swa_lr)
    
    # Scheduler
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Early stopping
    early_stopper = MultiMetricEarlyStopping(
        patience=cfg.es_patience,
        min_delta=cfg.es_min_delta,
        monitor_metrics=cfg.es_monitor_metrics
    )
    
    # Logging
    log_path = os.path.join(cfg.save_dir, cfg.log_csv)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,stage,train_loss,train_type_loss,train_reasoning_loss,train_answer_loss,"
                   "val_loss,val_accuracy,val_type_accuracy,bleu_score,"
                   "easy_acc,medium_acc,hard_acc,lambda_type,lambda_reasoning,lambda_answer\n")
    
    # Resume checkpoint
    best_val_loss = float('inf')
    start_epoch = 0
    resume_checkpoint = os.path.join(cfg.save_dir, "latest_checkpoint_enhanced_three_stage.pt")
    
    if os.path.exists(resume_checkpoint):
        print(f"\n{'='*70}")
        print(f"🔄 RESUMING FROM CHECKPOINT")
        print(f"{'='*70}")
        
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if cfg.use_swa and 'swa_model_state_dict' in checkpoint:
            swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        
        print(f"✅ Loaded from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        print(f"{'='*70}\n")
    
    # Training loop
    scaler = GradScaler()
    
    for epoch in range(start_epoch, cfg.num_epochs):
        # Determine stage
        if epoch < cfg.stage1_epochs:
            stage = 1
        elif epoch < cfg.stage1_epochs + cfg.stage2_epochs:
            stage = 2
        else:
            stage = 3
        
        set_training_stage(model, stage)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        
        # Dynamic loss weights
        lambda_type, lambda_reasoning, lambda_answer = get_dynamic_loss_weights(epoch, cfg.num_epochs, cfg)
        enhanced_loss.update_weights(lambda_type, lambda_reasoning, lambda_answer)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Stage {stage} | Trainable: {trainable_params:.1f}M")
        print(f"Loss weights: λ_type={lambda_type:.3f}, λ_reasoning={lambda_reasoning:.3f}, λ_answer={lambda_answer:.3f}")
        print(f"{'='*70}")
        
        # ==================
        # TRAINING
        # ==================
        model.train()
        train_loss = 0
        train_type_loss = 0
        train_reasoning_loss = 0
        train_answer_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}", ncols=100, leave=False)
        
        for step, batch in enumerate(pbar):
            with autocast():
                loss, t_loss, r_loss, a_loss = compute_loss(model, batch, device, enhanced_loss)
                loss = loss / cfg.accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # SWA update
                if cfg.use_swa and epoch >= cfg.swa_start_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
                
                optimizer.zero_grad()
            
            train_loss += loss.item() * cfg.accum_steps
            train_type_loss += t_loss.item() if t_loss.item() > 0 else 0
            train_reasoning_loss += r_loss.item() if r_loss.item() > 0 else 0
            train_answer_loss += a_loss.item() if a_loss.item() > 0 else 0
            
            if step % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item() * cfg.accum_steps:.4f}'})
            
            if (step + 1) % cfg.clear_cache_every_n_steps == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_t_loss = train_type_loss / len(train_loader)
        avg_train_r_loss = train_reasoning_loss / len(train_loader)
        avg_train_a_loss = train_answer_loss / len(train_loader)
        
        # ==================
        # VALIDATION
        # ==================
        eval_model = swa_model if (cfg.use_swa and epoch >= cfg.swa_start_epoch) else model
        eval_model.eval()
        
        val_loss = 0
        val_correct = 0
        val_correct_type = 0
        val_total = 0
        all_bleu_scores = []
        all_predictions = []
        all_ground_truths = []
        all_types = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val E{epoch+1}", ncols=100, leave=False)):
                # Compute loss
                loss, _, _, _ = compute_loss(eval_model, batch, device, enhanced_loss)
                val_loss += loss.item()
                
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Generate
                output_ids = eval_model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.max_length,
                    num_beams=cfg.num_beams,
                    length_penalty=cfg.length_penalty,
                    early_stopping=True
                )
                predictions = eval_model.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                # Evaluate
                for i, pred in enumerate(predictions):
                    gt_answer = batch['gt_answer'][i]
                    gt_type = batch['reasoning_type'][i]
                    teacher_reasoning = batch['teacher_reasoning'][i]
                    
                    pred_answer = extract_answer_from_three_stage(pred)
                    pred_type = extract_type_from_three_stage(pred)
                    pred_reasoning = extract_reasoning_from_three_stage(pred)
                    
                    # Normalize
                    pred_answer_norm = ' '.join(pred_answer.lower().split())
                    gt_answer_norm = ' '.join(gt_answer.lower().split())
                    pred_type_norm = pred_type.upper().strip()
                    gt_type_norm = gt_type.upper().strip()
                    
                    # Check correctness
                    if pred_answer_norm == gt_answer_norm:
                        val_correct += 1
                    if pred_type_norm == gt_type_norm:
                        val_correct_type += 1
                    
                    # BLEU score
                    if teacher_reasoning and pred_reasoning:
                        bleu = compute_bleu_score(pred_reasoning, teacher_reasoning)
                        all_bleu_scores.append(bleu)
                    
                    all_predictions.append(pred_answer_norm)
                    all_ground_truths.append(gt_answer_norm)
                    all_types.append(gt_type_norm)
                    
                    val_total += 1
                    
                    # Debug
                    should_debug = (epoch <= 10) or (epoch % 5 == 0)
                    if batch_idx == 0 and i < 2 and should_debug:
                        print(f"\n--- Sample {i+1} ---")
                        print(f"Question: {batch['question'][i]}")
                        print(f"\nPrediction:\n{pred}")
                        print(f"\nExtracted Answer: '{pred_answer}' | GT: '{gt_answer}' | {'✓' if pred_answer_norm == gt_answer_norm else '✗'}")
                        print(f"Extracted Type: '{pred_type}' | GT: '{gt_type}' | {'✓' if pred_type_norm == gt_type_norm else '✗'}")
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        val_type_accuracy = 100.0 * val_correct_type / val_total if val_total > 0 else 0.0
        avg_bleu = np.mean(all_bleu_scores) if all_bleu_scores else 0.0
        
        # Difficulty analysis
        difficulty_results = analyze_by_difficulty(all_predictions, all_ground_truths)
        easy_acc = difficulty_results.get('easy_acc', 0.0)
        medium_acc = difficulty_results.get('medium_acc', 0.0)
        hard_acc = difficulty_results.get('hard_acc', 0.0)
        
        # Per-type accuracy
        type_accuracies = compute_per_type_accuracy(all_predictions, all_types, all_ground_truths)
        
        # ==================
        # LOGGING
        # ==================
        print(f"\n[EPOCH {epoch+1}]")
        print(f"  Train Loss: {avg_train_loss:.4f} (T:{avg_train_t_loss:.4f}, R:{avg_train_r_loss:.4f}, A:{avg_train_a_loss:.4f})")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}% | Type Acc: {val_type_accuracy:.2f}% | BLEU: {avg_bleu:.3f}")
        print(f"  Difficulty: Easy={easy_acc:.2f}%, Medium={medium_acc:.2f}%, Hard={hard_acc:.2f}%")
        print(f"  Per-Type Acc: {type_accuracies}")
        
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{stage},{avg_train_loss:.6f},{avg_train_t_loss:.6f},{avg_train_r_loss:.6f},{avg_train_a_loss:.6f},"
                   f"{avg_val_loss:.6f},{val_accuracy:.2f},{val_type_accuracy:.2f},{avg_bleu:.4f},"
                   f"{easy_acc:.2f},{medium_acc:.2f},{hard_acc:.2f},"
                   f"{lambda_type:.4f},{lambda_reasoning:.4f},{lambda_answer:.4f}\n")
        
        # Early stopping
        metrics = {
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_type_accuracy': val_type_accuracy
        }
        should_stop, improved = early_stopper.step(metrics)
        
        # Save best model
        if improved or avg_val_loss < best_val_loss:
            best_val_loss = min(avg_val_loss, best_val_loss)
            model_to_save = swa_model if (cfg.use_swa and epoch >= cfg.swa_start_epoch) else model
            torch.save(model_to_save.state_dict(), os.path.join(cfg.save_dir, "best_enhanced_three_stage_model.pt"))
            print(f"✅ Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            print(f"⚠️  No improvement (ES counters: {early_stopper.counters})")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if cfg.use_swa and swa_model is not None:
            checkpoint['swa_model_state_dict'] = swa_model.state_dict()
        
        torch.save(checkpoint, os.path.join(cfg.save_dir, "latest_checkpoint_enhanced_three_stage.pt"))
        
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(cfg.save_dir, f"checkpoint_enhanced_epoch_{epoch+1}.pt"))
        
        # Early stopping
        if should_stop:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final SWA batch norm update
    if cfg.use_swa and swa_model is not None:
        print("\n" + "="*70)
        print("🔄 Updating SWA batch norm statistics...")
        print("="*70)
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            torch.save(swa_model.state_dict(), os.path.join(cfg.save_dir, "swa_final_model.pt"))
            print("✅ SWA model saved!")
        except Exception as e:
            print(f"⚠️  SWA batch norm update failed: {e}")
    
    print("\n" + "="*70)
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*70)

if __name__ == "__main__":
    train()