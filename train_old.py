"""
OPTIMIZED SIMPLE FORMAT TRAINING - Best Practices
Cải thiện tối đa từ tất cả các version trước

Key improvements:
1. Simple format: Answer: X / Reasoning: Y (100% parseable)
2. Progressive learning rate scheduling
3. Strong regularization (label smoothing, augmentation, EMA)
4. Smart checkpointing (best, periodic, auto-resume)
5. Comprehensive logging and monitoring
6. Memory-efficient training
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import BlipProcessor
from torchvision import transforms
from model import VQAGenModel
from contextlib import nullcontext
import numpy as np
import gc
import re
from datetime import datetime

# =====================
# CONFIG - OPTIMIZED
# =====================
DATA_PATH = "/kaggle/input/d/dngtrungngha25/teacher-checkpoint-11k/teacher_outputs_simple.jsonl"
SAVE_DIR = "/kaggle/working"

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_best.pt")
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "vqa_final.pt")
LOG_CSV = os.path.join(SAVE_DIR, "training_log.csv")

# Training config - Optimized for Kaggle T4
EPOCHS = 100
LR = 3e-6                    # Balanced LR
BATCH_SIZE = 4               # Larger batch (single forward pass)
VAL_RATIO = 0.15             # 15% validation (balance between size and reliability)
MAX_Q_LEN = 64
MAX_A_LEN = 160
EARLY_STOP_PATIENCE = 20
accum_steps = 2              # Effective batch = 8
WARMUP_EPOCHS = 5

# Resume training
RESUME_FROM = "/kaggle/input/29-11/transformers/default/1/checkpoints/latest_checkpoint.pt"
AUTO_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "latest_checkpoint.pt")

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True
EMPTY_CACHE_EVERY_N_STEPS = 50
PIN_MEMORY = False
GRADIENT_CLIP_VALUE = 0.5
USE_MIXED_PRECISION = True

# EMA for model stability
USE_EMA = True
EMA_DECAY = 0.9995

# Regularization
LABEL_SMOOTHING = 0.15
WEIGHT_DECAY = 3e-4
DROPOUT_RATE = 0.2

# Validation
VALIDATE_FORMAT_EVERY_N_EPOCHS = 5  # Check format quality

SEED = 2025
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# Check PyTorch version for label smoothing support
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
SUPPORTS_LABEL_SMOOTHING = TORCH_VERSION >= (1, 10)

# =====================
# UTILITIES
# =====================
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

# =====================
# EMA
# =====================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# =====================
# FORMAT VALIDATION
# =====================
def parse_output(text: str):
    """Parse Answer: X / Reasoning: Y format"""
    answer = ""
    reasoning = ""
    
    # Regex extraction
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    reasoning_match = re.search(r'Reasoning:\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Fallback: line-based
    if not answer or not reasoning:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            if line.lower().startswith('answer:'):
                answer = line.split(':', 1)[1].strip()
            elif line.lower().startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()
    
    return {
        'answer': answer,
        'reasoning': reasoning,
        'valid': bool(answer and reasoning)
    }

def validate_model_format(model, val_loader, device, tokenizer, vision_processor, num_samples=5):
    """Validate format on samples"""
    model.eval()
    valid_count = 0
    outputs = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            pixel_values = batch["pixel_values"][0:1].to(device)
            input_ids = batch["input_ids"][0:1].to(device)
            attention_mask = batch["attention_mask"][0:1].to(device)
            
            generated = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=160,
                num_beams=1,
                do_sample=False
            )
            
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            parsed = parse_output(text)
            
            if parsed['valid']:
                valid_count += 1
            
            outputs.append({
                'text': text,
                'valid': parsed['valid'],
                'answer': parsed['answer'],
                'reasoning': parsed['reasoning']
            })
    
    return valid_count, num_samples, outputs

# =====================
# DATASET
# =====================
class OptimizedDataset(Dataset):
    def __init__(self, jsonl_path, vision_processor, text_tokenizer, decoder_tokenizer,
                 max_q_len=MAX_Q_LEN, max_a_len=MAX_A_LEN, augment=False):
        
        # Load data
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.augment = augment
        
        # Strong augmentation
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.samples)

    def _safe_load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (255, 255, 255))

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s.get("question", "")
        img = self._safe_load_image(s.get("image_path", ""))
        
        if self.augment and self.augment_transform is not None:
            img = self.augment_transform(img)

        pixel_values = self.vision_processor(img, return_tensors="pt").pixel_values[0]
        q_enc = self.text_tokenizer(q, truncation=True, padding="max_length",
                                    max_length=self.max_q_len, return_tensors="pt")

        # Get simple format target
        target_text = s.get("teacher_simple", "")
        
        # Fallback: construct from answer + reasoning
        if not target_text:
            answer = s.get("teacher_answer", "")
            reasoning = s.get("teacher_reasoning", "")
            target_text = f"Answer: {answer}\nReasoning: {reasoning}"
        
        target_enc = self.decoder_tokenizer(
            target_text, truncation=True, padding="max_length",
            max_length=self.max_a_len, return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids[0],
            "attention_mask": q_enc.attention_mask[0],
            "labels": target_enc.input_ids[0]
        }

# =====================
# LOSS FUNCTION
# =====================
def compute_loss(logits, labels, label_smoothing=0.0):
    """Cross-entropy with optional label smoothing"""
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    if label_smoothing > 0 and SUPPORTS_LABEL_SMOOTHING:
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
    else:
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction='mean'
        )
    
    return loss

# =====================
# LOAD MODEL
# =====================
print("[INFO] Loading VQAGenModel...")
print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] Label smoothing: {'Enabled' if SUPPORTS_LABEL_SMOOTHING else 'Disabled'}")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

clear_memory()

model = VQAGenModel(
    vision_model_name="Salesforce/blip-vqa-base",
    phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
    vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer"
)

if USE_GRADIENT_CHECKPOINTING:
    if hasattr(model.decoder, 'gradient_checkpointing_enable'):
        model.decoder.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled")

model = model.to(device)
print_gpu_memory()

vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Resume training
start_epoch = 0
best_val_loss = float('inf')
early_stop_counter = 0

if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"[INFO] Resuming from: {RESUME_FROM}")
    
    # Add special tokens FIRST to match checkpoint vocab size
    print("[INFO] Adding special tokens to decoder...")
    added_tokens = model.add_special_tokens_and_resize()
    if added_tokens > 0:
        print(f"[INFO] Successfully added {added_tokens} special tokens")
    model = model.to(device)
    
    # Then load checkpoint
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    early_stop_counter = checkpoint.get('early_stop_counter', 0)
    print(f"[INFO] Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    del checkpoint
    clear_memory()
else:
    print("[INFO] Loading pretrained weights...")
    state_dict = torch.load("/kaggle/input/checkpoints/transformers/default/1/checkpoints/best_model.pth", map_location='cpu')
    model.load_state_dict(state_dict)
    print("[INFO] Pretrained weights loaded successfully!")
    del state_dict
    clear_memory()
    
    # Add special tokens AFTER loading pretrained weights
    print("[INFO] Adding special tokens to decoder...")
    added_tokens = model.add_special_tokens_and_resize()
    if added_tokens > 0:
        print(f"[INFO] Successfully added {added_tokens} special tokens")
    model = model.to(device)


# Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def get_lr_multiplier(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
)

# Load optimizer/scheduler state if resuming
if RESUME_FROM and os.path.exists(RESUME_FROM):
    checkpoint = torch.load(RESUME_FROM, map_location='cpu')
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[INFO] Optimizer state restored")
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("[INFO] Scheduler state restored")
    del checkpoint
    clear_memory()

scaler = torch.cuda.amp.GradScaler()

# EMA
ema = None
if USE_EMA:
    ema = EMA(model, decay=EMA_DECAY)
    print(f"[INFO] EMA enabled with decay={EMA_DECAY}")

# =====================
# DATASET & DATALOADERS
# =====================
train_dataset = OptimizedDataset(
    DATA_PATH, vision_processor, 
    model.text_tokenizer, model.decoder_tokenizer,
    augment=True
)
val_dataset = OptimizedDataset(
    DATA_PATH, vision_processor,
    model.text_tokenizer, model.decoder_tokenizer,
    augment=False
)

n_val = max(1, int(len(train_dataset) * VAL_RATIO))
indices = list(range(len(train_dataset)))
random.shuffle(indices)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

train_loader = DataLoader(
    Subset(train_dataset, train_indices), 
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2, pin_memory=PIN_MEMORY,
    prefetch_factor=2, persistent_workers=True
)
val_loader = DataLoader(
    Subset(val_dataset, val_indices), 
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2, pin_memory=PIN_MEMORY,
    prefetch_factor=2, persistent_workers=True
)

print(f"[INFO] Dataset: {len(train_dataset)} total, {len(train_indices)} train, {len(val_indices)} val")
print(f"[INFO] Effective batch size: {BATCH_SIZE * accum_steps}")
print(f"[INFO] Steps per epoch: {len(train_loader) // accum_steps}")

# =====================
# LOGGING
# =====================
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=[
        "epoch", "train_loss", "val_loss", "lr", "format_accuracy"
    ]).to_csv(LOG_CSV, index=False)

# =====================
# TRAINING LOOP
# =====================
print(f"\n{'='*70}")
print(f"OPTIMIZED SIMPLE FORMAT TRAINING")
print(f"{'='*70}")
print(f"[FORMAT] Answer: <text>\\nReasoning: <text>")
print(f"[STRATEGY] Single-task learning (content focus)")
print(f"\n[OPTIMIZATION]")
print(f"  • Base LR: {LR:.2e}")
print(f"  • LR Warmup: {WARMUP_EPOCHS} epochs")
print(f"  • Scheduler: Cosine Annealing")
print(f"  • Mixed Precision: {'Enabled' if USE_MIXED_PRECISION else 'Disabled'}")
print(f"  • Gradient Accumulation: {accum_steps} steps")
print(f"\n[REGULARIZATION]")
print(f"  • Label Smoothing: {LABEL_SMOOTHING}")
print(f"  • Weight Decay: {WEIGHT_DECAY}")
print(f"  • Gradient Clipping: {GRADIENT_CLIP_VALUE}")
print(f"  • Image Augmentation: Strong")
print(f"  • EMA: {'Enabled' if USE_EMA else 'Disabled'} (decay={EMA_DECAY if USE_EMA else 'N/A'})")
print(f"\n[VALIDATION]")
print(f"  • Validation Ratio: {VAL_RATIO*100:.0f}%")
print(f"  • Early Stop Patience: {EARLY_STOP_PATIENCE}")
print(f"  • Format Check: Every {VALIDATE_FORMAT_EVERY_N_EPOCHS} epochs")
print(f"{'='*70}\n")

if start_epoch > 0:
    print(f"[RESUME] Continuing from epoch {start_epoch+1}/{EPOCHS}")
print()

if device == "cuda":
    autocast_ctx = lambda: torch.amp.autocast('cuda')
else:
    autocast_ctx = nullcontext

for epoch in range(start_epoch, EPOCHS):
    clear_memory()
    
    # Warmup LR
    if epoch < WARMUP_EPOCHS:
        lr_mult = get_lr_multiplier(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR * lr_mult
    
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"LR: {current_lr:.2e}" + (f" [Warmup {epoch+1}/{WARMUP_EPOCHS}]" if epoch < WARMUP_EPOCHS else ""))
    print_gpu_memory()
    print(f"{'='*70}")
    
    # TRAINING
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    sum_loss = 0
    loop = tqdm(train_loader, desc=f"Train", leave=False)
    
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with autocast_ctx():
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            if hasattr(outputs, 'logits'):
                loss = compute_loss(outputs.logits, batch["labels"], LABEL_SMOOTHING)
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if ema is not None:
                ema.update()
        
        sum_loss += loss.item() * accum_steps
        
        if (step + 1) % EMPTY_CACHE_EVERY_N_STEPS == 0:
            clear_memory()
        
        loop.set_postfix({
            "loss": f"{sum_loss/(step+1):.4f}",
            "gpu_gb": f"{get_gpu_memory_gb():.1f}"
        })
    
    train_loss = sum_loss / len(train_loader)
    
    # VALIDATION
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val", leave=False):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            if hasattr(outputs, 'logits'):
                loss = compute_loss(outputs.logits, batch["labels"], label_smoothing=0)
            else:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    clear_memory()
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print_gpu_memory()
    
    # Format validation (periodic)
    format_accuracy = 0.0
    if (epoch + 1) % VALIDATE_FORMAT_EVERY_N_EPOCHS == 0:
        print(f"[FORMAT CHECK] Validating output format...")
        valid_count, total_count, sample_outputs = validate_model_format(
            model, val_loader, device, model.decoder_tokenizer, vision_processor, num_samples=5
        )
        format_accuracy = valid_count / total_count * 100
        print(f"[FORMAT CHECK] Valid: {valid_count}/{total_count} ({format_accuracy:.1f}%)")
        if valid_count < total_count:
            print(f"[FORMAT WARN] Some outputs have invalid format!")
            for i, out in enumerate(sample_outputs[:2]):
                if not out['valid']:
                    print(f"  Sample {i+1}: {out['text'][:80]}...")
    
    # Logging
    df = pd.read_csv(LOG_CSV)
    df.loc[len(df)] = [epoch+1, train_loss, val_loss, current_lr, format_accuracy]
    df.to_csv(LOG_CSV, index=False)
    
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()
    
    # CHECKPOINTING
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stop_counter': early_stop_counter,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, AUTO_CHECKPOINT_PATH)
    print(f"[CHECKPOINT] Auto-checkpoint saved: latest_checkpoint.pt")
    
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        early_stop_counter = 0
        
        if ema is not None:
            ema.apply_shadow()
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            ema.restore()
            print(f"[BEST] NEW BEST MODEL (EMA)! Val Loss: {best_val_loss:.4f}")
        else:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[BEST] NEW BEST MODEL! Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"[WARN] No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
    
    # Periodic backups
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, checkpoint_path)
        print(f"[CHECKPOINT] Backup saved: checkpoint_epoch{epoch+1}.pt")
    
    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[STOP] Early stopping at epoch {epoch+1}")
        break

# FINAL SAVE
if ema is not None:
    ema.apply_shadow()
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    ema.restore()
else:
    torch.save(model.state_dict(), FINAL_MODEL_PATH)

print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Total Epochs: {epoch+1}/{EPOCHS}")
print(f"\n[SAVED MODELS]")
print(f"  • Best: {BEST_MODEL_PATH}")
print(f"  • Final: {FINAL_MODEL_PATH}")
print(f"  • Latest: {AUTO_CHECKPOINT_PATH}")
print(f"\n[LOGS]")
print(f"  • Training log: {LOG_CSV}")
print(f"{'='*70}\n")