"""
Simple Teacher Distillation (NO TYPE CLASSIFICATION)

Architecture:
- VQAGenModel only (BLIP + PhoBERT + ViT5)
- NO type classification head
- Pure sequence generation: learn to generate "Answer: X\nReasoning: Y" from teacher

Loss:
- Cross-entropy with teacher output (answer + reasoning)
- NO type classification loss
- NO multi-task learning confusion

Goal: Test if removing type classification improves performance
"""

import os
import gc
import re
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from transformers import (
    BlipProcessor,
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
# CONFIG
# =====================
@dataclass
class TrainConfig:
    # Paths
    train_csv: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_dir: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    teacher_jsonl: str = "/kaggle/input/teacher-final/teacher_outputs_train.jsonl"
    checkpoint_dir: str = "/kaggle/input/base-model/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working"
    
    # Training hyperparameters
    batch_size: int = 2
    accum_steps: int = 16               # Effective batch = 32
    num_epochs: int = 100               # ✅ TRAIN TỚI EPOCH 100
    val_ratio: float = 0.1
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Learning rates (LOWER for fine-tuning)
    base_lr: float = 1e-5               # ✅ 3e-5 → 1e-5 (giảm 3x)
    vision_lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_ratio: float = 0.05          # ✅ Warmup ngắn hơn
    use_amp: bool = True
    resume_epoch: int = 60              # ✅ RESUME từ epoch 60
    
    # Progressive Training Strategy (FORCE STAGE 1 CHO TẤT CẢ)
    stage1_epochs: int = 100            # ✅ Stage 1 cho tất cả epochs
    stage2_epochs: int = 200            # ✅ Never reach (force stage 1)
    
    # Early stopping (MORE PATIENCE for fine-tuning)
    es_patience: int = 10               # ✅ 6 → 10 (kiên nhẫn hơn)
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log_simple_stage1_continue.csv"  # ✅ Log file riêng
    curve_png: str = "training_curve_simple_stage1_continue.png"
    clear_cache_every_n_steps: int = 20
    
    # Generation
    max_output_len: int = 128

# =====================
# PROGRESSIVE UNFREEZING
# =====================
def set_training_stage(model: VQAGenModel, stage: int):
    """
    Progressive unfreezing:
    Stage 1: Train fusion + decoder only
    Stage 2: + Unfreeze text encoder
    Stage 3: + Unfreeze vision encoder (last block)
    """
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False
    
    # Always train decoder and fusion
    for p in model.decoder.parameters():
        p.requires_grad = True
    if hasattr(model, "fusion"):
        for p in model.fusion.parameters():
            p.requires_grad = True
    
    # Cross-attention and projections if exist
    if hasattr(model, "cross_attn"):
        for p in model.cross_attn.parameters():
            p.requires_grad = True
    if hasattr(model, "vision_proj"):
        for p in model.vision_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "text_proj"):
        for p in model.text_proj.parameters():
            p.requires_grad = True
    
    # Stage 2+: Unfreeze text encoder
    if stage >= 2:
        for p in model.text_encoder.parameters():
            p.requires_grad = True
    
    # Stage 3: Unfreeze vision encoder (last block)
    if stage >= 3:
        try:
            last_block = model.vision_encoder.encoder.layers[-1]
            for p in last_block.parameters():
                p.requires_grad = True
        except Exception:
            for p in model.vision_encoder.parameters():
                p.requires_grad = True

def build_optimizer(model: VQAGenModel, cfg: TrainConfig):
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
# DATASET - SIMPLE (NO TYPE)
# =====================
class SimpleDistillDataset(Dataset):
    def __init__(self, csv_path, image_dir, teacher_jsonl, student_processor, 
                 student_text_tokenizer, student_decoder_tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.student_processor = student_processor
        self.student_text_tokenizer = student_text_tokenizer
        self.student_decoder_tokenizer = student_decoder_tokenizer
        self.max_len = max_len
        
        # Load teacher outputs - USE (img_id, question) AS KEY
        self.teacher_outputs = {}
        teacher_file = self._find_teacher_file(teacher_jsonl)
        if teacher_file and os.path.exists(teacher_file):
            with open(teacher_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    img_id = str(data.get('img_id', data.get('image_id')))
                    question = str(data.get('question', ''))
                    # Key: (img_id, question) để support multiple questions per image
                    key = (img_id, question)
                    self.teacher_outputs[key] = data
            print(f"[INFO] Loaded {len(self.teacher_outputs)} teacher outputs from {teacher_file}")
        else:
            print(f"[WARN] No teacher outputs found. Training with GT only.")
    
    def _find_teacher_file(self, default_path):
        """Find teacher_outputs file"""
        if os.path.exists(default_path):
            return default_path
        
        kaggle_input = "/kaggle/input"
        if not os.path.exists(kaggle_input):
            return None
        
        print(f"[INFO] 🔍 Searching for teacher_outputs_merged* in {kaggle_input}...")
        for root, dirs, files in os.walk(kaggle_input):
            for file in files:
                if file.startswith("teacher_outputs_merged") and file.endswith(".jsonl"):
                    found_path = os.path.join(root, file)
                    print(f"[INFO] ✅ Found: {found_path}")
                    return found_path
        
        return None
        
    def __len__(self):
        return len(self.df)
    
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
        
        # Get teacher output - LOOKUP BY (img_id, question)
        key = (img_id, question)
        teacher_data = self.teacher_outputs.get(key, {})
        teacher_answer = teacher_data.get("teacher_answer", gt_answer)
        teacher_reasoning = teacher_data.get("teacher_reasoning", "")
        
        # GT-GUIDED: Verify teacher_answer matches gt_answer
        if teacher_answer and teacher_answer.strip().lower() != gt_answer.strip().lower():
            teacher_answer = gt_answer
            teacher_reasoning = ""
        
        # Construct full teacher output: "Answer: X\nReasoning: Y"
        if teacher_reasoning:
            teacher_output = f"Answer: {teacher_answer}\nReasoning: {teacher_reasoning}"
        else:
            # Fallback: GT only (no reasoning)
            teacher_output = f"Answer: {gt_answer}"
        
        # Process image + question for student
        inputs = self.student_processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        # Tokenize question
        text_inputs = self.student_text_tokenizer(
            question,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        # Tokenize teacher output (target)
        teacher_inputs = self.student_decoder_tokenizer(
            teacher_output,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        teacher_labels = teacher_inputs["input_ids"].squeeze(0)
        teacher_labels[teacher_labels == self.student_decoder_tokenizer.pad_token_id] = -100
        
        # Tokenize GT answer (for monitoring only)
        gt_inputs = self.student_decoder_tokenizer(
            f"Answer: {gt_answer}",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        gt_labels = gt_inputs["input_ids"].squeeze(0)
        gt_labels[gt_labels == self.student_decoder_tokenizer.pad_token_id] = -100
        
        return {
            "student_pixel_values": pixel_values,
            "student_input_ids": input_ids,
            "student_attention_mask": attention_mask,
            "teacher_labels": teacher_labels,
            "gt_labels": gt_labels,
            "img_id": img_id
        }

# =====================
# SIMPLE LOSS (NO TYPE)
# =====================
def compute_simple_loss(model, batch, device):
    """
    Simple cross-entropy loss with teacher output
    NO type classification
    """
    pixel_values = batch["student_pixel_values"].to(device)
    input_ids = batch["student_input_ids"].to(device)
    attention_mask = batch["student_attention_mask"].to(device)
    gt_labels = batch["gt_labels"].to(device)
    teacher_labels = batch["teacher_labels"].to(device)
    
    # Forward pass
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=teacher_labels
    )
    
    student_logits = outputs.logits
    
    # Loss: CE with teacher output
    loss_teacher = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        teacher_labels.view(-1),
        ignore_index=-100
    )
    
    # GT loss (for monitoring only, not optimized)
    with torch.no_grad():
        gt_outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=gt_labels
        )
        loss_gt = gt_outputs.loss.item()
    
    return loss_teacher, loss_gt

# =====================
# PLOT TRAINING CURVES
# =====================
def plot_curves(csv_path, out_png):
    """Plot training/validation curves"""
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Loss curves
        axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
        axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='s')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss (Simple Distillation)")
        axes[0].legend()
        axes[0].grid(True)
        
        # GT loss monitoring
        axes[1].plot(df["epoch"], df["train_gt_loss"], label="Train GT Loss", marker='o', alpha=0.7)
        axes[1].plot(df["epoch"], df["val_gt_loss"], label="Val GT Loss", marker='s', alpha=0.7)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("GT Loss")
        axes[1].set_title("GT Loss Monitoring")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[INFO] Training curves saved to {out_png}")
    except Exception as e:
        print(f"[WARN] Could not plot curves: {e}")

# =====================
# TRAINING LOOP
# =====================
def train():
    # Initialize
    set_seed(42)
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Effective batch size: {cfg.batch_size} * {cfg.accum_steps} = {cfg.batch_size * cfg.accum_steps}")
    
    # Load model
    print("\n[INFO] Initializing VQAGenModel (NO TYPE CLASSIFICATION)...")
    print("[INFO] Using pretrained components:")
    print("  - Vision: BLIP ViT (pretrained on image-text)")
    print("  - Text: PhoBERT (pretrained on Vietnamese)")
    print("  - Decoder: ViT5 (pretrained on Vietnamese T5)")
    print("  - Fusion: RANDOM INIT (train from scratch)")
    print("  - NO type classifier!")
    
    model = VQAGenModel(
        vision_model_name="Salesforce/blip-vqa-base",
        phobert_dir=os.path.join(cfg.checkpoint_dir, "phobert_tokenizer"),
        vit5_dir=os.path.join(cfg.checkpoint_dir, "vit5_tokenizer")
    ).to(device)
    
    print("[INFO] Model architecture:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
    
    student_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    
    print("[INFO] Models loaded successfully!")
    
    # Dataset
    dataset = SimpleDistillDataset(
        cfg.train_csv, cfg.image_dir, cfg.teacher_jsonl,
        student_processor,
        model.text_tokenizer,
        model.decoder_tokenizer,
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
        with open(log_path, 'w') as f:
            f.write("epoch,stage,train_loss,val_loss,train_gt_loss,val_gt_loss,trainable_params\n")
    
    # Training state
    scaler = GradScaler(enabled=cfg.use_amp)
    best_val_loss = float('inf')
    start_epoch = cfg.resume_epoch
    es_counter = 0
    
    # ==================
    # RESUME FROM CHECKPOINT (EPOCH 60)
    # ==================
    checkpoint_path = "/kaggle/input/60-100/transformers/default/1/latest_checkpoint_simple.pt"
    if cfg.resume_epoch > 0 and os.path.exists(checkpoint_path):
        print(f"\n{'='*70}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_path}")
        print(f"{'='*70}")
        # latest_checkpoint_simple.pt chứa full training state
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model weights loaded from epoch {checkpoint['epoch']}")
        
        # Load training state từ epoch 60
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        es_counter = checkpoint['es_counter']
        
        print(f"   - Resuming from epoch: {start_epoch + 1}")
        print(f"   - Best val loss từ epoch 60: {best_val_loss:.4f}")
        print(f"   - Early stop counter: {es_counter}/{cfg.es_patience}")
        print(f"   - Training strategy: STAGE 1 ONLY (Freeze encoders)")
        print(f"   - Learning rate: {cfg.base_lr:.2e} (3x lower)")
        print(f"{'='*70}\n")
    elif cfg.resume_epoch > 0:
        print(f"\n⚠️  WARNING: resume_epoch={cfg.resume_epoch} but checkpoint not found!")
        print(f"   Looking for: {checkpoint_path}")
        print(f"   Training from scratch instead...\n")
        start_epoch = 0
    
    print(f"\n{'='*70}")
    print("STAGE 1 CONTINUED TRAINING (EPOCH 60 → 100)")
    print(f"{'='*70}")
    print(f"[RESUME] From epoch 60 (accuracy: 31.29%)")
    print(f"[STRATEGY] STAGE 1 ONLY - Train Fusion + Decoder ONLY")
    print(f"[FROZEN] Vision Encoder + Text Encoder")
    print(f"[LR] {cfg.base_lr:.2e} (3x lower than original)")
    print(f"[GOAL] Test if more decoder training improves accuracy")
    print(f"\n[TRAINING STRATEGY]")
    print(f"  Epochs {cfg.resume_epoch+1}-{cfg.num_epochs}: STAGE 1 (Fusion + Decoder ONLY)")
    print(f"  Encoders FROZEN throughout entire training")
    print(f"  Focus: Learn better fusion of pre-trained features")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, cfg.num_epochs):
        # Determine stage
        if epoch < cfg.stage1_epochs:
            stage = 1
        elif epoch < cfg.stage1_epochs + cfg.stage2_epochs:
            stage = 2
        else:
            stage = 3
        
        # Set training stage (progressive unfreezing)
        set_training_stage(model, stage)
        
        # Build optimizer
        optimizer = build_optimizer(model, cfg)
        
        # Scheduler
        num_training_steps = len(train_loader) * (cfg.num_epochs - epoch)
        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        trainable = count_trainable_params(model)
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{cfg.num_epochs} | STAGE {stage} | Trainable: {trainable/1e6:.1f}M")
        print(f"{'='*70}")
        
        # ==================
        # TRAINING
        # ==================
        model.train()
        train_loss_sum = 0.0
        train_gt_loss_sum = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}")
        for step, batch in enumerate(pbar):
            with autocast(enabled=cfg.use_amp):
                loss_teacher, loss_gt = compute_simple_loss(model, batch, device)
                loss = loss_teacher / cfg.accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss_sum += loss_teacher.item()
            train_gt_loss_sum += loss_gt
            
            pbar.set_postfix({
                'loss': f'{loss_teacher.item():.4f}',
                'gt_loss': f'{loss_gt:.4f}'
            })
            
            if (step + 1) % cfg.clear_cache_every_n_steps == 0:
                clear_memory()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_gt_loss = train_gt_loss_sum / len(train_loader)
        
        # ==================
        # VALIDATION
        # ==================
        model.eval()
        val_loss_sum = 0.0
        val_gt_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val E{epoch+1}"):
                loss_teacher, loss_gt = compute_simple_loss(model, batch, device)
                val_loss_sum += loss_teacher.item()
                val_gt_loss_sum += loss_gt
        
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_gt_loss = val_gt_loss_sum / len(val_loader)
        
        print(f"\n[EPOCH {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"           Train GT: {avg_train_gt_loss:.4f} | Val GT: {avg_val_gt_loss:.4f}")
        
        # Logging
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{stage},{avg_train_loss:.6f},{avg_val_loss:.6f},"
                   f"{avg_train_gt_loss:.6f},{avg_val_gt_loss:.6f},{trainable}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss - cfg.es_min_delta:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "vqa_simple_stage1_best.pt"))
            print(f"   ✅ Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            es_counter += 1
            print(f"   ⚠️  No improvement ({es_counter}/{cfg.es_patience})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path_save = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'es_counter': es_counter,
            }, checkpoint_path_save)
            print(f"   💾 Checkpoint saved: checkpoint_epoch_{epoch+1}.pt")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'es_counter': es_counter,
        }
        torch.save(checkpoint, os.path.join(cfg.save_dir, "latest_checkpoint_simple.pt"))
        
        # Early stopping
        if es_counter >= cfg.es_patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
            break
        
        clear_memory()
    
    # Final save
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "vqa_simple_stage1_final.pt"))
    
    # Plot training curves
    plot_curves(log_path, os.path.join(cfg.save_dir, cfg.curve_png))
    
    print(f"\n{'='*70}")
    print("STAGE 1 TRAINING COMPLETE (EPOCH 60 → 100)")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {epoch+1 - cfg.resume_epoch}")
    print(f"Final epoch: {epoch+1}/{cfg.num_epochs}")
    print(f"Logs saved to: {log_path}")
    print(f"\n[NEXT STEP] Run evaluation to check if accuracy > 31.29%")
    print(f"Expected: Small improvement from better fusion learning")

if __name__ == "__main__":
    train()
