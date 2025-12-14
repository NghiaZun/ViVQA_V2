"""
Advanced Teacher Distillation with Cross-Attention Fusion

Architecture:
- VQAAdvancedModel (CLIP + PhoBERT + BimodalCrossAttention + VietT5)
- Bidirectional cross-attention fusion
- Multi-layer fusion with residual connections

Loss:
- Cross-entropy with teacher output (answer + reasoning)
- NO type classification loss
- Pure sequence generation

Goal: Leverage state-of-the-art architecture for better Vietnamese VQA
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
    CLIPProcessor,
    get_cosine_schedule_with_warmup
)
from model_advanced import VQAAdvancedModel, VQALightweightModel

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
    teacher_jsonl: str = "/kaggle/input/8-12-teacher/teacher_outputs_train.jsonl"
    checkpoint_dir: str = "/kaggle/input/model-base/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working"
    
    # Model architecture
    use_lightweight: bool = False       # True = VQALightweightModel, False = VQAAdvancedModel
    hidden_dim: int = 768               # 768 for Advanced, 512 for Lightweight
    num_fusion_layers: int = 2          # Number of cross-attention fusion layers
    num_heads: int = 8                  # Multi-head attention heads
    fusion_dropout: float = 0.1         # Dropout in fusion layers
    
    # Training hyperparameters
    batch_size: int = 2
    accum_steps: int = 16               # Effective batch = 32
    num_epochs: int = 100
    val_ratio: float = 0.1
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Learning rates
    base_lr: float = 2e-5               # Base learning rate
    vision_lr: float = 1e-5             # Lower LR for vision encoder
    fusion_lr: float = 3e-5             # Higher LR for fusion (training from scratch)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_ratio: float = 0.05
    use_amp: bool = True
    
    # Progressive Training Strategy
    stage1_epochs: int = 60             # Stage 1: Fusion + Decoder only
    stage2_epochs: int = 20             # Stage 2: + Text encoder
    stage3_epochs: int = 20             # Stage 3: + Vision encoder (last layers)
    
    # Early stopping
    es_patience: int = 10
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log_advanced.csv"
    curve_png: str = "training_curve_advanced.png"
    clear_cache_every_n_steps: int = 20
    
    # Generation
    max_output_len: int = 128

# =====================
# PROGRESSIVE UNFREEZING FOR ADVANCED MODEL
# =====================
def set_training_stage(model, stage: int):
    """
    Progressive unfreezing for VQAAdvancedModel (SMART STRATEGY):
    
    Stage 1 (Epochs 1-60): 
        - Train: Fusion + Decoder ONLY
        - Freeze: Vision encoder (CLIP) + Text encoder (PhoBERT) 
        - Rationale: Learn to combine pretrained features first
    
    Stage 2 (Epochs 61-80): 
        - Train: Fusion + Decoder + LAST 2 LAYERS of PhoBERT
        - Freeze: Vision encoder + PhoBERT (other layers)
        - Rationale: Fine-tune text encoder slightly for Vietnamese VQA
    
    Stage 3 (Epochs 81-100): 
        - Train: Fusion + Decoder + LAST 2 LAYERS of PhoBERT + LAST 2 LAYERS of CLIP
        - Freeze: Other layers of both encoders
        - Rationale: Minimal fine-tuning of vision encoder (pretrained features are good!)
    """
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False
    
    # ========================================
    # ALWAYS TRAINABLE (All stages)
    # ========================================
    
    # 1. Decoder (generate answer in Vietnamese)
    for p in model.decoder.parameters():
        p.requires_grad = True
    
    # 2. Fusion layers (train from scratch)
    if hasattr(model, "fusion_layers"):
        for fusion_layer in model.fusion_layers:
            for p in fusion_layer.parameters():
                p.requires_grad = True
    elif hasattr(model, "fusion"):
        for p in model.fusion.parameters():
            p.requires_grad = True
    
    # 3. Projection layers (adapt pretrained features)
    if hasattr(model, "vision_proj"):
        for p in model.vision_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "text_proj"):
        for p in model.text_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "decoder_input_proj"):
        for p in model.decoder_input_proj.parameters():
            p.requires_grad = True
    
    # ========================================
    # STAGE 2: Unfreeze LAST 2 layers of PhoBERT
    # ========================================
    if stage >= 2:
        try:
            # PhoBERT có 12 layers, unfreeze 2 layers cuối (layer 10, 11)
            if hasattr(model.text_encoder, "encoder"):
                if hasattr(model.text_encoder.encoder, "layer"):
                    # RoBERTa/BERT structure
                    last_text_layers = model.text_encoder.encoder.layer[-2:]
                    for layer in last_text_layers:
                        for p in layer.parameters():
                            p.requires_grad = True
                    print(f"[INFO] Stage {stage}: Unfroze LAST 2 layers of PhoBERT")
                else:
                    # Fallback: unfreeze last 2 transformer layers
                    layers = list(model.text_encoder.encoder.children())
                    for layer in layers[-2:]:
                        for p in layer.parameters():
                            p.requires_grad = True
            else:
                # If structure is different, unfreeze pooler only
                if hasattr(model.text_encoder, "pooler"):
                    for p in model.text_encoder.pooler.parameters():
                        p.requires_grad = True
                print(f"[WARN] Stage {stage}: Could not find PhoBERT layers, unfroze pooler only")
        except Exception as e:
            print(f"[WARN] Stage {stage}: Could not unfreeze PhoBERT layers: {e}")
    
    # ========================================
    # STAGE 3: Unfreeze LAST 2 layers of CLIP
    # ========================================
    if stage >= 3:
        try:
            # CLIP ViT có 12 transformer layers, unfreeze 2 layers cuối (layer 10, 11)
            if hasattr(model.vision_encoder, "vision_model"):
                # CLIP structure: vision_model.encoder.layers
                if hasattr(model.vision_encoder.vision_model, "encoder"):
                    last_vision_layers = model.vision_encoder.vision_model.encoder.layers[-2:]
                    for layer in last_vision_layers:
                        for p in layer.parameters():
                            p.requires_grad = True
                    print(f"[INFO] Stage {stage}: Unfroze LAST 2 layers of CLIP ViT")
                else:
                    print(f"[WARN] Stage {stage}: CLIP structure not as expected")
            else:
                # Fallback: unfreeze last blocks
                encoder_layers = list(model.vision_encoder.children())
                for layer in encoder_layers[-2:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                print(f"[WARN] Stage {stage}: Unfroze last 2 modules of vision encoder")
        except Exception as e:
            print(f"[WARN] Stage {stage}: Could not unfreeze CLIP layers: {e}")
            # Don't unfreeze anything if failed - better safe than sorry!

def build_optimizer(model, cfg: TrainConfig):
    """Build optimizer with different LR for vision, fusion, text, decoder"""
    vision_params = []
    fusion_params = []
    text_params = []
    decoder_params = []
    other_params = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        
        if "vision_encoder" in n:
            vision_params.append(p)
        elif "fusion" in n:
            fusion_params.append(p)
        elif "text_encoder" in n:
            text_params.append(p)
        elif "decoder" in n:
            decoder_params.append(p)
        else:
            other_params.append(p)
    
    param_groups = []
    if vision_params:
        param_groups.append({"params": vision_params, "lr": cfg.vision_lr})
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": cfg.fusion_lr})
    if text_params:
        param_groups.append({"params": text_params, "lr": cfg.base_lr})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": cfg.base_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": cfg.base_lr})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer

def count_trainable_params(model: nn.Module):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =====================
# DATASET - ADVANCED MODEL
# =====================
class AdvancedDistillDataset(Dataset):
    def __init__(self, csv_path, image_dir, teacher_jsonl, vision_processor, 
                 text_tokenizer, decoder_tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len
        
        # Load teacher outputs
        self.teacher_outputs = {}
        teacher_file = self._find_teacher_file(teacher_jsonl)
        if teacher_file and os.path.exists(teacher_file):
            with open(teacher_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    img_id = str(data.get('img_id', data.get('image_id')))
                    question = str(data.get('question', ''))
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
        
        print(f"[INFO] 🔍 Searching for teacher_outputs in {kaggle_input}...")
        for root, dirs, files in os.walk(kaggle_input):
            for file in files:
                if "teacher_outputs" in file and file.endswith(".jsonl"):
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
        
        # Get teacher output
        key = (img_id, question)
        teacher_data = self.teacher_outputs.get(key, {})
        teacher_answer = teacher_data.get("teacher_answer", gt_answer)
        teacher_reasoning = teacher_data.get("teacher_reasoning", "")
        
        # GT-GUIDED: Verify teacher_answer matches gt_answer
        if teacher_answer and teacher_answer.strip().lower() != gt_answer.strip().lower():
            teacher_answer = gt_answer
            teacher_reasoning = ""
        
        # Construct teacher output
        if teacher_reasoning:
            teacher_output = f"Answer: {teacher_answer}\nReasoning: {teacher_reasoning}"
        else:
            teacher_output = f"Answer: {gt_answer}"
        
        # Process image with CLIP processor
        vision_inputs = self.vision_processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = vision_inputs["pixel_values"].squeeze(0)
        
        # Tokenize question
        text_inputs = self.text_tokenizer(
            question,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        # Tokenize teacher output (target)
        teacher_inputs = self.decoder_tokenizer(
            teacher_output,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        teacher_labels = teacher_inputs["input_ids"].squeeze(0)
        teacher_labels[teacher_labels == self.decoder_tokenizer.pad_token_id] = -100
        
        # Tokenize GT answer (for monitoring)
        gt_inputs = self.decoder_tokenizer(
            f"Answer: {gt_answer}",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        gt_labels = gt_inputs["input_ids"].squeeze(0)
        gt_labels[gt_labels == self.decoder_tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "teacher_labels": teacher_labels,
            "gt_labels": gt_labels,
            "img_id": img_id
        }

# =====================
# LOSS COMPUTATION
# =====================
def compute_loss(model, batch, device):
    """
    Cross-entropy loss with teacher output
    """
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    gt_labels = batch["gt_labels"].to(device)
    teacher_labels = batch["teacher_labels"].to(device)
    
    # Forward pass with teacher labels
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=teacher_labels
    )
    
    loss_teacher = outputs.loss
    
    # GT loss (for monitoring only)
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
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Loss curves
        axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
        axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='s')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss (Advanced Model)")
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
        
        # Training stage visualization
        axes[2].plot(df["epoch"], df["trainable_params"], marker='o', color='green')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Trainable Parameters (M)")
        axes[2].set_title("Progressive Unfreezing")
        axes[2].grid(True)
        
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
    print(f"[CONFIG] Model: {'VQALightweightModel' if cfg.use_lightweight else 'VQAAdvancedModel'}")
    print(f"[CONFIG] Effective batch size: {cfg.batch_size} * {cfg.accum_steps} = {cfg.batch_size * cfg.accum_steps}")
    
    # Load model
    print("\n" + "="*70)
    print("INITIALIZING ADVANCED VQA MODEL")
    print("="*70)
    print("[INFO] Architecture:")
    print("  - Vision: CLIP ViT-Base/32 (pretrained on 400M image-text pairs)")
    print("  - Text: PhoBERT-base (pretrained on Vietnamese)")
    print("  - Fusion: Bidirectional Cross-Attention (train from scratch)")
    print(f"  - Fusion Layers: {cfg.num_fusion_layers}")
    print(f"  - Attention Heads: {cfg.num_heads}")
    print("  - Decoder: VietT5-base (pretrained on Vietnamese)")
    print("="*70 + "\n")
    
    if cfg.use_lightweight:
        model = VQALightweightModel(
            vision_model_name="openai/clip-vit-base-patch32",
            phobert_dir=os.path.join(cfg.checkpoint_dir, "phobert_tokenizer"),
            vit5_dir=os.path.join(cfg.checkpoint_dir, "vit5_tokenizer"),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.fusion_dropout
        ).to(device)
    else:
        model = VQAAdvancedModel(
            vision_model_name="openai/clip-vit-base-patch32",
            phobert_dir=os.path.join(cfg.checkpoint_dir, "phobert_tokenizer"),
            vit5_dir=os.path.join(cfg.checkpoint_dir, "vit5_tokenizer"),
            hidden_dim=cfg.hidden_dim,
            num_fusion_layers=cfg.num_fusion_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.fusion_dropout
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Total parameters: {total_params:.1f}M")
    
    # Processors
    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("[INFO] Models and processors loaded successfully!")
    
    # Dataset
    dataset = AdvancedDistillDataset(
        cfg.train_csv, cfg.image_dir, cfg.teacher_jsonl,
        vision_processor,
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
    es_counter = 0
    
    print(f"\n{'='*70}")
    print("PROGRESSIVE TRAINING STRATEGY")
    print(f"{'='*70}")
    print(f"  Stage 1 (Epochs 1-{cfg.stage1_epochs}): Fusion + Decoder ONLY")
    print(f"  Stage 2 (Epochs {cfg.stage1_epochs+1}-{cfg.stage1_epochs+cfg.stage2_epochs}): + Text Encoder")
    print(f"  Stage 3 (Epochs {cfg.stage1_epochs+cfg.stage2_epochs+1}-{cfg.num_epochs}): + Vision Encoder")
    print(f"{'='*70}\n")
    
    for epoch in range(cfg.num_epochs):
        # Determine stage
        if epoch < cfg.stage1_epochs:
            stage = 1
        elif epoch < cfg.stage1_epochs + cfg.stage2_epochs:
            stage = 2
        else:
            stage = 3
        
        # Set training stage (progressive unfreezing)
        set_training_stage(model, stage)
        
        # Build optimizer (rebuild each epoch for stage changes)
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
                loss, gt_loss = compute_loss(model, batch, device)
                loss = loss / cfg.accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss_sum += loss.item() * cfg.accum_steps
            train_gt_loss_sum += gt_loss
            
            pbar.set_postfix({
                "loss": f"{loss.item() * cfg.accum_steps:.4f}",
                "gt_loss": f"{gt_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
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
                loss, gt_loss = compute_loss(model, batch, device)
                val_loss_sum += loss.item()
                val_gt_loss_sum += gt_loss
        
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_gt_loss = val_gt_loss_sum / len(val_loader)
        
        print(f"\n[EPOCH {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"           Train GT: {avg_train_gt_loss:.4f} | Val GT: {avg_val_gt_loss:.4f}")
        
        # Logging
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{stage},{avg_train_loss:.6f},{avg_val_loss:.6f},"
                   f"{avg_train_gt_loss:.6f},{avg_val_gt_loss:.6f},{trainable/1e6:.2f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss - cfg.es_min_delta:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_advanced_model.pt"))
            print(f"✅ Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            es_counter += 1
            print(f"⚠️  No improvement ({es_counter}/{cfg.es_patience})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'es_counter': es_counter,
            }
            torch.save(checkpoint, os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            print(f"💾 Checkpoint saved at epoch {epoch+1}")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'es_counter': es_counter,
        }
        torch.save(checkpoint, os.path.join(cfg.save_dir, "latest_checkpoint_advanced.pt"))
        
        # Early stopping
        if es_counter >= cfg.es_patience:
            print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
            break
        
        clear_memory()
    
    # Final save
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "final_advanced_model.pt"))
    
    # Plot training curves
    plot_curves(log_path, os.path.join(cfg.save_dir, cfg.curve_png))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Total epochs: {epoch+1}/{cfg.num_epochs}")
    print(f"Logs saved to: {log_path}")
    print(f"Best model saved to: {os.path.join(cfg.save_dir, 'best_advanced_model.pt')}")
    print(f"\n[NEXT STEP] Run evaluation to compare with baseline (~31.29%)")
    print(f"Expected: 4-9% absolute improvement with advanced architecture")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train()
