"""
STAGED TRAINING STRATEGY: Progressive Unfreezing
================================================
3-Stage Training cho DINOv2 + BARTpho VQA với 11K samples:

Stage 1 (Epochs 1-10): FUSION ONLY
  - Freeze: Vision Encoder + Language Encoder + Language Decoder
  - Train: Vision Projection + Cross-Attention Fusion + LM Head
  - Params: ~13M (~1,180 params/sample) ✅
  - LR: 5e-4 (cao vì random init)
  
Stage 2 (Epochs 11-25): + LANGUAGE MODELS
  - Freeze: Vision Encoder
  - Train: Fusion + Language Encoder (last 6 layers) + Language Decoder
  - Params: ~120M (~10,900 params/sample) ⚠️ 
  - LR: 1e-4 (giảm xuống)
  
Stage 3 (Epochs 26-40): + VISION HEAD (NOT full ViT!)
  - Freeze: Vision Encoder body (blocks 0-10)
  - Train: All previous + Vision Encoder head (block 11 + norm)
  - Params: ~135M (~12,200 params/sample) ⚠️
  - LR: 5e-5 (giảm tiếp)

Strategy: Chỉ unfreeze vision HEAD thay vì toàn bộ ViT vì:
  ✅ Head đã học semantic features (gần task)
  ✅ Body là low-level features (không cần fine-tune)
  ✅ Tiết kiệm params: 135M vs 527M (giảm 74%!)
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from transformers import get_cosine_schedule_with_warmup, AutoImageProcessor
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from dataset import VQAGenDataset
from model_dinov2_bartpho_2 import DINOv2BARTphoVQA


# ============================================================================
# STAGE CONFIGURATIONS
# ============================================================================

@dataclass
class StageConfig:
    """Configuration for each training stage"""
    name: str
    start_epoch: int
    end_epoch: int
    learning_rate: float
    weight_decay: float
    dropout: float
    description: str
    
    # What to unfreeze
    unfreeze_fusion: bool = False
    unfreeze_encoder_last_n: int = 0  # BARTpho encoder layers
    unfreeze_decoder: bool = False
    unfreeze_vision_head: bool = False  # Only head, not full ViT
    

# ============================================================================
# FREEZING UTILITIES
# ============================================================================

def freeze_all(model: DINOv2BARTphoVQA):
    """Freeze toàn bộ model"""
    for param in model.parameters():
        param.requires_grad = False


def configure_stage(model: DINOv2BARTphoVQA, stage: StageConfig):
    """
    Configure model cho từng stage
    
    Stage 1: Fusion only (~13M params)
    Stage 2: + Language models (~120M params)  
    Stage 3: + Vision head (~135M params)
    """
    print(f"\n{'='*80}")
    print(f"🎯 CONFIGURING {stage.name.upper()}")
    print(f"{'='*80}")
    print(f"Description: {stage.description}")
    print(f"Epochs: {stage.start_epoch} → {stage.end_epoch}")
    print(f"Learning Rate: {stage.learning_rate:.2e}")
    print(f"Weight Decay: {stage.weight_decay}")
    
    # Start fresh: freeze everything
    freeze_all(model)
    
    trainable_groups = []
    
    # 1. FUSION (Vision Proj + Cross-Attention + LM Head)
    if stage.unfreeze_fusion:
        print("\n✅ UNFREEZING: Fusion Components")
        for param in model.vision_proj.parameters():
            param.requires_grad = True
        for param in model.cross_attention_fusion.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
        
        proj_params = sum(p.numel() for p in model.vision_proj.parameters())
        fusion_params = sum(p.numel() for p in model.cross_attention_fusion.parameters())
        lm_params = sum(p.numel() for p in model.lm_head.parameters())
        print(f"  • Vision Projection: {proj_params/1e6:.1f}M")
        print(f"  • Cross-Attention: {fusion_params/1e6:.1f}M")
        print(f"  • LM Head: {lm_params/1e6:.1f}M")
        trainable_groups.append(f"Fusion ({(proj_params+fusion_params+lm_params)/1e6:.1f}M)")
    
    # 2. LANGUAGE ENCODER (Last N layers)
    if stage.unfreeze_encoder_last_n > 0:
        print(f"\n✅ UNFREEZING: BARTpho Encoder (Last {stage.unfreeze_encoder_last_n} layers)")
        total_layers = len(model.encoder.layers)
        encoder_params = 0
        for i, layer in enumerate(model.encoder.layers):
            if i >= total_layers - stage.unfreeze_encoder_last_n:
                for param in layer.parameters():
                    param.requires_grad = True
                encoder_params += sum(p.numel() for p in layer.parameters())
        print(f"  • Encoder layers: {encoder_params/1e6:.1f}M")
        trainable_groups.append(f"Encoder ({encoder_params/1e6:.1f}M)")
    
    # 3. LANGUAGE DECODER
    if stage.unfreeze_decoder:
        print("\n✅ UNFREEZING: BARTpho Decoder (Full)")
        for param in model.decoder.parameters():
            param.requires_grad = True
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"  • Decoder: {decoder_params/1e6:.1f}M")
        trainable_groups.append(f"Decoder ({decoder_params/1e6:.1f}M)")
    
    # 4. VISION HEAD (Block 11 + LayerNorm only, NOT full ViT!)
    if stage.unfreeze_vision_head:
        print("\n✅ UNFREEZING: DINOv2 Head (Block 11 + Norm)")
        # DINOv2 has 12 blocks (0-11), unfreeze only last block + norm
        total_blocks = len(model.vision_encoder.encoder.layer)
        vision_params = 0
        
        # Unfreeze last block only
        last_block = model.vision_encoder.encoder.layer[-1]
        for param in last_block.parameters():
            param.requires_grad = True
        vision_params += sum(p.numel() for p in last_block.parameters())
        
        # Unfreeze final layernorm
        for param in model.vision_encoder.layernorm.parameters():
            param.requires_grad = True
        vision_params += sum(p.numel() for p in model.vision_encoder.layernorm.parameters())
        
        print(f"  • Vision Head (Block 11 + Norm): {vision_params/1e6:.1f}M")
        print(f"  • Vision Body (Blocks 0-10): FROZEN ❄️")
        trainable_groups.append(f"Vision Head ({vision_params/1e6:.1f}M)")
    
    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*80}")
    print(f"📊 STAGE SUMMARY")
    print(f"{'='*80}")
    print(f"Trainable: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")
    print(f"Frozen: {frozen_params/1e6:.1f}M ({frozen_params/total_params*100:.1f}%)")
    print(f"Components: {' + '.join(trainable_groups)}")
    
    # Safety check: với 11K samples
    params_per_sample = trainable_params / 11000
    print(f"\n⚠️  Params/Sample: {params_per_sample:.0f}")
    if params_per_sample > 15000:
        print(f"   WARNING: High risk of overfitting! (>15K)")
    elif params_per_sample > 5000:
        print(f"   CAUTION: Moderate risk (5K-15K)")
    else:
        print(f"   SAFE: Low risk (<5K)")
    
    print(f"{'='*80}\n")
    
    return trainable_params


# ============================================================================
# TRAINING CONFIG
# ============================================================================

@dataclass
class TrainConfig:
    csv_path: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_folder: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    save_dir: str = "/kaggle/working/checkpoints"
    
    batch_size: int = 4
    accum_steps: int = 8  # Effective batch = 32
    val_split: float = 0.1
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1  # 10% warmup
    use_amp: bool = True
    
    # Early stopping per stage - AGGRESSIVE to prevent overfitting
    es_patience: int = 3  # 🔥 Giảm từ 5 → 3 (stop sớm hơn)
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_staged_log.csv"
    curve_png: str = "training_staged_curve.png"


# Define 3 stages - ULTRA CONSERVATIVE (based on analysis)
STAGES = [
    # Stage 1: Fusion Only - REDUCED to prevent overfitting after epoch 6
    StageConfig(
        name="Stage 1: Fusion Warm-up",
        start_epoch=0,
        end_epoch=8,  # 🔥 Giảm từ 10 → 8 (stop trước khi overfit)
        learning_rate=5e-4,  # Cao vì random init
        weight_decay=0.01,
        dropout=0.2,
        description="Train fusion components from scratch (vision proj + cross-attn + lm_head)",
        unfreeze_fusion=True,
        unfreeze_encoder_last_n=0,
        unfreeze_decoder=False,
        unfreeze_vision_head=False
    ),
    
    # Stage 2: + Language Models - SUPER GENTLE (prevent catastrophic forgetting)
    StageConfig(
        name="Stage 2: Language Fine-tune",
        start_epoch=8,
        end_epoch=13,  # 🔥 Giảm từ 25 → 13 (chỉ 5 epochs, dừng sớm)
        learning_rate=1e-5,  # 🔥 Giảm 10x: 1e-4 → 1e-5 (prevent distribution shift)
        weight_decay=0.03,  # 🔥 Tăng regularization
        dropout=0.4,  # 🔥 Tăng dropout 0.3 → 0.4 (364M params!)
        description="GENTLE language fine-tune (encoder last 3 layers + decoder only)",
        unfreeze_fusion=True,
        unfreeze_encoder_last_n=3,  # 🔥 Giảm từ 6 → 3 layers (more conservative)
        unfreeze_decoder=True,
        unfreeze_vision_head=False
    ),
    
    # Stage 3: MICRO POLISH - Chỉ để polish, không expect cải thiện lớn
    StageConfig(
        name="Stage 3: Micro Polish",
        start_epoch=13,
        end_epoch=18,  # 🔥 Giảm từ 40 → 18 (chỉ 5 epochs)
        learning_rate=5e-6,  # 🔥 Giảm 10x: 5e-5 → 5e-6 (cực kỳ nhẹ nhàng)
        weight_decay=0.04,  # 🔥 Tăng tiếp
        dropout=0.4,
        description="Micro-polish all components with minimal LR (expect minimal gain)",
        unfreeze_fusion=True,
        unfreeze_encoder_last_n=3,
        unfreeze_decoder=True,
        unfreeze_vision_head=True  # Allow but with tiny LR
    ),
]


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: DINOv2BARTphoVQA, lr: float, weight_decay: float):
    """Build optimizer cho trainable params"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    return optimizer


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    """Build cosine schedule với warmup"""
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler


def run_one_epoch(model, loader, optimizer, scaler, device, cfg, scheduler=None, train=True):
    """Train/Val một epoch"""
    if train:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    steps = 0
    
    pbar = tqdm(loader, disable=False, leave=False)
    accum_steps = cfg.accum_steps if train else 1
    
    for step, batch in enumerate(pbar):
        pixel_values, input_ids, attention_mask, labels = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.set_grad_enabled(train):
            if train:
                if cfg.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
                        loss = loss / accum_steps
                    scaler.scale(loss).backward()
                else:
                    loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
                    loss = loss / accum_steps
                    loss.backward()
            else:
                if cfg.use_amp:
                    with autocast('cuda', dtype=torch.float16):
                        loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
                else:
                    loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
        
        if train and (step + 1) % accum_steps == 0:
            if cfg.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        running_loss += loss.item() * accum_steps
        steps += 1
        pbar.set_description(f"{'Train' if train else 'Val'} loss: {running_loss/steps:.4f}")
    
    return running_loss / max(steps, 1)


def plot_curves(csv_path, out_png):
    """Plot training curves với stage markers"""
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss", marker='o', markersize=3)
    plt.plot(df["epoch"], df["val_loss"], label="val_loss", marker='s', markersize=3)
    
    # Add stage boundaries
    for stage in STAGES[1:]:  # Skip first stage
        plt.axvline(x=stage.start_epoch, color='red', linestyle='--', alpha=0.5, 
                   label=f'{stage.name}' if stage == STAGES[1] else '')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Staged Training: Progressive Unfreezing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[INFO] Curve saved to {out_png}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    
    cfg = TrainConfig()
    
    # Parse args (optional)
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--csv_path", type=str)
        parser.add_argument("--image_folder", type=str)
        parser.add_argument("--save_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        args = parser.parse_args()
        
        if args.csv_path: cfg.csv_path = args.csv_path
        if args.image_folder: cfg.image_folder = args.image_folder
        if args.save_dir: cfg.save_dir = args.save_dir
        if args.batch_size: cfg.batch_size = args.batch_size
    except:
        pass
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("🚀 STAGED TRAINING: Progressive Unfreezing Strategy")
    print("="*80)
    print(f"Device: {device}")
    print(f"Stages: {len(STAGES)}")
    print(f"Total Epochs: {STAGES[-1].end_epoch}")
    print(f"Dataset: {cfg.csv_path}")
    print("="*80)
    
    # Load dataset
    vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    full_dataset = VQAGenDataset(cfg.csv_path, cfg.image_folder, vision_processor)
    
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"\n[INFO] Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    
    # Initialize model
    print("\n[INFO] Initializing model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        num_heads=16,
        dropout=0.2,  # Will be updated per stage
        gradient_checkpointing=True
    ).to(device)
    
    print(f"[INFO] Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.1f}M total params")
    
    # Initialize logging
    log_path = os.path.join(cfg.save_dir, cfg.log_csv)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "epoch", "stage", "trainable_params", "lr", 
            "train_loss", "val_loss", "best_val", "es_counter"
        ]).to_csv(log_path, index=False)
    
    scaler = GradScaler(enabled=cfg.use_amp)
    best_val_overall = float("inf")
    
    # ========================================================================
    # STAGED TRAINING LOOP
    # ========================================================================
    
    for stage in STAGES:
        print(f"\n{'#'*80}")
        print(f"# STARTING {stage.name.upper()}")
        print(f"{'#'*80}\n")
        
        # Configure model for this stage
        trainable_params = configure_stage(model, stage)
        
        # Build optimizer & scheduler for this stage
        optimizer = build_optimizer(model, stage.learning_rate, stage.weight_decay)
        
        stage_epochs = stage.end_epoch - stage.start_epoch
        steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
        total_steps = steps_per_epoch * stage_epochs
        scheduler = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)
        
        # Early stopping for this stage
        best_val_stage = float("inf")
        es_counter = 0
        
        # Train this stage
        for epoch in range(stage.start_epoch, stage.end_epoch):
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Train & Val
            train_loss = run_one_epoch(
                model, train_loader, optimizer, scaler, device, cfg, scheduler, train=True
            )
            val_loss = run_one_epoch(
                model, val_loader, None, scaler, device, cfg, None, train=False
            )
            
            # Check improvement
            improved = (best_val_stage - val_loss) > cfg.es_min_delta
            if improved:
                best_val_stage = val_loss
                es_counter = 0
                
                # Save stage best
                torch.save(
                    model.state_dict(), 
                    os.path.join(cfg.save_dir, f"best_model_{stage.name.replace(' ', '_').replace(':', '')}.pth")
                )
                
                # Update overall best
                if val_loss < best_val_overall:
                    best_val_overall = val_loss
                    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pth"))
                    print(f"[INFO] 🏆 NEW OVERALL BEST @ epoch {epoch+1}: val={val_loss:.4f}")
            else:
                es_counter += 1
            
            # Log
            row = {
                "epoch": epoch + 1,
                "stage": stage.name,
                "trainable_params": trainable_params,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val_stage,
                "es_counter": es_counter
            }
            df = pd.read_csv(log_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(log_path, index=False)
            
            print(f"[{stage.name}] Epoch {epoch+1}/{stage.end_epoch} | "
                  f"Params={trainable_params/1e6:.1f}M | LR={current_lr:.2e} | "
                  f"Train={train_loss:.4f} | Val={val_loss:.4f} | "
                  f"Best={best_val_stage:.4f} | ES={es_counter}/{cfg.es_patience}")
            
            # Early stopping check
            if es_counter >= cfg.es_patience:
                print(f"\n[INFO] Early stopping triggered for {stage.name}")
                print(f"       Best val: {best_val_stage:.4f}")
                break
        
        # Save last checkpoint for this stage
        torch.save(model.state_dict(), 
                  os.path.join(cfg.save_dir, f"last_model_{stage.name.replace(' ', '_').replace(':', '')}.pth"))
        
        print(f"\n[INFO] ✓ {stage.name} completed!")
        print(f"       Best val this stage: {best_val_stage:.4f}")
        print(f"       Best val overall: {best_val_overall:.4f}")
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 STAGED TRAINING COMPLETED!")
    print("="*80)
    print(f"Best validation loss: {best_val_overall:.4f}")
    print(f"Checkpoints saved to: {cfg.save_dir}")
    print(f"Training log: {log_path}")
    
    # Save tokenizer
    try:
        full_dataset.dataset.tokenizer.save_pretrained(
            os.path.join(cfg.save_dir, "bartpho_tokenizer")
        )
        print(f"Tokenizer saved to: {os.path.join(cfg.save_dir, 'bartpho_tokenizer')}")
    except Exception as e:
        print(f"[WARN] Could not save tokenizer: {e}")
    
    # Plot curves
    try:
        plot_curves(log_path, os.path.join(cfg.save_dir, cfg.curve_png))
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")
    
    print("="*80)


if __name__ == "__main__":
    main()
