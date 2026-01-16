#!/usr/bin/env python3
"""
ANTI-HALLUCINATION 3-Stage Training Pipeline for SimpleFusionVQA
==================================================================

SAME AS run_3stage_simple.py + 3 CRITICAL anti-hallucination fixes:
1. IMAGE DROPOUT (20%) - Force model to die without image  
2. ANSWER FREQUENCY REWEIGHTING - Punish common answers
3. CONTRASTIVE NEGATIVE IMAGES - Different images -> different answers

Optimized strategy:
- Stage 1 (10-15 epochs): Freeze all, train fusion only
- Stage 2 (7-10 epochs): Unfreeze BARTpho decoder last 2-3 layers (lr=5e-6, LOWER!)
- Stage 2.5 (0-5 epochs): Unfreeze BARTpho encoder last 2 layers (lr=3e-6, optional)
- DINOv2: ALWAYS FROZEN (pre-trained on 142M images, no need to fine-tune)

Expected improvement: val_loss 1.034 → 0.85-0.90 (15-20% better!)

Usage:
    python run_anti_hallucination.py \
        --csv_path /path/to/train.csv \
        --image_folder /path/to/images \
        --stage1_epochs 12 \
        --stage2_epochs 8 \
        --stage2_5_epochs 0 \
        --decoder_lr 5e-6 \
        --use_image_dropout \
        --use_freq_reweight \
        --use_contrastive
"""

import torch
import os
import gc
from dataclasses import dataclass
from train import (
    FixedTrainConfig, set_seed
)
from model import SimpleFusionVQA
from dataset import VQAGenDataset
from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# Import anti-hallucination tools
from anti_hallucination import (
    AntiHallucinationLoss,
    apply_image_dropout,
    shuffle_images_in_batch,
    compute_answer_frequency,
    test_hallucination
)


def get_current_stage(epoch: int, stage1_end: int, stage2_end: int):
    """Determine current stage (1, 2, or 2.5) based on epoch number"""
    if epoch < stage1_end:
        return 1
    elif epoch < stage2_end:
        return 2
    else:
        return 2.5  # Stage 2.5 - encoder unfreezing


def log_memory_usage(step_name=""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  💾 {step_name} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB")


def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_training_curves(history, save_dir, stage1_epochs, stage2_epochs):
    """
    Plot and save training curves (SAME AS run_3stage_simple.py)
    - Loss curves (train vs val)
    - Learning rate schedule
    - Stage transitions marked
    """
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    lr = history['lr']
    stages = history['stage']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # === Plot 1: Loss Curves ===
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4, alpha=0.8)
    ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4, alpha=0.8)
    
    # Mark stage transitions
    stage1_end = stage1_epochs
    stage2_end = stage1_epochs + stage2_epochs
    
    if len(epochs) > stage1_end:
        ax1.axvline(x=stage1_end, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Stage 1→2 (Decoder)')
    if len(epochs) > stage2_end:
        ax1.axvline(x=stage2_end, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Stage 2→2.5 (Encoder)')
    
    # Add stage background colors
    if len(epochs) > 0:
        stage1_x = [e for e in epochs if e <= stage1_end]
        if stage1_x:
            ax1.axvspan(min(stage1_x), max(stage1_x), alpha=0.1, color='blue', label='Stage 1 (Fusion)')
        
        stage2_x = [e for e in epochs if stage1_end < e <= stage2_end]
        if stage2_x:
            ax1.axvspan(min(stage2_x), max(stage2_x), alpha=0.1, color='orange', label='Stage 2 (+ Decoder)')
        
        stage2_5_x = [e for e in epochs if e > stage2_end]
        if stage2_5_x:
            ax1.axvspan(min(stage2_5_x), max(stage2_5_x), alpha=0.1, color='purple', label='Stage 2.5 (+ Encoder)')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training with Anti-Hallucination Fixes (DINOv2 Frozen)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Learning Rate Schedule ===
    ax2 = axes[1]
    ax2.plot(epochs, lr, 'g-o', linewidth=2, markersize=4, alpha=0.8)
    
    # Mark stage transitions
    if len(epochs) > stage1_end:
        ax2.axvline(x=stage1_end, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    if len(epochs) > stage2_end:
        ax2.axvline(x=stage2_end, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add stage background colors
    if len(epochs) > 0:
        stage1_x = [e for e in epochs if e <= stage1_end]
        if stage1_x:
            ax2.axvspan(min(stage1_x), max(stage1_x), alpha=0.1, color='blue')
        
        stage2_x = [e for e in epochs if stage1_end < e <= stage2_end]
        if stage2_x:
            ax2.axvspan(min(stage2_x), max(stage2_x), alpha=0.1, color='orange')
        
        stage2_5_x = [e for e in epochs if e > stage2_end]
        if stage2_5_x:
            ax2.axvspan(min(stage2_5_x), max(stage2_5_x), alpha=0.1, color='purple')
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Learning Rate Schedule (Cosine with Stage Transitions)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for LR changes
    if len(epochs) > stage1_end and stage1_end < len(lr):
        ax2.annotate(f'Decoder LR\n{lr[stage1_end]:.2e}', 
                    xy=(stage1_end, lr[stage1_end]), 
                    xytext=(stage1_end-2, lr[stage1_end]*2),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                    fontsize=9, ha='right')
    
    if len(epochs) > stage2_end and stage2_end < len(lr):
        ax2.annotate(f'Encoder LR\n{lr[stage2_end]:.2e}', 
                    xy=(stage2_end, lr[stage2_end]), 
                    xytext=(stage2_end-2, lr[stage2_end]*2),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                    fontsize=9, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    curve_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved training curves: {curve_path}")
    plt.close()
    
    # === Additional Plot: Loss comparison ===
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot both losses with fill between
    ax.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.2, color='gray', label='Train-Val Gap')
    
    # Mark best model
    if val_loss:
        best_epoch_idx = val_loss.index(min(val_loss))
        best_epoch = epochs[best_epoch_idx]
        best_val = val_loss[best_epoch_idx]
        ax.scatter([best_epoch], [best_val], color='red', s=200, marker='*', 
                  edgecolors='black', linewidth=2, zorder=5, label=f'Best Model (Epoch {best_epoch})')
        ax.annotate(f'Best: {best_val:.4f}', 
                   xy=(best_epoch, best_val), 
                   xytext=(best_epoch+1, best_val+0.05),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, fontweight='bold')
    
    # Stage transitions
    if len(epochs) > stage1_end:
        ax.axvline(x=stage1_end, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(stage1_end, ax.get_ylim()[1]*0.95, 'Stage 1→2\n(+ Decoder)', 
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    if len(epochs) > stage2_end:
        ax.axvline(x=stage2_end, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(stage2_end, ax.get_ylim()[1]*0.95, 'Stage 2→2.5\n(+ Encoder)', 
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss Comparison with Best Model Marker', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_compare_path = os.path.join(save_dir, "loss_comparison.png")
    plt.savefig(loss_compare_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved loss comparison: {loss_compare_path}")
    plt.close()



def run_one_epoch_anti_hallucination(
    model, loader, optimizer, scaler, device, cfg,
    anti_hallucination_loss,
    use_image_dropout=True,
    use_contrastive=True,
    scheduler=None,
    train=True
):
    """
    Run one epoch with anti-hallucination training
    
    SAME AS run_one_epoch_simple() BUT with 3 additions:
    1. Random image dropout 20% of batches
    2. Contrastive learning 10% of batches  
    3. Frequency-reweighted loss always
    """
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    if train:
        optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=120)
    
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        pixel_values_orig = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # ====================================================================
        # ANTI-HALLUCINATION AUGMENTATION (ONLY ADDITIONS!)
        # ====================================================================
        
        apply_dropout_this_batch = False
        apply_contrastive_this_batch = False
        contrastive_logits = None
        
        if train:
            # 1. Image Dropout (20% probability)
            if use_image_dropout and torch.rand(1).item() < 0.2:
                pixel_values, _ = apply_image_dropout(pixel_values_orig, dropout_prob=0.5)
                apply_dropout_this_batch = True
            else:
                pixel_values = pixel_values_orig
            
            # 2. Contrastive Learning (10% probability)
            if use_contrastive and torch.rand(1).item() < 0.1:
                shuffled_images, _ = shuffle_images_in_batch(pixel_values_orig)
                apply_contrastive_this_batch = True
        else:
            pixel_values = pixel_values_orig
        
        # ====================================================================
        # FORWARD PASS (SAME AS run_3stage_simple.py)
        # ====================================================================
        
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # Standard forward
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logits = outputs.logits
            
            # Contrastive forward (if needed)
            if train and apply_contrastive_this_batch:
                outputs_shuffled = model(
                    pixel_values=shuffled_images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                contrastive_logits = outputs_shuffled.logits
            
            # Compute anti-hallucination loss (replaces simple outputs.loss)
            loss, loss_dict = anti_hallucination_loss(
                logits=logits,
                labels=labels,
                pixel_values=pixel_values,
                apply_dropout=apply_dropout_this_batch,
                contrastive_logits=contrastive_logits
            )
            
            if train:
                loss = loss / cfg.accum_steps
        
        # ====================================================================
        # BACKWARD PASS (SAME AS run_3stage_simple.py)
        # ====================================================================
        
        if train:
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        
        # Accumulate
        total_loss += loss.item() * cfg.accum_steps if train else loss.item()
        num_batches += 1
        
        # Progress bar (enhanced with loss components)
        pbar_dict = {'Loss': f"{loss.item() * cfg.accum_steps if train else loss.item():.3f}"}
        if 'base_loss' in loss_dict:
            pbar_dict['Base'] = f"{loss_dict['base_loss']:.3f}"
        if 'dropout_penalty' in loss_dict and loss_dict.get('dropout_penalty', 0) > 0:
            pbar_dict['Drop'] = f"{loss_dict['dropout_penalty']:.3f}"
        if 'contrastive_loss' in loss_dict and loss_dict.get('contrastive_loss', 0) > 0:
            pbar_dict['Contr'] = f"{loss_dict['contrastive_loss']:.3f}"
        pbar.set_postfix(pbar_dict)
    
    return total_loss / num_batches




def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Anti-Hallucination 3-Stage Training (BASED ON run_3stage_simple.py)")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", "--accumulation_steps", type=int, default=8, dest="accum_steps",
                       help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--learning_rate", "--base_lr", type=float, default=5e-5, dest="base_lr",
                       help="Base learning rate for Stage 1 (default: 5e-5)")
    parser.add_argument("--stage1_epochs", type=int, default=12,
                       help="Stage 1: Train fusion only (10-15 recommended)")
    parser.add_argument("--stage2_epochs", type=int, default=8,
                       help="Stage 2: Unfreeze decoder last layers (7-10 recommended)")
    parser.add_argument("--stage2_5_epochs", type=int, default=0,
                       help="Stage 2.5: Unfreeze encoder last layers (0=skip, RECOMMENDED!)")
    parser.add_argument("--num_fusion_layers", type=int, default=3,
                       help="Number of fusion layers (default: 3 for SOTA)")
    parser.add_argument("--num_decoder_layers", type=int, default=3,
                       help="Decoder layers to unfreeze in stage 2 (2-3 recommended)")
    parser.add_argument("--num_encoder_layers", type=int, default=2,
                       help="Encoder layers to unfreeze in stage 2.5 (2 recommended)")
    parser.add_argument("--decoder_lr", type=float, default=5e-6,
                       help="Learning rate for decoder in Stage 2 (LOWER: 5e-6 vs 1e-5)")
    parser.add_argument("--encoder_lr", type=float, default=3e-6,
                       help="Learning rate for encoder in Stage 2.5 (3e-6 recommended)")
    
    # Anti-hallucination flags (ONLY ADDITIONS!)
    parser.add_argument("--use_image_dropout", action="store_true",
                       help="Enable image dropout 20%% (CRITICAL FIX!)")
    parser.add_argument("--use_freq_reweight", action="store_true",
                       help="Enable answer frequency reweighting (FIX bias!)")
    parser.add_argument("--use_contrastive", action="store_true",
                       help="Enable contrastive negative images (FIX shortcuts!)")
    parser.add_argument("--test_hallucination", action="store_true",
                       help="Test hallucination rate before training")
    
    parser.add_argument("--save_dir", type=str, default="checkpoints_anti_hallucination")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Configuration (SAME AS run_3stage_simple.py)
    cfg = FixedTrainConfig()
    cfg.csv_path = args.csv_path
    cfg.image_folder = args.image_folder
    cfg.batch_size = args.batch_size
    cfg.accum_steps = args.accum_steps
    cfg.save_dir = args.save_dir
    cfg.use_amp = True
    cfg.max_grad_norm = 1.0
    cfg.base_lr = args.base_lr
    cfg.weight_decay = 0.05
    cfg.warmup_ratio = 0.06
    cfg.max_q_len = 64
    cfg.max_a_len = 10
    
    # Total epochs
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage2_5_epochs
    stage1_end = args.stage1_epochs
    stage2_end = args.stage1_epochs + args.stage2_epochs
    
    print("="*80)
    print("🔥 ANTI-HALLUCINATION 3-STAGE TRAINING: SimpleFusionVQA")
    if args.resume_from:
        print(f"🔄 RESUME MODE: {args.resume_from}")
    print("="*80)
    print(f"\n✅ Anti-Hallucination Fixes:")
    print(f"  1. Image Dropout:        {'YES ✓' if args.use_image_dropout else 'NO ✗'}")
    print(f"  2. Frequency Reweighting: {'YES ✓' if args.use_freq_reweight else 'NO ✗'}")
    print(f"  3. Contrastive Learning:  {'YES ✓' if args.use_contrastive else 'NO ✗'}")
    print(f"\n🎯 Strategy: Prevent Overfitting + Hallucination")
    print(f"  ✓ DINOv2: ALWAYS frozen (142M images pre-training)")
    print(f"  ✓ Stage 1: Only fusion layers trainable")
    print(f"  ✓ Stage 2: + Decoder last {args.num_decoder_layers} layers (LR={args.decoder_lr:.2e}, LOWER!)")
    if args.stage2_5_epochs > 0:
        print(f"  ✓ Stage 2.5: + Encoder last {args.num_encoder_layers} layers (LR={args.encoder_lr:.2e})")
    else:
        print(f"  ✓ Stage 2.5: SKIPPED (recommended for anti-hallucination!)")
    print(f"\n📊 Stage boundaries:")
    print(f"  Stage 1:   Epochs 1-{stage1_end} ({args.stage1_epochs} epochs)")
    print(f"  Stage 2:   Epochs {stage1_end+1}-{stage2_end} ({args.stage2_epochs} epochs)")
    if args.stage2_5_epochs > 0:
        print(f"  Stage 2.5: Epochs {stage2_end+1}-{total_epochs} ({args.stage2_5_epochs} epochs)")
    print(f"  TOTAL: {total_epochs} epochs")
    print(f"\n📈 Expected: val_loss 0.85-0.90 (15-20% better than 1.034!)")
    print("="*80 + "\n")
    
    # Setup (SAME AS run_3stage_simple.py)
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Clear memory before starting
    if torch.cuda.is_available():
        clear_memory()
        print(f"\n💾 GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Total Memory: {total_memory:.2f}GB")
        log_memory_usage("Initial")
        print()
    
    # Model (SAME AS run_3stage_simple.py)
    print("[1/7] Loading SimpleFusionVQA model...")
    model = SimpleFusionVQA(
        num_fusion_layers=args.num_fusion_layers,
        gradient_checkpointing=True
    )
    model.freeze_all_pretrained()  # Start with all frozen
    model = model.to(device)
    
    # Dataset (SAME AS run_3stage_simple.py)
    print("\n[2/7] Loading dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    full_dataset = VQAGenDataset(
        csv_path=cfg.csv_path,
        image_folder=cfg.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=cfg.max_q_len,
        max_a_len=cfg.max_a_len
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # ANTI-HALLUCINATION SETUP (NEW!)
    print("\n[3/7] Computing answer token frequencies...")
    answer_freq_dict = None
    if args.use_freq_reweight:
        answer_freq_dict = compute_answer_frequency(
            full_dataset, 
            model.tokenizer, 
            max_length=cfg.max_a_len
        )
    else:
        print("  Skipping frequency reweighting (disabled)")
    
    print("\n[4/7] Setting up anti-hallucination loss...")
    anti_hallucination_loss = AntiHallucinationLoss(
        answer_freq_dict=answer_freq_dict,
        vocab_size=len(model.tokenizer) if answer_freq_dict is not None else None,
        image_dropout_prob=0.2 if args.use_image_dropout else 0.0,
        contrastive_weight=0.05 if args.use_contrastive else 0.0,  # REDUCED: 0.1 → 0.05
        dropout_penalty_weight=0.5,  # REDUCED: 2.0 → 0.5 (too aggressive!)
        freq_smoothing=10.0
    )
    print("  ✓ Loss configured")
    
    # Test hallucination rate (optional)
    if args.test_hallucination:
        print("\n[5/7] Testing baseline hallucination rate...")
        hallucination_rate = test_hallucination(model, val_loader, device, num_samples=50)
    else:
        print("\n[5/7] Skipping hallucination test")
    
    # Optimizer & Scheduler (SAME AS run_3stage_simple.py)
    print("\n[6/7] Setting up optimizer...")
    total_steps = len(train_loader) // cfg.accum_steps * total_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Training loop (SAME STRUCTURE AS run_3stage_simple.py)
    print("\n[7/7] Starting training...")
    print("="*80 + "\n")
    
    history = {
        'epoch': [], 'stage': [], 'train_loss': [], 'val_loss': [], 'lr': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(total_epochs):
        # Determine current stage (FIX: use boundaries!)
        current_stage = get_current_stage(epoch, stage1_end, stage2_end)
        
        # Stage transition: Update model freezing (SAME AS run_3stage_simple.py)
        if epoch == stage1_end and args.stage2_epochs > 0:
            print("\n" + "="*80)
            print(f"🟡 STAGE 2: Unfreezing Decoder (Last {args.num_decoder_layers} Layers)")
            print("="*80)
            print(f"  Strategy: Decoder needs VQA-specific adaptation")
            print(f"  LR: {cfg.base_lr:.2e} → {args.decoder_lr:.2e} (LOWER for anti-hallucination!)")
            print(f"  Reason: Task-specific generation patterns")
            
            model.unfreeze_text_components(
                num_encoder_layers=0,  # Keep encoder frozen
                num_decoder_layers=args.num_decoder_layers
            )
            
            # Rebuild optimizer with new LR
            remaining_steps = len(train_loader) // cfg.accum_steps * (total_epochs - epoch)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.decoder_lr,
                weight_decay=cfg.weight_decay
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=remaining_steps
            )
            print()
            
        elif epoch == stage2_end and args.stage2_5_epochs > 0:
            print("\n" + "="*80)
            print(f"🟣 STAGE 2.5: Unfreezing Encoder (Last {args.num_encoder_layers} Layers)")
            print("="*80)
            print(f"  Strategy: Fine-tune question understanding")
            print(f"  LR: {args.decoder_lr:.2e} → {args.encoder_lr:.2e}")
            print(f"  Reason: VQA-specific question patterns")
            print(f"  Note: DINOv2 stays FROZEN (critical for generalization)")
            
            model.unfreeze_text_components(
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers
            )
            
            # Rebuild optimizer with even lower LR
            remaining_steps = len(train_loader) // cfg.accum_steps * (total_epochs - epoch)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.encoder_lr,
                weight_decay=cfg.weight_decay
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=remaining_steps
            )
            print()
        
        print(f"EPOCH {epoch+1}/{total_epochs} (Stage {current_stage})")
        print("="*80)
        
        # Train (WITH ANTI-HALLUCINATION!)
        train_loss = run_one_epoch_anti_hallucination(
            model, train_loader, optimizer, scaler, device, cfg,
            anti_hallucination_loss,
            use_image_dropout=args.use_image_dropout,
            use_contrastive=args.use_contrastive,
            scheduler=scheduler,
            train=True
        )
        
        # Clear memory before validation
        clear_memory()
        
        # Validation
        with torch.no_grad():
            val_loss = run_one_epoch_anti_hallucination(
                model, val_loader, optimizer, scaler, device, cfg,
                anti_hallucination_loss,
                use_image_dropout=False,
                use_contrastive=False,
                train=False
            )
        
        # Clear memory after validation
        clear_memory()
        
        # Logging (SAME AS run_3stage_simple.py)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Stage: {current_stage}")
        
        # Log memory every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_memory_usage(f"Epoch {epoch+1}")
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['stage'].append(current_stage)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Save checkpoints (SAME AS run_3stage_simple.py)
        checkpoint = {
            'epoch': epoch + 1,
            'stage': current_stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': cfg.__dict__,
            'history': history,
            # Save anti-hallucination settings
            'anti_hallucination_settings': {
                'use_image_dropout': args.use_image_dropout,
                'use_freq_reweight': args.use_freq_reweight,
                'use_contrastive': args.use_contrastive
            }
        }
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"  ✅ New best model saved! (val_loss: {best_val_loss:.4f})")
        
        # Save last checkpoint
        last_path = os.path.join(cfg.save_dir, "last.pt")
        torch.save(checkpoint, last_path)
        
        # Save training history CSV (every epoch)
        df = pd.DataFrame(history)
        csv_path = os.path.join(cfg.save_dir, "training_history.csv")
        df.to_csv(csv_path, index=False)
        
        # Plot training curves (every 5 epochs)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs:
            try:
                plot_training_curves(history, cfg.save_dir, args.stage1_epochs, args.stage2_epochs)
                print(f"  📊 Updated training curves and plots")
            except Exception as e:
                print(f"  ⚠️  Failed to plot curves: {e}")
        
        print()
    
    # Final summary (SAME AS run_3stage_simple.py)
    print("\n" + "="*80)
    print("✅ ANTI-HALLUCINATION TRAINING COMPLETED!")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nAll files saved in: {cfg.save_dir}/")
    print(f"  📦 Checkpoints:")
    print(f"     - best.pt (best validation model)")
    print(f"     - last.pt (last epoch checkpoint)")
    print(f"  📊 Training logs:")
    print(f"     - training_history.csv (metrics per epoch)")
    print(f"     - training_curves.png (loss & LR plots)")
    print(f"     - loss_comparison.png (train vs val with best marker)")
    print(f"\n🔥 Anti-Hallucination Fixes Applied:")
    print(f"  ✓ Image Dropout: {args.use_image_dropout}")
    print(f"  ✓ Frequency Reweighting: {args.use_freq_reweight}")
    print(f"  ✓ Contrastive Learning: {args.use_contrastive}")
    print(f"\nNext steps:")
    print(f"  1. Compare with baseline: diff {cfg.save_dir}/training_history.csv checkpoints_simple_3stage/training_history.csv")
    print(f"  2. Test hallucination: python test_hallucination_quick.py --checkpoint {cfg.save_dir}/best.pt")
    print(f"  3. Evaluate: python eval_best.py --checkpoint {cfg.save_dir}/best.pt")
    print()


if __name__ == '__main__':
    main()
