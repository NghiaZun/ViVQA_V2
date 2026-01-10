"""
IMPROVED TRAINING SCRIPT: Latent Reasoning Distillation VQA
==========================================================
Training với all improvements:
1. Hierarchical reasoning (coarse + fine)
2. Auxiliary tasks (visual grounding, reasoning type)
3. Contrastive learning
4. Better monitoring & visualization
5. Curriculum learning (optional)
6. Advanced optimization strategies

Phase 1 (Current): Multi-task learning
- Answer supervision (main task)
- Visual grounding (auxiliary)
- Reasoning type classification (auxiliary)
- Contrastive learning (regularization)

Phase 2 (Future): Add preference learning with teacher model
"""

import os
import math
import time
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np

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
from model_latent_reasoning_improved import ImprovedLatentReasoningVQA


# ============================================================================
# SETUP & CONFIG
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True giảm tốc độ
    torch.backends.cudnn.benchmark = True


@dataclass
class ImprovedTrainConfig:
    """Training configuration with all improvements"""
    
    # ========== DATA ==========
    csv_path: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_folder: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    checkpoint_dir: str = "/kaggle/input/checkpoint/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working/checkpoints_improved"
    
    # ========== TRAINING ==========
    batch_size: int = 4
    accum_steps: int = 8  # Effective batch = 32
    num_epochs: int = 60
    val_split: float = 0.1
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # ========== OPTIMIZATION ==========
    base_lr: float = 5e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    use_amp: bool = True
    
    # ========== MODEL ARCHITECTURE ==========
    num_coarse_tokens: int = 8
    num_fine_tokens: int = 16
    num_reasoning_layers: int = 2
    use_stochastic_reasoning: bool = True
    latent_dim: int = 512
    num_cross_attn_layers: int = 3
    unfreeze_encoder_layers: int = 3
    num_reasoning_types: int = 8
    use_auxiliary_tasks: bool = True
    
    # ========== LOSS WEIGHTS ==========
    kl_weight: float = 0.01
    visual_grounding_weight: float = 0.05
    reasoning_type_weight: float = 0.02
    contrastive_weight: float = 0.05
    
    # Adaptive loss weights (increase over time)
    use_adaptive_weights: bool = True
    aux_warmup_epochs: int = 10  # Gradually increase aux weights
    
    # ========== REGULARIZATION ==========
    label_smoothing: float = 0.1
    reasoning_dropout: float = 0.1
    
    # ========== EARLY STOPPING ==========
    es_patience: int = 8
    es_min_delta: float = 1e-4
    
    # ========== CURRICULUM LEARNING ==========
    use_curriculum: bool = False
    curriculum_start_easy: bool = True
    
    # ========== LOGGING ==========
    log_csv: str = "train_log_improved.csv"
    curve_png: str = "training_curve_improved.png"
    loss_breakdown_png: str = "loss_breakdown.png"
    save_every_n_epochs: int = 5
    resume_epoch: int = 0
    
    # ========== MONITORING ==========
    log_reasoning_stats: bool = True  # Log reasoning latent statistics
    visualize_attention: bool = True  # Visualize attention every N epochs
    vis_every_n_epochs: int = 10


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser(description="Train Improved Latent Reasoning VQA")
    
    # Data
    p.add_argument("--csv_path", type=str)
    p.add_argument("--image_folder", type=str)
    p.add_argument("--save_dir", type=str)
    
    # Training
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--accum_steps", type=int)
    p.add_argument("--base_lr", type=float)
    
    # Model
    p.add_argument("--num_coarse_tokens", type=int)
    p.add_argument("--num_fine_tokens", type=int)
    p.add_argument("--use_stochastic_reasoning", type=int, help="0 or 1")
    p.add_argument("--use_auxiliary_tasks", type=int, help="0 or 1")
    
    # Loss weights
    p.add_argument("--kl_weight", type=float)
    p.add_argument("--visual_grounding_weight", type=float)
    p.add_argument("--reasoning_type_weight", type=float)
    p.add_argument("--contrastive_weight", type=float)
    
    # Resume
    p.add_argument("--resume_epoch", type=int)
    
    args = p.parse_args()
    return args


# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================

def build_optimizer_and_scheduler(
    model: ImprovedLatentReasoningVQA, 
    cfg: ImprovedTrainConfig,
    num_training_steps: int
):
    """
    Build optimizer with layer-wise learning rate decay
    Different LR for different components
    """
    
    # Separate parameters by component
    param_groups = [
        # Reasoning module (highest LR)
        {
            'params': model.hierarchical_reasoning.parameters(),
            'lr': cfg.base_lr,
            'name': 'reasoning'
        },
        # Fusion layers
        {
            'params': model.cross_attention_fusion.parameters(),
            'lr': cfg.base_lr * 0.8,
            'name': 'fusion'
        },
        # Vision projection
        {
            'params': model.vision_proj.parameters(),
            'lr': cfg.base_lr * 0.5,
            'name': 'vision_proj'
        },
        # Encoder unfrozen layers
        {
            'params': [p for p in model.encoder.parameters() if p.requires_grad],
            'lr': cfg.base_lr * 0.3,
            'name': 'encoder'
        },
        # LM head
        {
            'params': model.lm_head.parameters(),
            'lr': cfg.base_lr * 0.5,
            'name': 'lm_head'
        }
    ]
    
    # Add auxiliary heads if used
    if cfg.use_auxiliary_tasks:
        param_groups.extend([
            {
                'params': model.visual_grounding_head.parameters(),
                'lr': cfg.base_lr,
                'name': 'visual_grounding'
            },
            {
                'params': model.reasoning_type_head.parameters(),
                'lr': cfg.base_lr,
                'name': 'reasoning_type'
            }
        ])
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine schedule with warmup
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


# ============================================================================
# ADAPTIVE LOSS WEIGHTS
# ============================================================================

def get_adaptive_weights(epoch: int, cfg: ImprovedTrainConfig) -> Dict[str, float]:
    """
    Gradually increase auxiliary task weights over training
    Allows model to first focus on main task, then add regularization
    """
    if not cfg.use_adaptive_weights:
        return {
            'kl': cfg.kl_weight,
            'visual_grounding': cfg.visual_grounding_weight,
            'reasoning_type': cfg.reasoning_type_weight,
            'contrastive': cfg.contrastive_weight
        }
    
    # Warmup schedule: linear increase from 0 to target weight
    warmup_ratio = min(1.0, epoch / cfg.aux_warmup_epochs)
    
    return {
        'kl': cfg.kl_weight,  # KL always active
        'visual_grounding': cfg.visual_grounding_weight * warmup_ratio,
        'reasoning_type': cfg.reasoning_type_weight * warmup_ratio,
        'contrastive': cfg.contrastive_weight * warmup_ratio
    }


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def run_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    cfg,
    scheduler=None,
    train=True,
    epoch=0
):
    """
    Run one epoch with all auxiliary tasks
    
    Returns:
        Dict of all losses
    """
    if train:
        model.train()
    else:
        model.eval()
    
    # Loss accumulators
    loss_dict = {
        'total': 0.0,
        'answer': 0.0,
        'kl': 0.0,
        'visual_grounding': 0.0,
        'reasoning_type': 0.0,
        'contrastive': 0.0
    }
    
    # Reasoning statistics
    reasoning_stats = {
        'coarse_norm': [],
        'fine_norm': [],
        'coarse_std': [],
        'fine_std': []
    }
    
    num_batches = 0
    
    if train:
        optimizer.zero_grad()
        # Get adaptive weights for this epoch
        weights = get_adaptive_weights(epoch, cfg)
    else:
        weights = {
            'kl': cfg.kl_weight,
            'visual_grounding': cfg.visual_grounding_weight,
            'reasoning_type': cfg.reasoning_type_weight,
            'contrastive': cfg.contrastive_weight
        }
    
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=120)
    
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        # Move to device
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with autocast(device_type='cuda', enabled=cfg.use_amp):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                reasoning_type_labels=None,  # Could add if available
                deterministic_reasoning=not train,
                kl_weight=weights['kl'],
                visual_grounding_weight=weights['visual_grounding'],
                reasoning_type_weight=weights['reasoning_type'],
                contrastive_weight=weights['contrastive']
            )
            
            loss = outputs.total_loss
            
            # Scale loss for gradient accumulation
            if train:
                loss = loss / cfg.accum_steps
        
        # Backward pass (training only)
        if train:
            scaler.scale(loss).backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % cfg.accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    cfg.max_grad_norm
                )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Scheduler step
                if scheduler is not None:
                    scheduler.step()
        
        # Accumulate losses
        loss_dict['total'] += outputs.total_loss.item()
        if outputs.answer_loss is not None:
            loss_dict['answer'] += outputs.answer_loss.item()
        if outputs.kl_loss is not None:
            loss_dict['kl'] += outputs.kl_loss.item()
        if outputs.visual_grounding_loss is not None:
            loss_dict['visual_grounding'] += outputs.visual_grounding_loss.item()
        if outputs.reasoning_type_loss is not None:
            loss_dict['reasoning_type'] += outputs.reasoning_type_loss.item()
        if outputs.contrastive_loss is not None:
            loss_dict['contrastive'] += outputs.contrastive_loss.item()
        
        # Collect reasoning statistics
        if cfg.log_reasoning_stats and not train:  # Only during validation
            with torch.no_grad():
                coarse_norm = outputs.coarse_reasoning.norm(dim=-1).mean().item()
                fine_norm = outputs.fine_reasoning.norm(dim=-1).mean().item()
                coarse_std = outputs.coarse_reasoning.std(dim=-1).mean().item()
                fine_std = outputs.fine_reasoning.std(dim=-1).mean().item()
                
                reasoning_stats['coarse_norm'].append(coarse_norm)
                reasoning_stats['fine_norm'].append(fine_norm)
                reasoning_stats['coarse_std'].append(coarse_std)
                reasoning_stats['fine_std'].append(fine_std)
        
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{outputs.total_loss.item():.4f}',
            'ans': f'{outputs.answer_loss.item():.3f}' if outputs.answer_loss else 'N/A',
            'kl': f'{outputs.kl_loss.item():.3f}' if outputs.kl_loss else 'N/A'
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in loss_dict.items()}
    
    # Average reasoning stats
    if cfg.log_reasoning_stats and not train and reasoning_stats['coarse_norm']:
        avg_reasoning_stats = {
            k: np.mean(v) for k, v in reasoning_stats.items()
        }
    else:
        avg_reasoning_stats = None
    
    return avg_losses, avg_reasoning_stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history: Dict, save_path: str):
    """Plot comprehensive training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # 1. Total loss
    axes[0, 0].plot(epochs, history['train_total_loss'], 'b-', label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val_total_loss'], 'r-', label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Answer loss
    axes[0, 1].plot(epochs, history['train_answer_loss'], 'b-', label='Train', marker='o')
    axes[0, 1].plot(epochs, history['val_answer_loss'], 'r-', label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Answer Loss (Main Task)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. KL loss
    if 'train_kl_loss' in history and history['train_kl_loss']:
        axes[0, 2].plot(epochs, history['train_kl_loss'], 'b-', label='Train', marker='o')
        axes[0, 2].plot(epochs, history['val_kl_loss'], 'r-', label='Val', marker='s')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('KL Divergence Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Visual grounding loss
    if 'train_visual_grounding_loss' in history and history['train_visual_grounding_loss']:
        axes[1, 0].plot(epochs, history['train_visual_grounding_loss'], 'b-', marker='o')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Visual Grounding Loss (Auxiliary)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Reasoning type loss
    if 'train_reasoning_type_loss' in history and history['train_reasoning_type_loss']:
        axes[1, 1].plot(epochs, history['train_reasoning_type_loss'], 'b-', marker='o')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Reasoning Type Loss (Auxiliary)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Contrastive loss
    if 'train_contrastive_loss' in history and history['train_contrastive_loss']:
        axes[1, 2].plot(epochs, history['train_contrastive_loss'], 'b-', marker='o')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Contrastive Loss (Regularization)')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reasoning_stats(history: Dict, save_path: str):
    """Plot reasoning latent statistics"""
    if 'val_coarse_norm' not in history or not history['val_coarse_norm']:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Reasoning Latent Statistics', fontsize=14, fontweight='bold')
    
    epochs = history['epoch']
    
    # Norms
    axes[0].plot(epochs, history['val_coarse_norm'], 'b-', label='Coarse', marker='o')
    axes[0].plot(epochs, history['val_fine_norm'], 'r-', label='Fine', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_title('Reasoning Latent Norms')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Std dev
    axes[1].plot(epochs, history['val_coarse_std'], 'b-', label='Coarse', marker='o')
    axes[1].plot(epochs, history['val_fine_std'], 'r-', label='Fine', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Std Dev')
    axes[1].set_title('Reasoning Latent Diversity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function"""
    
    # Parse args and merge with config
    args = parse_args()
    cfg = ImprovedTrainConfig()
    
    # Override config with CLI args
    if args.csv_path:
        cfg.csv_path = args.csv_path
    if args.image_folder:
        cfg.image_folder = args.image_folder
    if args.save_dir:
        cfg.save_dir = args.save_dir
    if args.num_epochs:
        cfg.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.accum_steps:
        cfg.accum_steps = args.accum_steps
    if args.base_lr:
        cfg.base_lr = args.base_lr
    if args.num_coarse_tokens:
        cfg.num_coarse_tokens = args.num_coarse_tokens
    if args.num_fine_tokens:
        cfg.num_fine_tokens = args.num_fine_tokens
    if args.use_stochastic_reasoning is not None:
        cfg.use_stochastic_reasoning = bool(args.use_stochastic_reasoning)
    if args.use_auxiliary_tasks is not None:
        cfg.use_auxiliary_tasks = bool(args.use_auxiliary_tasks)
    if args.kl_weight:
        cfg.kl_weight = args.kl_weight
    if args.visual_grounding_weight:
        cfg.visual_grounding_weight = args.visual_grounding_weight
    if args.reasoning_type_weight:
        cfg.reasoning_type_weight = args.reasoning_type_weight
    if args.contrastive_weight:
        cfg.contrastive_weight = args.contrastive_weight
    if args.resume_epoch:
        cfg.resume_epoch = args.resume_epoch
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("IMPROVED LATENT REASONING VQA - TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Effective batch size: {cfg.batch_size} × {cfg.accum_steps} = {cfg.batch_size * cfg.accum_steps}")
    print(f"Reasoning: {cfg.num_coarse_tokens} coarse + {cfg.num_fine_tokens} fine tokens")
    print(f"Stochastic: {cfg.use_stochastic_reasoning}")
    print(f"Auxiliary tasks: {cfg.use_auxiliary_tasks}")
    print(f"Learning rate: {cfg.base_lr}")
    print(f"Loss weights - KL: {cfg.kl_weight}, VG: {cfg.visual_grounding_weight}, "
          f"RT: {cfg.reasoning_type_weight}, CL: {cfg.contrastive_weight}")
    print("="*80 + "\n")
    
    # ========================================================================
    # 1. MODEL
    # ========================================================================
    print("[1/5] Initializing model...")
    model = ImprovedLatentReasoningVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=cfg.num_cross_attn_layers,
        num_coarse_tokens=cfg.num_coarse_tokens,
        num_fine_tokens=cfg.num_fine_tokens,
        num_reasoning_layers=cfg.num_reasoning_layers,
        use_stochastic_reasoning=cfg.use_stochastic_reasoning,
        latent_dim=cfg.latent_dim,
        num_reasoning_types=cfg.num_reasoning_types,
        use_auxiliary_tasks=cfg.use_auxiliary_tasks,
        gradient_checkpointing=True
    )
    
    model.freeze_pretrained_weights(
        unfreeze_encoder_last_n_layers=cfg.unfreeze_encoder_layers
    )
    
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params/1e6:.1f}M")
    print(f"Trainable params: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")
    
    # ========================================================================
    # 2. DATASET
    # ========================================================================
    print("\n[2/5] Loading dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    full_dataset = VQAGenDataset(
        csv_path=cfg.csv_path,
        image_folder=cfg.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=32
    )
    
    # Split
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers
    )
    
    # ========================================================================
    # 3. OPTIMIZER & SCHEDULER
    # ========================================================================
    print("\n[3/5] Setting up optimizer...")
    total_steps = len(train_loader) // cfg.accum_steps * cfg.num_epochs
    
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, total_steps
    )
    
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Parameter groups: {len(optimizer.param_groups)}")
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # ========================================================================
    # 4. RESUME CHECKPOINT
    # ========================================================================
    start_epoch = 0
    history = {
        'epoch': [],
        'train_total_loss': [],
        'val_total_loss': [],
        'train_answer_loss': [],
        'val_answer_loss': [],
        'train_kl_loss': [],
        'val_kl_loss': [],
        'train_visual_grounding_loss': [],
        'train_reasoning_type_loss': [],
        'train_contrastive_loss': [],
        'val_coarse_norm': [],
        'val_fine_norm': [],
        'val_coarse_std': [],
        'val_fine_std': [],
        'lr': []
    }
    
    if cfg.resume_epoch > 0:
        print(f"\n[4/5] Resuming from epoch {cfg.resume_epoch}...")
        checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{cfg.resume_epoch}.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            print(f"✓ Resumed from epoch {cfg.resume_epoch}")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")
    else:
        print("\n[4/5] Starting from scratch...")
    
    # ========================================================================
    # 5. TRAINING LOOP
    # ========================================================================
    print("\n[5/5] Starting training...")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{cfg.num_epochs}")
        print(f"{'='*80}")
        
        # Show current loss weights
        current_weights = get_adaptive_weights(epoch, cfg)
        print(f"Loss weights - KL: {current_weights['kl']:.4f}, "
              f"VG: {current_weights['visual_grounding']:.4f}, "
              f"RT: {current_weights['reasoning_type']:.4f}, "
              f"CL: {current_weights['contrastive']:.4f}")
        
        # Train
        train_losses, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            scheduler=scheduler,
            train=True,
            epoch=epoch
        )
        
        # Validation
        with torch.no_grad():
            val_losses, val_reasoning_stats = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                cfg=cfg,
                train=False,
                epoch=epoch
            )
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*80}")
        print(f"Train - Total: {train_losses['total']:.4f}, Answer: {train_losses['answer']:.4f}, "
              f"KL: {train_losses['kl']:.4f}")
        print(f"Val   - Total: {val_losses['total']:.4f}, Answer: {val_losses['answer']:.4f}, "
              f"KL: {val_losses['kl']:.4f}")
        
        if cfg.use_auxiliary_tasks:
            print(f"Aux   - VG: {train_losses['visual_grounding']:.4f}, "
                  f"RT: {train_losses['reasoning_type']:.4f}, "
                  f"CL: {train_losses['contrastive']:.4f}")
        
        if val_reasoning_stats:
            print(f"Reasoning - Coarse norm: {val_reasoning_stats['coarse_norm']:.3f}, "
                  f"Fine norm: {val_reasoning_stats['fine_norm']:.3f}")
            print(f"            Coarse std: {val_reasoning_stats['coarse_std']:.3f}, "
                  f"Fine std: {val_reasoning_stats['fine_std']:.3f}")
        
        print(f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"{'='*80}\n")
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_total_loss'].append(train_losses['total'])
        history['val_total_loss'].append(val_losses['total'])
        history['train_answer_loss'].append(train_losses['answer'])
        history['val_answer_loss'].append(val_losses['answer'])
        history['train_kl_loss'].append(train_losses['kl'])
        history['val_kl_loss'].append(val_losses['kl'])
        history['train_visual_grounding_loss'].append(train_losses['visual_grounding'])
        history['train_reasoning_type_loss'].append(train_losses['reasoning_type'])
        history['train_contrastive_loss'].append(train_losses['contrastive'])
        history['lr'].append(current_lr)
        
        if val_reasoning_stats:
            history['val_coarse_norm'].append(val_reasoning_stats['coarse_norm'])
            history['val_fine_norm'].append(val_reasoning_stats['fine_norm'])
            history['val_coarse_std'].append(val_reasoning_stats['coarse_std'])
            history['val_fine_std'].append(val_reasoning_stats['fine_std'])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'history': history,
            'config': cfg
        }
        
        # Save every N epochs
        if (epoch + 1) % cfg.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_losses['total'] < best_val_loss - cfg.es_min_delta:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            best_path = os.path.join(cfg.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✓ New best model! Val loss: {val_losses['total']:.4f}")
        else:
            patience_counter += 1
            print(f"⚠ No improvement. Patience: {patience_counter}/{cfg.es_patience}")
        
        # Plot training curves every epoch
        if len(history['epoch']) > 1:
            plot_training_curves(
                history, 
                os.path.join(cfg.save_dir, cfg.curve_png)
            )
            
            if cfg.log_reasoning_stats and val_reasoning_stats:
                plot_reasoning_stats(
                    history,
                    os.path.join(cfg.save_dir, 'reasoning_stats.png')
                )
        
        # Early stopping
        if patience_counter >= cfg.es_patience:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING at epoch {epoch+1}")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"{'='*80}\n")
            break
    
    # ========================================================================
    # 6. SAVE FINAL RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Save training history CSV
    df = pd.DataFrame(history)
    csv_path = os.path.join(cfg.save_dir, cfg.log_csv)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved training log: {csv_path}")
    
    # Final plots
    plot_training_curves(
        history, 
        os.path.join(cfg.save_dir, cfg.curve_png)
    )
    print(f"✓ Saved training curves: {os.path.join(cfg.save_dir, cfg.curve_png)}")
    
    if cfg.log_reasoning_stats and history['val_coarse_norm']:
        plot_reasoning_stats(
            history,
            os.path.join(cfg.save_dir, 'reasoning_stats.png')
        )
        print(f"✓ Saved reasoning stats: {os.path.join(cfg.save_dir, 'reasoning_stats.png')}")
    
    # Save final model
    final_checkpoint_path = os.path.join(cfg.save_dir, "final_model.pt")
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': cfg,
        'best_val_loss': best_val_loss
    }
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"✓ Saved final model: {final_checkpoint_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total epochs: {len(history['epoch'])}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {history['train_total_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_total_loss'][-1]:.4f}")
    
    if history['val_coarse_norm']:
        print(f"\nFinal reasoning statistics:")
        print(f"  Coarse norm: {history['val_coarse_norm'][-1]:.3f}")
        print(f"  Fine norm: {history['val_fine_norm'][-1]:.3f}")
        print(f"  Coarse diversity (std): {history['val_coarse_std'][-1]:.3f}")
        print(f"  Fine diversity (std): {history['val_fine_std'][-1]:.3f}")
    
    # Improvement over baseline
    if len(history['val_total_loss']) > 5:
        initial_val_loss = np.mean(history['val_total_loss'][:3])
        final_val_loss = history['val_total_loss'][-1]
        improvement = (initial_val_loss - final_val_loss) / initial_val_loss * 100
        print(f"\nImprovement: {improvement:.1f}% (from {initial_val_loss:.4f} to {final_val_loss:.4f})")
    
    print(f"{'='*80}\n")
    
    # Save config
    config_dict = {
        'batch_size': cfg.batch_size,
        'accum_steps': cfg.accum_steps,
        'num_epochs': cfg.num_epochs,
        'base_lr': cfg.base_lr,
        'num_coarse_tokens': cfg.num_coarse_tokens,
        'num_fine_tokens': cfg.num_fine_tokens,
        'use_stochastic_reasoning': cfg.use_stochastic_reasoning,
        'use_auxiliary_tasks': cfg.use_auxiliary_tasks,
        'kl_weight': cfg.kl_weight,
        'visual_grounding_weight': cfg.visual_grounding_weight,
        'reasoning_type_weight': cfg.reasoning_type_weight,
        'contrastive_weight': cfg.contrastive_weight,
        'best_val_loss': best_val_loss,
        'total_epochs_trained': len(history['epoch'])
    }
    
    config_path = os.path.join(cfg.save_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved config: {config_path}")
    
    print("\n🎉 All done! Check the checkpoint directory for results.")
    print(f"📁 {cfg.save_dir}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Progress has been saved in checkpoints")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise