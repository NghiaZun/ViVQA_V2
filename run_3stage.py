#!/usr/bin/env python3
"""
Full 3-Stage Training Pipeline (Continuous)
============================================

Run all 3 stages in a single continuous training session
Automatically switches stages based on epoch milestones

Usage:
    python run_full_3stage_training.py --csv_path <path> --image_folder <path>
"""

import torch
import os
from dataclasses import dataclass
from train import (
    FixedTrainConfig, set_seed, run_one_epoch, TrainingCurriculum
)
from model import FixedLatentReasoningVQA, TeacherEvaluator
from dataset import VQAGenDataset
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from tqdm import tqdm


def get_current_stage(epoch: int, stage1_epochs: int, stage2_epochs: int):
    """Determine current stage based on epoch number"""
    if epoch < stage1_epochs:
        return 1
    elif epoch < stage1_epochs + stage2_epochs:
        return 2
    else:
        return 3


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full 3-stage training in one session")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--teacher_type", type=str, default="rule_based", 
                       choices=["rule_based", "vlm"])
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage3_epochs", type=int, default=20)
    parser.add_argument("--num_reasoning_samples", type=int, default=3)
    parser.add_argument("--max_kl_weight", type=float, default=15.0,
                       help="Max KL weight (15.0 with 0.01 factor → effective 0.15)")
    
    args = parser.parse_args()
    
    # Configuration
    cfg = FixedTrainConfig()
    cfg.csv_path = args.csv_path
    cfg.image_folder = args.image_folder
    cfg.batch_size = args.batch_size
    cfg.teacher_type = args.teacher_type
    cfg.num_reasoning_samples = args.num_reasoning_samples
    
    # Add missing attributes
    cfg.max_q_len = 64  # Max question length
    cfg.max_a_len = 10  # Max answer length (VQA answers are short: 1-3 words)
    cfg.learning_rate = cfg.base_lr  # Add learning_rate alias
    cfg.use_teacher = True  # Enable teacher in Stage 3
    cfg.teacher_weight = 0.5  # Teacher loss weight
    cfg.reasoning_temperature = 0.7  # Temperature for stochastic sampling
    cfg.preference_margin = 0.1  # Margin for ranking loss
    cfg.use_amp = True  # Enable automatic mixed precision
    
    # Total epochs
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    stage1_end = args.stage1_epochs
    stage2_end = args.stage1_epochs + args.stage2_epochs
    
    print("="*80)
    print("CONTINUOUS 3-STAGE TRAINING")
    print("="*80)
    print(f"\nStage boundaries:")
    print(f"  Stage 1 (Baseline): Epochs 0-{stage1_end-1}")
    print(f"  Stage 2 (Warmup):   Epochs {stage1_end}-{stage2_end-1}")
    print(f"  Stage 3 (Full):     Epochs {stage2_end}-{total_epochs-1}")
    print(f"  TOTAL: {total_epochs} epochs")
    print(f"\nKL weight config:")
    print(f"  Max KL weight: {args.max_kl_weight} (effective = {args.max_kl_weight * 0.01:.3f} due to 0.01 factor in loss)")
    print(f"  Stage 1: KL weight = 0.0")
    print(f"  Stage 2: KL weight = 0.0 → {args.max_kl_weight} (linear warmup)")
    print(f"  Stage 3: KL weight = {args.max_kl_weight}")
    print("="*80 + "\n")
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    print("[1/6] Loading model...")
    model = FixedLatentReasoningVQA(
        num_reasoning_tokens=cfg.num_reasoning_tokens,
        latent_dim=cfg.latent_dim,
        num_reasoning_layers=cfg.num_reasoning_layers,
        num_fusion_layers=cfg.num_fusion_layers,
        free_bits=cfg.free_bits,
        ortho_weight=cfg.ortho_weight,
        image_dropout_prob=cfg.image_dropout_prob,
        token_dropout_prob=cfg.token_dropout_prob,
        gradient_checkpointing=True
    )
    
    # Freeze with decoder unfrozen (will handle per-stage later)
    model.freeze_pretrained(unfreeze_encoder_layers=3, unfreeze_decoder=True)
    model = model.to(device)
    
    # Dataset
    print("\n[2/6] Loading dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    full_dataset = VQAGenDataset(
        csv_path=cfg.csv_path,
        image_folder=cfg.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',  # Pass name, not object
        max_q_len=cfg.max_q_len,
        max_a_len=cfg.max_a_len
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Teacher (initialize once, use in Stage 3)
    print("\n[3/6] Setting up teacher...")
    teacher_evaluator = TeacherEvaluator(
        teacher_type=args.teacher_type,
        device=device,
        tokenizer=model.tokenizer
    )
    
    # Optimizer & Scheduler
    print("\n[4/6] Setting up optimizer...")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Curriculum (warmup over ENTIRE Stage 2, not just 1 epoch!)
    total_stage2_steps = len(train_loader) * args.stage2_epochs
    curriculum = TrainingCurriculum(
        total_steps_per_stage=total_stage2_steps,  # Total batches in Stage 2
        max_kl_weight=args.max_kl_weight  # Tunable KL weight
    )
    
    # Training loop
    print("\n[5/6] Starting continuous training...")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(total_epochs):
        # Determine current stage
        current_stage = get_current_stage(epoch, args.stage1_epochs, args.stage2_epochs)
        
        # Stage transition announcements
        if epoch == 0:
            print("\n" + "="*80)
            print("🔵 STAGE 1: BASELINE (No Reasoning)")
            print("="*80 + "\n")
        elif epoch == stage1_end:
            print("\n" + "="*80)
            print("🟡 STAGE 2: WARMUP (Reasoning KL Warmup)")
            print("="*80 + "\n")
            curriculum.current_step = 0  # Reset for warmup
        elif epoch == stage2_end:
            print("\n" + "="*80)
            print("🟢 STAGE 3: FULL (Complete + Teacher)")
            print("="*80 + "\n")
            curriculum.current_step = 0  # Reset for full training
        
        # Determine if teacher should be used
        use_teacher_this_epoch = (current_stage == 3)
        
        print(f"EPOCH {epoch+1}/{total_epochs} (Stage {current_stage})")
        print("="*80)
        
        # Train
        train_losses = run_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            curriculum, current_stage,
            teacher_evaluator=teacher_evaluator if use_teacher_this_epoch else None,
            scheduler=scheduler,
            train=True
        )
        
        # Validation
        with torch.no_grad():
            val_losses = run_one_epoch(
                model, val_loader, optimizer, scaler, device, cfg,
                curriculum, current_stage,
                teacher_evaluator=None,
                train=False
            )
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        kl_weight = curriculum.get_kl_weight(current_stage)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.4f}, Answer: {train_losses['answer']:.4f}, "
              f"KL: {train_losses['kl']:.4f}, Teacher: {train_losses['teacher']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, Answer: {val_losses['answer']:.4f}, "
              f"KL: {val_losses['kl']:.4f}")
        print(f"  LR: {current_lr:.6f}, KL weight: {kl_weight:.2f}, Stage: {current_stage}")
        
        # Prepare checkpoint dict
        os.makedirs(cfg.save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'stage': current_stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': cfg.__dict__
        }
        
        # Save best model (always check and update)
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = os.path.join(cfg.save_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"  ✅ New best model saved! (val_loss: {best_val_loss:.4f})")
        
        # Save last checkpoint (overwrite each epoch to save disk space)
        last_path = os.path.join(cfg.save_dir, "last.pt")
        torch.save(checkpoint, last_path)
        
        # Sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n" + "="*80)
            print(f"📝 SAMPLE PREDICTIONS (Epoch {epoch+1}, Stage {current_stage})")
            print("="*80)
            model.eval()
            with torch.no_grad():
                # Get 3 random samples from validation set
                import random
                sample_indices = random.sample(range(len(val_loader.dataset)), min(3, len(val_loader.dataset)))
                
                for i, idx in enumerate(sample_indices):
                    # Get sample from dataset
                    sample = val_loader.dataset[idx]
                    pixel_values = sample[0].unsqueeze(0).to(device)
                    input_ids = sample[1].unsqueeze(0).to(device)
                    attention_mask = sample[2].unsqueeze(0).to(device)
                    labels = sample[3].unsqueeze(0)
                    
                    # Get ground truth
                    gt_tokens = labels[0][labels[0] != -100]
                    ground_truth = model.tokenizer.decode(gt_tokens, skip_special_tokens=True)
                    
                    # Get question
                    question = model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    
                    # Forward pass to get reasoning info
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        deterministic_reasoning=True,
                        kl_weight=0.0
                    )
                    
                    # Generate prediction from reasoning latents
                    prediction = model.generate_from_reasoning(
                        reasoning_latents=outputs.reasoning_latents,
                        max_length=10,
                        num_beams=1
                    )[0]
                    
                    # Check match
                    match = prediction.lower().strip() == ground_truth.lower().strip()
                    partial_match = ground_truth.lower().strip() in prediction.lower().strip() or \
                                   prediction.lower().strip() in ground_truth.lower().strip()
                    
                    print(f"\n📋 Sample {i+1}:")
                    print(f"  ❓ Question: {question}")
                    print(f"  ✓ Ground Truth: {ground_truth}")
                    print(f"  🤖 Prediction: {prediction}")
                    if outputs.kl_loss is not None:
                        print(f"  📊 KL: {outputs.kl_loss.item():.4f}")
                    if match:
                        print(f"  ✅ EXACT MATCH")
                    elif partial_match:
                        print(f"  🟡 PARTIAL MATCH")
                    else:
                        print(f"  ❌ WRONG")
            
            print("="*80 + "\n")
            model.train()
        
        scheduler.step()
        curriculum.step()
        print()
    
    print("\n" + "="*80)
    print("✅ ALL 3 STAGES COMPLETED!")
    print("="*80)
    print(f"\nCheckpoints saved in: {cfg.save_dir}/")
    print(f"  - best.pt (best validation model - use for inference)")
    print(f"  - last.pt (last epoch checkpoint - use for resume)")
    print()


if __name__ == "__main__":
    main()
