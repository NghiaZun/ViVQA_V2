#!/usr/bin/env python3
"""
Full 3-Stage Training Pipeline for SimpleFusionVQA (Continuous)
================================================================

Run all 3 stages in a single continuous training session
Automatically switches stages based on epoch milestones

Similar to ViT + PhoBERT + ViT5 training strategy:
- Stage 1: Train fusion only (all encoders frozen)
- Stage 2: Unfreeze text encoder + decoder
- Stage 3: Unfreeze vision encoder

Expected accuracy: 63-68% (matching/exceeding old architecture)

Usage:
    python run_3stage_simple.py --csv_path <path> --image_folder <path>
"""

import torch
import os
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


def get_current_stage(epoch: int, stage1_epochs: int, stage2_epochs: int):
    """Determine current stage based on epoch number"""
    if epoch < stage1_epochs:
        return 1
    elif epoch < stage1_epochs + stage2_epochs:
        return 2
    else:
        return 3


def run_one_epoch_simple(
    model, loader, optimizer, scaler, device, cfg,
    scheduler=None, train=True
):
    """Run one epoch for SimpleFusionVQA (no KL/ortho losses)"""
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    if train:
        optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=100)
    
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # Simple forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            if train:
                loss = loss / cfg.accum_steps
        
        # Backward
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
        
        pbar.set_postfix({
            'Loss': f"{loss.item() * cfg.accum_steps if train else loss.item():.3f}"
        })
    
    return total_loss / num_batches


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full 3-stage training for SimpleFusionVQA")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--stage1_epochs", type=int, default=10,
                       help="Stage 1: Train fusion only")
    parser.add_argument("--stage2_epochs", type=int, default=10,
                       help="Stage 2: Unfreeze text encoder + decoder")
    parser.add_argument("--stage3_epochs", type=int, default=5,
                       help="Stage 3: Unfreeze vision encoder")
    parser.add_argument("--num_fusion_layers", type=int, default=3,
                       help="Number of fusion layers (default: 3 for SOTA)")
    parser.add_argument("--num_encoder_layers", type=int, default=3,
                       help="Text encoder layers to unfreeze in stage 2")
    parser.add_argument("--num_decoder_layers", type=int, default=3,
                       help="Decoder layers to unfreeze in stage 2")
    parser.add_argument("--num_vision_layers", type=int, default=3,
                       help="Vision layers to unfreeze in stage 3")
    parser.add_argument("--save_dir", type=str, default="checkpoints_simple_3stage")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from (e.g., checkpoints_simple_3stage/last.pt)")
    
    args = parser.parse_args()
    
    # Configuration
    cfg = FixedTrainConfig()
    cfg.csv_path = args.csv_path
    cfg.image_folder = args.image_folder
    cfg.batch_size = args.batch_size
    cfg.accum_steps = args.accum_steps
    cfg.save_dir = args.save_dir
    cfg.use_amp = True
    cfg.max_grad_norm = 1.0
    cfg.base_lr = 5e-5
    cfg.weight_decay = 0.05
    cfg.warmup_ratio = 0.06
    cfg.max_q_len = 64
    cfg.max_a_len = 10
    
    # Total epochs
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    stage1_end = args.stage1_epochs
    stage2_end = args.stage1_epochs + args.stage2_epochs
    
    print("="*80)
    print("CONTINUOUS 3-STAGE TRAINING: SimpleFusionVQA (SOTA Fusion)")
    if args.resume_from:
        print(f"🔄 RESUME MODE: {args.resume_from}")
    print("="*80)
    print(f"\nArchitecture:")
    print(f"  - DINOv2 vision encoder")
    print(f"  - BARTpho text encoder")
    print(f"  - Vision-First Gated Fusion ({args.num_fusion_layers} layers)")
    print(f"  - BARTpho decoder")
    print(f"\nStage boundaries:")
    print(f"  Stage 1 (Fusion Only):  Epochs 0-{stage1_end-1}")
    print(f"  Stage 2 (Text Unfreeze): Epochs {stage1_end}-{stage2_end-1}")
    print(f"  Stage 3 (Vision Unfreeze): Epochs {stage2_end}-{total_epochs-1}")
    print(f"  TOTAL: {total_epochs} epochs")
    print(f"\nExpected accuracy: 63-68% (matching ViT+PhoBERT+ViT5)")
    print("="*80 + "\n")
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Model
    print("[1/6] Loading SimpleFusionVQA model...")
    model = SimpleFusionVQA(
        num_fusion_layers=args.num_fusion_layers,
        gradient_checkpointing=True
    )
    
    # Check if resuming from checkpoint
    start_epoch = 0
    loaded_history = None
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n📂 Loading checkpoint from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        loaded_stage = checkpoint['stage']
        loaded_history = checkpoint.get('history', None)
        
        # Restore best loss if available
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        
        print(f"  ✓ Resumed from epoch {start_epoch}, stage {loaded_stage}")
        print(f"  ✓ Best val loss: {best_val_loss:.4f}")
        
        # Apply correct freeze strategy based on loaded stage
        if loaded_stage == 1:
            model.freeze_all_pretrained()
            print("  ✓ Applied Stage 1 freezing (fusion only)")
        elif loaded_stage == 2:
            model.freeze_all_pretrained()
            model.unfreeze_text_components(
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers
            )
            print("  ✓ Applied Stage 2 freezing (fusion + text)")
        elif loaded_stage == 3:
            model.freeze_all_pretrained()
            model.unfreeze_text_components(
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers
            )
            model.unfreeze_vision_encoder(num_layers=args.num_vision_layers)
            print("  ✓ Applied Stage 3 freezing (fusion + text + vision)")
    else:
        # Fresh training - Stage 1: Freeze all
        model.freeze_all_pretrained()
        if args.resume_from:
            print(f"\n⚠️  Checkpoint not found: {args.resume_from}")
            print("  Starting fresh training...\n")
    
    model = model.to(device)
    
    # Dataset
    print("\n[2/6] Loading dataset...")
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
    
    # Optimizer & Scheduler
    print("\n[3/6] Setting up optimizer...")
    
    # Determine initial learning rate based on resume stage
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        loaded_stage = checkpoint['stage']
        if loaded_stage == 1:
            initial_lr = cfg.base_lr
        elif loaded_stage == 2:
            initial_lr = cfg.base_lr * 0.5
        else:  # stage 3
            initial_lr = cfg.base_lr * 0.1
        print(f"  Resuming with LR: {initial_lr:.6f} (Stage {loaded_stage})")
    else:
        initial_lr = cfg.base_lr
    
    total_steps = len(train_loader) // cfg.accum_steps * total_epochs
    remaining_steps = len(train_loader) // cfg.accum_steps * (total_epochs - start_epoch)
    warmup_steps = int(total_steps * cfg.warmup_ratio) if start_epoch == 0 else 0
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        weight_decay=cfg.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=remaining_steps if start_epoch > 0 else total_steps
    )
    
    # Restore optimizer and scheduler state if resuming
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✓ Restored optimizer state")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Restored scheduler state")
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Training loop
    print("\n[4/6] Starting continuous training...")
    if start_epoch > 0:
        print(f"  📌 Resuming from epoch {start_epoch + 1}/{total_epochs}")
    print("="*80 + "\n")
    
    # Initialize or restore history
    if loaded_history:
        history = loaded_history
        print(f"  ✓ Restored training history ({len(history['epoch'])} epochs)")
    else:
        history = {
            'epoch': [], 'stage': [], 'train_loss': [], 'val_loss': [], 'lr': []
        }
    
    for epoch in range(start_epoch, total_epochs):
        # Determine current stage
        current_stage = get_current_stage(epoch, args.stage1_epochs, args.stage2_epochs)
        
        # Stage transition: Update model freezing (only if not already at this stage from resume)
        if epoch == stage1_end:
            print("\n" + "="*80)
            print("🟡 STAGE 2: Unfreezing Text Components")
            print("="*80)
            model.unfreeze_text_components(
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers
            )
            # Rebuild optimizer AND scheduler with new trainable params
            remaining_steps = len(train_loader) // cfg.accum_steps * (total_epochs - epoch)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.base_lr * 0.5,  # Lower LR for fine-tuning
                weight_decay=cfg.weight_decay
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,  # No warmup for stage transitions
                num_training_steps=remaining_steps
            )
            print()
        elif epoch == stage2_end:
            print("\n" + "="*80)
            print("🟢 STAGE 3: Unfreezing Vision Encoder")
            print("="*80)
            model.unfreeze_vision_encoder(num_layers=args.num_vision_layers)
            # Rebuild optimizer AND scheduler with new trainable params
            remaining_steps = len(train_loader) // cfg.accum_steps * (total_epochs - epoch)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.base_lr * 0.1,  # Even lower LR for vision fine-tuning
                weight_decay=cfg.weight_decay
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,  # No warmup for stage transitions
                num_training_steps=remaining_steps
            )
            print()
        
        print(f"EPOCH {epoch+1}/{total_epochs} (Stage {current_stage})")
        print("="*80)
        
        # Train
        train_loss = run_one_epoch_simple(
            model, train_loader, optimizer, scaler, device, cfg,
            scheduler=scheduler,
            train=True
        )
        
        # Validation
        with torch.no_grad():
            val_loss = run_one_epoch_simple(
                model, val_loader, optimizer, scaler, device, cfg,
                train=False
            )
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Stage: {current_stage}")
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['stage'].append(current_stage)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'stage': current_stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': cfg.__dict__,
            'history': history
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
        
        # Sample predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\n" + "="*80)
            print(f"📝 SAMPLE PREDICTIONS (Epoch {epoch+1}, Stage {current_stage})")
            print("="*80)
            model.eval()
            with torch.no_grad():
                # Get 3 random samples
                import random
                sample_indices = random.sample(range(len(val_loader.dataset)), min(3, len(val_loader.dataset)))
                
                for i, idx in enumerate(sample_indices):
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
                    
                    # Generate prediction
                    prediction = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=10,
                        num_beams=4
                    )[0]
                    
                    # Check match
                    match = prediction.lower().strip() == ground_truth.lower().strip()
                    partial_match = ground_truth.lower().strip() in prediction.lower().strip() or \
                                   prediction.lower().strip() in ground_truth.lower().strip()
                    
                    print(f"\n📋 Sample {i+1}:")
                    print(f"  ❓ Question: {question}")
                    print(f"  ✓ Ground Truth: {ground_truth}")
                    print(f"  🤖 Prediction: {prediction}")
                    if match:
                        print(f"  ✅ EXACT MATCH")
                    elif partial_match:
                        print(f"  🟡 PARTIAL MATCH")
                    else:
                        print(f"  ❌ WRONG")
            
            print("="*80 + "\n")
            model.train()
        
        print()
    
    # Save final history
    df = pd.DataFrame(history)
    csv_path = os.path.join(cfg.save_dir, "training_history.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("✅ ALL 3 STAGES COMPLETED!")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\nCheckpoints saved in: {cfg.save_dir}/")
    print(f"  - best.pt (best validation model - use for inference)")
    print(f"  - last.pt (last epoch checkpoint - use for resume)")
    print(f"  - training_history.csv (loss curves)")
    print("\nNext steps:")
    print(f"  1. Resume training (if interrupted):")
    print(f"     python run_3stage_simple.py --csv_path ... --image_folder ... --resume_from {cfg.save_dir}/last.pt")
    print(f"  2. Evaluate: python eval_autoregressive_cot.py --checkpoint {cfg.save_dir}/best.pt --mode val")
    print(f"  3. Test: python eval_autoregressive_cot.py --checkpoint {cfg.save_dir}/best.pt --mode test")
    print()


if __name__ == "__main__":
    main()
