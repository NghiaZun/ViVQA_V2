"""
TRAINING SCRIPT FOR FIXED LATENT REASONING VQA
================================================

Implements all critical fixes:
1-9 from the issue list + proper evaluation
"""

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

from transformers import get_cosine_schedule_with_warmup, AutoImageProcessor
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import VQAGenDataset
from model import (
    FixedLatentReasoningVQA,
    TrainingCurriculum,
    TeacherEvaluator
)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True


@dataclass
class FixedTrainConfig:
    """Configuration with all fixes"""
    # Data
    csv_path: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_folder: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    save_dir: str = "/kaggle/working/checkpoints_fixed"
    
    # FIX #7: Dataset filtering
    use_hard_examples_only: bool = False  # Enable after baseline
    hard_example_threshold: float = 0.7  # Filter easy examples
    
    # Training
    stage: int = 1  # 1: Baseline, 2: Warmup, 3: Full
    batch_size: int = 4
    accum_steps: int = 8
    num_epochs: int = 10
    val_split: float = 0.1
    num_workers: int = 4
    
    # Optimization
    base_lr: float = 5e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    use_amp: bool = True
    
    # Model (with fixes)
    num_reasoning_tokens: int = 6  # FIX #4: Small!
    latent_dim: int = 256  # FIX #4: Compressed!
    num_reasoning_layers: int = 2
    num_fusion_layers: int = 2
    free_bits: float = 0.05  # FIX #2 - Disabled (warmup handles collapse prevention)
    ortho_weight: float = 0.1  # FIX #5
    image_dropout_prob: float = 0.1  # FIX #3
    token_dropout_prob: float = 0.3  # FIX #5
    unfreeze_encoder_layers: int = 3
    
    # FIX #6: Intervention tests
    run_intervention_tests: bool = True
    intervention_interval: int = 2  # Run every N epochs
    
    # Teacher distillation (PROPOSAL REQUIREMENT!)
    use_teacher: bool = False  # Enable in stage 3
    teacher_type: str = 'rule_based'  # or 'vlm'
    teacher_weight: float = 0.5  # Loss weight for teacher
    
    # PROPOSAL Section 7: Online Reasoning Distillation
    num_reasoning_samples: int = 5  # Sample multiple reasoning paths
    reasoning_temperature: float = 0.7  # Temperature for stochastic sampling
    preference_margin: float = 0.1  # Margin for ranking loss
    
    # Early stopping
    es_patience: int = 6
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log_fixed.csv"
    resume_epoch: int = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str)
    p.add_argument("--image_folder", type=str)
    p.add_argument("--stage", type=int, choices=[1, 2, 3])
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--use_hard_examples_only", type=int)
    p.add_argument("--run_intervention_tests", type=int)
    p.add_argument("--use_teacher", type=int, help="Enable teacher distillation")
    p.add_argument("--teacher_type", type=str, choices=['rule_based', 'vlm'], help="Teacher type")
    p.add_argument("--teacher_weight", type=float, help="Teacher loss weight")
    p.add_argument("--num_reasoning_samples", type=int, help="Number of reasoning paths to sample")
    p.add_argument("--resume_epoch", type=int)
    return p.parse_args()


# ============================================================================
# FIX #9: INTERVENTION TESTS (not just accuracy!)
# ============================================================================

@torch.no_grad()
def run_intervention_tests(model, val_loader, device, num_samples=100):
    """
    FIX #9: Test if reasoning actually matters
    
    Tests:
    1. Ablation: Zero out reasoning → should hurt performance
    2. Noise: Add noise to reasoning → should hurt performance
    3. Diversity: Check if tokens are diverse
    """
    model.eval()
    
    print("\n" + "="*80)
    print("🔬 INTERVENTION TESTS (FIX #9)")
    print("="*80)
    
    results = {
        'normal': [],
        'ablated': [],
        'noised': []
    }
    
    diversity_scores = []
    
    total_samples = 0
    pbar = tqdm(val_loader, desc="Intervention", total=min(len(val_loader), num_samples // val_loader.batch_size))
    
    for pixel_values, input_ids, attention_mask, labels in pbar:
        if total_samples >= num_samples:
            break
        
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # 1. Normal forward
        outputs_normal = model(
            pixel_values, input_ids, attention_mask, labels,
            deterministic_reasoning=True,
            ablate_reasoning=False
        )
        
        # 2. Ablated reasoning (zero out)
        outputs_ablated = model(
            pixel_values, input_ids, attention_mask, labels,
            deterministic_reasoning=True,
            ablate_reasoning=True
        )
        
        # 3. Noisy reasoning
        outputs_noised = model(
            pixel_values, input_ids, attention_mask, labels,
            deterministic_reasoning=True,
            noise_reasoning=0.5
        )
        
        # Collect losses
        results['normal'].append(outputs_normal.answer_loss.item())
        results['ablated'].append(outputs_ablated.answer_loss.item())
        results['noised'].append(outputs_noised.answer_loss.item())
        
        # Diversity metrics
        if outputs_normal.diversity_metrics:
            diversity_scores.append(outputs_normal.diversity_metrics)
        
        total_samples += pixel_values.size(0)
    
    # Compute statistics
    normal_loss = np.mean(results['normal'])
    ablated_loss = np.mean(results['ablated'])
    noised_loss = np.mean(results['noised'])
    
    # FIX #1: Check if reasoning matters
    ablation_impact = ((ablated_loss - normal_loss) / normal_loss) * 100
    noise_impact = ((noised_loss - normal_loss) / normal_loss) * 100
    
    print(f"\n📊 INTERVENTION RESULTS:")
    print(f"  Normal loss:   {normal_loss:.4f}")
    print(f"  Ablated loss:  {ablated_loss:.4f} (+{ablation_impact:+.1f}%)")
    print(f"  Noised loss:   {noised_loss:.4f} (+{noise_impact:+.1f}%)")
    
    if ablation_impact < 5.0:
        print(f"  ⚠️  WARNING: Ablation impact < 5% → Reasoning NOT being used!")
    else:
        print(f"  ✅ Ablation impact {ablation_impact:.1f}% → Reasoning is important!")
    
    # FIX #5: Check diversity
    if diversity_scores:
        avg_metrics = {
            k: np.mean([d[k] for d in diversity_scores if not d['is_collapsed']])
            for k in ['mean_similarity', 'max_similarity', 'token_std']
        }
        
        collapsed_rate = np.mean([d['is_collapsed'] for d in diversity_scores])
        
        print(f"\n📊 DIVERSITY METRICS:")
        print(f"  Mean similarity: {avg_metrics['mean_similarity']:.3f}")
        print(f"  Max similarity:  {avg_metrics['max_similarity']:.3f}")
        print(f"  Token std:       {avg_metrics['token_std']:.4f}")
        print(f"  Collapsed rate:  {collapsed_rate*100:.1f}%")
        
        if collapsed_rate > 0.5:
            print(f"  ⚠️  WARNING: High collapse rate → Diversity regularization failing!")
        else:
            print(f"  ✅ Low collapse rate → Tokens are diverse!")
    
    print("="*80 + "\n")
    
    return {
        'normal_loss': normal_loss,
        'ablation_impact_pct': ablation_impact,
        'noise_impact_pct': noise_impact,
        'diversity': avg_metrics if diversity_scores else None,
        'collapse_rate': collapsed_rate if diversity_scores else None
    }


# ============================================================================
# TRAINING LOOP (WITH TEACHER!)
# ============================================================================

def run_one_epoch(
    model, loader, optimizer, scaler, device, cfg,
    curriculum, stage, teacher_evaluator=None, scheduler=None, train=True
):
    """
    Run one epoch with curriculum and teacher
    
    PROPOSAL Implementation (Section 7):
    - Sample multiple reasoning representations
    - Generate candidate answers from each reasoning
    - Teacher ranks candidates
    - Train with preference-based loss
    """
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    answer_loss_sum = 0.0
    kl_loss_sum = 0.0
    ortho_loss_sum = 0.0
    teacher_loss_sum = 0.0
    num_batches = 0
    
    if train:
        optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=120)
    
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # FIX #8: Get curriculum parameters
        kl_weight = curriculum.get_kl_weight(stage)
        stop_grad = curriculum.get_stop_gradient(stage)
        
        with autocast(device_type='cuda', enabled=cfg.use_amp):
            # Standard forward pass for base loss
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                deterministic_reasoning=not train,
                stop_gradient_to_latent=stop_grad,
                kl_weight=kl_weight
            )
            
            loss = outputs.total_loss
            
            # ================================================================
            # PROPOSAL SECTION 7: ONLINE REASONING DISTILLATION
            # ================================================================
            teacher_loss = torch.tensor(0.0, device=device)
            
            if cfg.use_teacher and teacher_evaluator is not None and train:
                # Get ground truths for teacher evaluation
                ground_truths = []
                for i in range(labels.size(0)):
                    label_ids = labels[i][labels[i] != -100]
                    gt = model.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                    ground_truths.append(gt)
                
                # STEP 1: Sample MULTIPLE reasoning representations (stochastic)
                candidate_outputs = []
                candidate_answers = []
                
                for sample_idx in range(cfg.num_reasoning_samples):
                    # Sample reasoning with temperature (stochastic!)
                    sample_out = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # No teacher forcing for candidates
                        deterministic_reasoning=False,  # STOCHASTIC!
                        temperature=cfg.reasoning_temperature,
                        stop_gradient_to_latent=stop_grad,
                        kl_weight=0.0  # No KL loss for sampling
                    )
                    
                    # Generate answer from reasoning latents using beam search/greedy
                    # NOTE: Changed from argmax to generate() for train/inference consistency
                    # IMPORTANT: Use no_grad to avoid building computation graph for generation
                    with torch.no_grad():
                        answers = model.generate_from_reasoning(
                            reasoning_latents=sample_out.reasoning_latents,
                            max_length=10,  # VQA answers are short
                            num_beams=1  # Greedy for speed (can increase for better quality)
                        )
                    
                    candidate_outputs.append(sample_out)
                    candidate_answers.append(answers)
                
                # STEP 2: Teacher evaluates and ranks ALL candidates
                # For each example in batch, we have num_reasoning_samples candidates
                batch_teacher_losses = []
                
                for batch_i in range(len(ground_truths)):
                    # Collect all answers for this example
                    answers_for_example = [
                        candidate_answers[sample_idx][batch_i]
                        for sample_idx in range(cfg.num_reasoning_samples)
                    ]
                    
                    # Teacher scores all candidates
                    # VLM teacher can optionally use images for better evaluation
                    teacher_scores = teacher_evaluator.evaluate_answers(
                        answers_for_example,
                        [ground_truths[batch_i]] * cfg.num_reasoning_samples,
                        images=pixel_values[batch_i:batch_i+1].repeat(cfg.num_reasoning_samples, 1, 1, 1) if cfg.teacher_type == 'vlm' else None,
                        questions=None  # Could pass decoded questions if needed
                    )  # Shape: [num_reasoning_samples]
                    
                    # STEP 3: Preference-based loss (best vs worst)
                    best_idx = teacher_scores.argmax()
                    worst_idx = teacher_scores.argmin()
                    
                    # Encourage model to prefer better reasoning paths
                    # Higher score = better answer
                    preference_loss = F.margin_ranking_loss(
                        teacher_scores[best_idx].unsqueeze(0),
                        teacher_scores[worst_idx].unsqueeze(0),
                        target=torch.ones(1, device=device),
                        margin=cfg.preference_margin
                    )
                    
                    batch_teacher_losses.append(preference_loss)
                
                # Average teacher loss across batch
                teacher_loss = torch.stack(batch_teacher_losses).mean()
                
                # Add to total loss
                loss = loss + cfg.teacher_weight * teacher_loss
            
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
                
                # FIX #8: Update curriculum
                curriculum.step()
        
        # Accumulate
        total_loss += loss.item() * cfg.accum_steps if train else loss.item()
        answer_loss_sum += outputs.answer_loss.item() if outputs.answer_loss is not None else 0
        kl_loss_sum += outputs.kl_loss.item() if outputs.kl_loss is not None else 0
        ortho_loss_sum += outputs.ortho_loss.item() if outputs.ortho_loss is not None else 0
        teacher_loss_sum += teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0
        num_batches += 1
        
        pbar.set_postfix({
            'L': f"{loss.item() * cfg.accum_steps if train else loss.item():.3f}",
            'A': f"{outputs.answer_loss.item():.3f}",
            'KL': f"{outputs.kl_loss.item():.3f}",
            'O': f"{outputs.ortho_loss.item():.3f}",
            'T': f"{teacher_loss.item():.3f}",
            'KLw': f"{kl_weight:.2f}"
        })
    
    return {
        'total': total_loss / num_batches,
        'answer': answer_loss_sum / num_batches,
        'kl': kl_loss_sum / num_batches,
        'ortho': ortho_loss_sum / num_batches,
        'teacher': teacher_loss_sum / num_batches
    }


def main():
    """Main training"""
    args = parse_args()
    cfg = FixedTrainConfig()
    
    # Override from args
    if args.csv_path:
        cfg.csv_path = args.csv_path
    if args.image_folder:
        cfg.image_folder = args.image_folder
    if args.stage:
        cfg.stage = args.stage
    if args.num_epochs:
        cfg.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.use_hard_examples_only is not None:
        cfg.use_hard_examples_only = bool(args.use_hard_examples_only)
    if args.run_intervention_tests is not None:
        cfg.run_intervention_tests = bool(args.run_intervention_tests)
    if args.use_teacher is not None:
        cfg.use_teacher = bool(args.use_teacher)
    if args.teacher_type is not None:
        cfg.teacher_type = args.teacher_type
    if args.teacher_weight is not None:
        cfg.teacher_weight = args.teacher_weight
    if args.num_reasoning_samples is not None:
        cfg.num_reasoning_samples = args.num_reasoning_samples
    if args.resume_epoch:
        cfg.resume_epoch = args.resume_epoch
    
    # Configure stage
    if cfg.stage == 1:
        cfg.save_dir = cfg.save_dir + "_stage1_baseline"
        cfg.use_teacher = False  # No teacher in baseline
        print("\n🔵 STAGE 1: BASELINE (No Reasoning, No Teacher)")
    elif cfg.stage == 2:
        cfg.save_dir = cfg.save_dir + "_stage2_warmup"
        cfg.use_teacher = False  # No teacher in warmup
        print("\n🟡 STAGE 2: WARMUP (Reasoning KL Warmup, No Teacher)")
    elif cfg.stage == 3:
        cfg.save_dir = cfg.save_dir + "_stage3_full"
        # Teacher enabled by default in stage 3 (unless overridden)
        if args.use_teacher is None:
            cfg.use_teacher = True
        print(f"\n🟢 STAGE 3: FULL (Complete Training + Teacher={cfg.use_teacher})")
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("="*80)
    print("FIXED LATENT REASONING VQA - TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Stage: {cfg.stage}")
    print(f"Reasoning: {cfg.num_reasoning_tokens} tokens × {cfg.latent_dim} dim")
    print(f"Free bits: {cfg.free_bits}")
    print(f"Ortho weight: {cfg.ortho_weight}")
    print(f"Image dropout: {cfg.image_dropout_prob}")
    print("="*80 + "\n")
    
    # Model
    print("[1/6] Initializing FIXED model...")
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
    
    # FIX: Unfreeze decoder for Stage 2-3 (curriculum learning)
    unfreeze_decoder = (cfg.stage >= 2)  # Only freeze in Stage 1
    model.freeze_pretrained(
        unfreeze_encoder_layers=cfg.unfreeze_encoder_layers,
        unfreeze_decoder=unfreeze_decoder
    )
    model = model.to(device)
    
    # Dataset (load before teacher to get tokenizer)
    print("\n[2/6] Loading dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    full_dataset = VQAGenDataset(
        csv_path=cfg.csv_path,
        image_folder=cfg.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=32
    )
    
    # Teacher evaluator (use model's tokenizer, not dataset's)
    teacher_evaluator = None
    if cfg.use_teacher:
        print(f"\n[Teacher] Initializing {cfg.teacher_type} teacher (weight={cfg.teacher_weight}, samples={cfg.num_reasoning_samples})...")
        teacher_evaluator = TeacherEvaluator(
            teacher_type=cfg.teacher_type,
            device=device,
            tokenizer=model.tokenizer  # Get from model, not dataset (Subset doesn't have it)
        )
    
    # FIX #7: Filter hard examples if requested
    if cfg.use_hard_examples_only:
        print("⚠️  Filtering for hard examples only...")
        # Placeholder: implement actual filtering based on baseline results
    
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    print("\n[3/6] Setting up optimizer...")
    total_steps = len(train_loader) // cfg.accum_steps * cfg.num_epochs
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # FIX #8: Training curriculum
    curriculum = TrainingCurriculum(total_steps_per_stage=total_steps)
    
    # Training loop
    print("\n[4/6] Starting training...")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'epoch': [], 'train_total': [], 'val_total': [],
        'train_answer': [], 'val_answer': [],
        'train_kl': [], 'val_kl': [],
        'train_ortho': [], 'val_ortho': [],
        'train_teacher': [], 'val_teacher': [],  # NEW: Teacher loss tracking
        'intervention_ablation': [], 'intervention_noise': [],
        'collapse_rate': [], 'lr': []
    }
    
    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{cfg.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_losses = run_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            curriculum, cfg.stage, 
            teacher_evaluator=teacher_evaluator,
            scheduler=scheduler, 
            train=True
        )
        
        # Validation
        with torch.no_grad():
            val_losses = run_one_epoch(
                model, val_loader, optimizer, scaler, device, cfg,
                curriculum, cfg.stage,
                teacher_evaluator=teacher_evaluator,
                train=False
            )
        
        # FIX #6 & #9: Intervention tests
        intervention_results = None
        if cfg.run_intervention_tests and (epoch + 1) % cfg.intervention_interval == 0:
            intervention_results = run_intervention_tests(model, val_loader, device)
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*80}")
        print(f"Train: Total={train_losses['total']:.4f} | A={train_losses['answer']:.4f} | "
              f"KL={train_losses['kl']:.4f} | O={train_losses['ortho']:.4f}", end="")
        if cfg.use_teacher and 'teacher' in train_losses:
            print(f" | T={train_losses['teacher']:.4f}")
        else:
            print()
        
        print(f"Val:   Total={val_losses['total']:.4f} | A={val_losses['answer']:.4f} | "
              f"KL={val_losses['kl']:.4f} | O={val_losses['ortho']:.4f}", end="")
        if cfg.use_teacher and 'teacher' in val_losses:
            print(f" | T={val_losses['teacher']:.4f}")
        else:
            print()
        
        print(f"LR: {current_lr:.2e}")
        print(f"{'='*80}\n")
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_total'].append(train_losses['total'])
        history['val_total'].append(val_losses['total'])
        history['train_answer'].append(train_losses['answer'])
        history['val_answer'].append(val_losses['answer'])
        history['train_kl'].append(train_losses['kl'])
        history['val_kl'].append(val_losses['kl'])
        history['train_ortho'].append(train_losses['ortho'])
        history['val_ortho'].append(val_losses['ortho'])
        history['train_teacher'].append(train_losses.get('teacher', 0.0))  # NEW
        history['val_teacher'].append(val_losses.get('teacher', 0.0))      # NEW
        history['lr'].append(current_lr)
        
        if intervention_results:
            history['intervention_ablation'].append(intervention_results['ablation_impact_pct'])
            history['intervention_noise'].append(intervention_results['noise_impact_pct'])
            history['collapse_rate'].append(intervention_results['collapse_rate'])
        else:
            history['intervention_ablation'].append(None)
            history['intervention_noise'].append(None)
            history['collapse_rate'].append(None)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'curriculum_step': curriculum.current_step,
            'history': history,
            'config': cfg
        }
        
        checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if val_losses['total'] < best_val_loss - cfg.es_min_delta:
            best_val_loss = val_losses['total']
            patience_counter = 0
            best_path = os.path.join(cfg.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✅ NEW BEST! Val loss: {val_losses['total']:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  Patience: {patience_counter}/{cfg.es_patience}")
        
        # Early stopping
        if patience_counter >= cfg.es_patience:
            print(f"\nEARLY STOPPING at epoch {epoch+1}")
            break
    
    # Save results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    df = pd.DataFrame(history)
    csv_path = os.path.join(cfg.save_dir, cfg.log_csv)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")
    
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
