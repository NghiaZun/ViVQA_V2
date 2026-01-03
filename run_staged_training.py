"""
AUTO STAGED TRAINING: 2-Stage Training with Scheduled Sampling
==============================================================
Script chạy tự động 2 stages:
- Stage 1: Freeze DINOv2, train fusion + decoders (epochs 0-9)
- Stage 2: Unfreeze all, end-to-end fine-tune (epochs 10-19)

Scheduled Sampling curriculum chạy xuyên suốt cả 2 stages:
  Phase 1 (0-2): Foundation - 0% generated
  Phase 2 (3-6): Exposure - 0→30% generated  
  Phase 3 (7-11): Robustness - 30→50% generated
  Phase 4 (12+): Refinement - 50→60% generated

Mỗi stage resume từ checkpoint của stage trước!
"""

import subprocess
import os
import sys
from pathlib import Path
import json


def check_checkpoint_exists(checkpoint_path):
    """Check if checkpoint exists"""
    return Path(checkpoint_path).exists()


def get_latest_checkpoint(output_dir):
    """Get latest checkpoint from a directory"""
    output_dir = Path(output_dir)
    
    # Try latest checkpoint first (contains most recent training state)
    latest_checkpoint = output_dir / 'checkpoint_latest.pt'
    if latest_checkpoint.exists():
        return str(latest_checkpoint)
    
    # Fallback to best model if latest doesn't exist
    best_model = output_dir / 'best_model.pt'
    if best_model.exists():
        return str(best_model)
    
    return None


def run_stage(stage_num, stage_config, resume_from=None):
    """Run one training stage"""
    print(f"\n{'='*80}")
    print(f"STAGE {stage_num}: {stage_config['name']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        'python', 'train_implicit_reasoning.py',
        '--output_dir', stage_config['output_dir'],
        '--batch_size', str(stage_config['batch_size']),
        '--gradient_accumulation_steps', str(stage_config['grad_accum']),
        '--num_epochs', str(stage_config['num_epochs']),
        '--learning_rate', str(stage_config['learning_rate']),
        '--alpha_reasoning_start', str(stage_config['alpha_start']),
        '--alpha_reasoning_end', str(stage_config['alpha_end']),
        '--detach_test_every', str(stage_config['detach_test_every']),
    ]
    
    # Add freeze_vision flag if specified
    if stage_config.get('freeze_vision', False):
        cmd.append('--freeze_vision')
        print(f"[INFO] 🔒 Vision encoder will be FROZEN")
    else:
        print(f"[INFO] 🔓 All parameters will be trainable")
    
    # Add resume if checkpoint exists
    if resume_from and check_checkpoint_exists(resume_from):
        cmd.extend(['--resume', resume_from])
        print(f"[INFO] Resuming from: {resume_from}")
    else:
        print(f"[INFO] Starting fresh training")
    
    # Add optional params
    if 'reasoning_bottleneck' in stage_config and stage_config['reasoning_bottleneck']:
        cmd.extend(['--reasoning_bottleneck', str(stage_config['reasoning_bottleneck'])])
    
    print(f"[INFO] Command: {' '.join(cmd)}\n")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[INFO] ✅ Stage {stage_num} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] ❌ Stage {stage_num} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n[WARNING] ⚠️ Stage {stage_num} interrupted by user")
        return False


def main():
    """Run all 3 stages automatically"""
    
    # ========================================================================
    # 2-STAGE CONFIGURATION with SCHEDULED SAMPLING
    # ========================================================================
    # Stage 1: Freeze vision, train fusion + decoders
    #   - Epochs 0-9 (10 epochs)
    #   - Higher LR (5e-5) for faster convergence
    #   - Scheduled sampling: Phase 1-2 (0% → 30%)
    #   - Alpha: 0.75 → 0.60 (high reasoning weight)
    #
    # Stage 2: Unfreeze all, end-to-end fine-tune  
    #   - Epochs 10-19 (10 epochs)
    #   - Lower LR (2e-5) for stability
    #   - Scheduled sampling: Phase 3-4 (30% → 60%)
    #   - Alpha: 0.60 → 0.50 (balanced weights)
    # ========================================================================
    
    STAGES = {
        1: {
            'name': 'STAGE 1: Freeze Vision + Train Fusion/Decoders',
            'output_dir': '/kaggle/working/stage1_freeze_vision',
            'batch_size': 4,
            'grad_accum': 16,
            'num_epochs': 10,  # Epochs 0-9
            'learning_rate': 5e-5,  # Higher LR since vision frozen
            'alpha_start': 0.75,  # High reasoning weight
            'alpha_end': 0.60,
            'detach_test_every': 2,
            'reasoning_bottleneck': None,
            'freeze_vision': True,  # 🔒 KEY: Freeze DINOv2
        },
        2: {
            'name': 'STAGE 2: Unfreeze All + End-to-End Fine-tune',
            'output_dir': '/kaggle/working/stage2_finetune_all',
            'batch_size': 4,
            'grad_accum': 16,
            'num_epochs': 20,  # Will train from epoch 10 → 19 (10 more epochs)
            'learning_rate': 2e-5,  # Lower LR for stability
            'alpha_start': 0.60,  # Balanced weights
            'alpha_end': 0.50,
            'detach_test_every': 2,
            'reasoning_bottleneck': None,
            'freeze_vision': False,  # 🔓 KEY: Unfreeze all
        }
    }
    
    # ========================================================================
    # RUN STAGES SEQUENTIALLY
    # ========================================================================
    
    print("="*80)
    print("2-STAGE TRAINING - DINOv2 + BARTpho Implicit Reasoning")
    print("="*80)
    print("\nScheduled Sampling Curriculum (20 epochs total):")
    print("  Phase 1 (epochs 0-2):  Foundation - 0% generated")
    print("  Phase 2 (epochs 3-6):  Exposure - 0→30% generated")
    print("  Phase 3 (epochs 7-11): Robustness - 30→50% generated")
    print("  Phase 4 (epochs 12+):  Refinement - 50→60% generated")
    print("\nStages to run:")
    for stage_num, config in STAGES.items():
        print(f"  Stage {stage_num}: {config['name']}")
        print(f"    - Epochs: {config['num_epochs']}")
        print(f"    - LR: {config['learning_rate']:.0e}")
        print(f"    - Alpha: {config['alpha_start']}→{config['alpha_end']}")
        print(f"    - Vision: {'🔒 Frozen' if config.get('freeze_vision') else '🔓 Trainable'}")
        
    resume_checkpoint = None
    
    for stage_num in [1, 2]:
        config = STAGES[stage_num]
        
        # Run stage
        success = run_stage(stage_num, config, resume_from=resume_checkpoint)
        
        if not success:
            print(f"\n[ERROR] Training stopped at Stage {stage_num}")
            sys.exit(1)
        
        # Get checkpoint for next stage
        resume_checkpoint = get_latest_checkpoint(config['output_dir'])
        if resume_checkpoint:
            print(f"[INFO] Will resume Stage {stage_num + 1} from: {resume_checkpoint}")
        else:
            print(f"[WARNING] No checkpoint found in {config['output_dir']}")
            if stage_num < 2:
                print(f"[WARNING] Stage {stage_num + 1} will start fresh!")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 ALL 2 STAGES COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    print("\nFinal checkpoints:")
    for stage_num, config in STAGES.items():
        checkpoint = get_latest_checkpoint(config['output_dir'])
        if checkpoint:
            print(f"  Stage {stage_num}: {checkpoint}")
    
    print("\n[INFO] Run eval_dinov2_bartpho.py on final checkpoint to evaluate!")
    print(f"[INFO] Best model: {STAGES[2]['output_dir']}/best_model.pt")
    print(f"[INFO] Latest checkpoint: {STAGES[2]['output_dir']}/checkpoint_latest.pt")


if __name__ == '__main__':
    main()
