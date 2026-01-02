"""
AUTO STAGED TRAINING: Run all 3 stages automatically
=====================================================
Script này sẽ chạy tự động cả 3 stages:
- Stage 1: Warmup (5 epochs)
- Stage 2: Main (15 epochs) 
- Stage 3: Fine-tune (10 epochs)

Mỗi stage sẽ resume từ checkpoint của stage trước!
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
    
    # Try best model first
    best_model = output_dir / 'best_model.pt'
    if best_model.exists():
        return str(best_model)
    
    # Try latest checkpoint
    latest_checkpoint = output_dir / 'checkpoint_latest.pt'
    if latest_checkpoint.exists():
        return str(latest_checkpoint)
    
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
    # CONFIGURATION FOR ALL STAGES
    # ========================================================================
    
    STAGES = {
        1: {
            'name': 'WARMUP',
            'output_dir': '/kaggle/working/stage1_warmup',
            'batch_size': 4,
            'grad_accum': 16,
            'num_epochs': 5,
            'learning_rate': 1e-5,
            'alpha_start': 0.5,
            'alpha_end': 0.5,  # Fixed alpha
            'detach_test_every': 1,  # Test every epoch
            'reasoning_bottleneck': None,
        },
        2: {
            'name': 'MAIN TRAINING',
            'output_dir': '/kaggle/working/stage2_main',
            'batch_size': 4,
            'grad_accum': 16,
            'num_epochs': 15,  # Will train from epoch 5 → 15 (10 more epochs)
            'learning_rate': 5e-5,  # INCREASED to break out of local minima
            'alpha_start': 0.75,  # INCREASED to force reasoning learning
            'alpha_end': 0.5,  # Higher minimum alpha
            'detach_test_every': 2,  # Test more frequently
            'reasoning_bottleneck': 384,  # Force information compression
        },
        3: {
            'name': 'FINE-TUNING',
            'output_dir': '/kaggle/working/stage3_finetune',
            'batch_size': 4,
            'grad_accum': 16,
            'num_epochs': 30,  # Will train from epoch 20 → 30 (10 more epochs)
            'learning_rate': 5e-6,
            'alpha_start': 0.2,
            'alpha_end': 0.1,  # Anneal
            'detach_test_every': 2,
            'reasoning_bottleneck': None,
        }
    }
    
    # ========================================================================
    # RUN STAGES SEQUENTIALLY
    # ========================================================================
    
    print("="*80)
    print("AUTO STAGED TRAINING - DINOv2 + BARTpho Implicit Reasoning")
    print("="*80)
    print("\nStages to run:")
    for stage_num, config in STAGES.items():
        print(f"  Stage {stage_num}: {config['name']}")
        print(f"    - Epochs: {config['num_epochs']}")
        print(f"    - LR: {config['learning_rate']:.0e}")
        print(f"    - Alpha: {config['alpha_start']}→{config['alpha_end']}")
        
    resume_checkpoint = None
    
    for stage_num in [1, 2, 3]:
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
            if stage_num < 3:
                print(f"[WARNING] Stage {stage_num + 1} will start fresh!")
    
    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎉 ALL STAGES COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    print("\nFinal checkpoints:")
    for stage_num, config in STAGES.items():
        checkpoint = get_latest_checkpoint(config['output_dir'])
        if checkpoint:
            print(f"  Stage {stage_num}: {checkpoint}")
    
    print("\n[INFO] Run SANITY_TESTS.py on final checkpoint to validate!")
    print(f"[INFO] Best model: {STAGES[3]['output_dir']}/best_model.pt")


if __name__ == '__main__':
    main()
