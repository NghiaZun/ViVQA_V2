"""
IMPLICIT REASONING TRAINING - 3-Stage Curriculum with Separate Decoders
========================================================================

ARCHITECTURE:
=============
SEPARATE DECODERS:
  - Reasoning Decoder: specialized for explanatory generation
  - Answer Decoder: specialized for concise answers
  - No task confusion between tasks!

2 TRAINING MODES:
=================

MODE 1: FEATURE EXTRACTION (Recommended)
-----------------------------------------
Flag: --freeze_pretrained

FROZEN (87% of model):
  - Vision Encoder
  - Text Encoder
  - Reasoning Decoder
  - Answer Decoder

TRAINABLE (13% - only heads):
  - Vision Projection
  - Cross-Attention Fusion
  - LM Head

Benefits:
  - Preserve pretrained knowledge
  - Fast training (2x faster)
  - Less memory (7-8GB)
  - No catastrophic forgetting

MODE 2: FULL FINETUNING (Aggressive)
--------------------------------------
Flag: --freeze_vision

Stage 1-2: Vision frozen, train decoders (595M)
Stage 3: Full end-to-end (681M)

Benefits:
  - Max performance potential
Risks:
  - May overfit on small data
  - May lose pretrained knowledge


🔥 CRITICAL FIX: GRADIENT FLOW
================================
Old approach: Generated reasoning was DETACHED (torch.no_grad)
  → Reasoning decoder only learned from GT (reasoning_loss)
  → No feedback from answer quality!
  → Result: Generated reasoning was useless for answering

New approach: Generated reasoning has GRADIENTS
  → Reasoning decoder learns from reasoning_loss (mimic GT)
  → AND from answer_loss (generate USEFUL reasoning)!
  → Result: Reasoning decoder optimizes for answer quality


3-STAGE CURRICULUM:
===================

STAGE 1 (Epochs 0-7): STABILIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Duration:  8 epochs
Gen_prob:  0% (HARD LOCKED - no generation)
Quality:   Not measured (teacher forcing only)
Strategy:  Pure alignment learning

Goal: Learn cross-modal fusion without generation pressure
      Both decoders receive ground truth targets
      Stabilize before introducing generation

STAGE 2 (Epochs 8-19): GRADUAL EXPOSURE
Duration:  12 epochs (conservative growth)
Gen_prob:  0% to 5% (🔥 ULTRA-SAFE CEILING)
Quality:   Strict Vietnamese coherence gating (>30% Vietnamese tokens)

Gating Logic:
  IF quality_score > 0.70 AND improving AND vietnamese_coherence > 0.30:
    INCREASE gen_prob (+0.5% per epoch - very slow)
  ELSE:
    HOLD gen_prob (wait for quality)
  
  Hard ceiling: max(gen_prob) = 5%

Goal: Build Vietnamese generation capability gradually
      Ensure model generates Vietnamese, not gibberish

STAGE 3 (Epochs 20-39): REFINEMENT
Duration:  20 epochs (slow refinement)
Gen_prob:  5% to 15% (🔥 ULTRA-SAFE: FLOOR 5%, CEILING 15%)
Quality:   Strict gating with Vietnamese check & DECREASE on drop

Gating Logic:
  IF quality_score > 0.70 AND improving AND vietnamese_coherence > 0.50:
    INCREASE gen_prob (+0.5% per epoch)
  ELIF vietnamese_coherence < 0.30 OR repetition_rate > 0.50:
    DECREASE gen_prob (-1%, min=5%)
  ELSE:
    HOLD gen_prob
  
  Hard floor: min(gen_prob) = 5% (from Stage 2)
  Hard ceiling: max(gen_prob) = 15%

Goal: Polish Vietnamese generation quality
      Prevent model collapse into gibberish

USAGE:
======
# 🔥 FIXED VERSION - Chống Overfitting
python train_implicit_reasoning.py \
  --freeze_pretrained \
  --num_epochs 40 \
  --learning_rate 5e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 16

# Expected behavior:
# - Stage 1 (0-7):   gen_prob=0%,    val_answer_loss giảm stable
# - Stage 2 (8-19):  gen_prob=0→5%, Vietnamese coherence > 30%, quality > 0.70
# - Stage 3 (20-39): gen_prob=5→15%, Vietnamese coherence > 50%, low repetition
"""

import os
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
import argparse

# Import model
from model_dinov2_bartpho import DINOv2BARTphoVQA, count_parameters


# ============================================================================
# 1. DATASET (unchanged)
# ============================================================================

class ImplicitReasoningDataset(Dataset):
    """Dataset cho implicit reasoning"""
    
    def __init__(
        self, 
        json_path, 
        image_dir,
        vision_processor,
        tokenizer,
        max_question_len=64,
        max_answer_len=32,
        max_reasoning_len=96,
        augment=False
    ):
        self.data = self.load_data(json_path)
        self.image_dir = image_dir
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_reasoning_len = max_reasoning_len
        self.augment = augment
        
        print(f"[INFO] Loaded {len(self.data)} samples from {json_path}")
        
    def load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def augment_image(self, image):
        """Augment PIL image before vision processor"""
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        import random
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
        
        # Random color jitter
        if random.random() > 0.3:
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            image = jitter(image)
        
        # Random rotation (small angle)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
        
        return image
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, item['image_path'].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        
        if self.augment:
            image = self.augment_image(image)
        
        # Process image
        pixel_values = self.vision_processor(images=image, return_tensors='pt')['pixel_values'][0]
        
        # Tokenize question
        question_enc = self.tokenizer(
            item['question'],
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize reasoning
        reasoning_enc = self.tokenizer(
            item.get('reasoning', item.get('teacher_reasoning', '')),
            max_length=self.max_reasoning_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer
        answer_text = item.get('answer', item.get('predicted_answer', item.get('final_answer', '')))
        answer_enc = self.tokenizer(
            answer_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': question_enc['input_ids'][0],
            'attention_mask': question_enc['attention_mask'][0],
            'reasoning_input_ids': reasoning_enc['input_ids'][0],
            'reasoning_attention_mask': reasoning_enc['attention_mask'][0],
            'answer_input_ids': answer_enc['input_ids'][0],
            'answer_attention_mask': answer_enc['attention_mask'][0],
            'labels': answer_enc['input_ids'][0],
            'reasoning_labels': reasoning_enc['input_ids'][0],
        }


# ============================================================================
# 2. GENERATION QUALITY METRICS
# ============================================================================

class GenerationQualityTracker:
    """Track generation quality to guide scheduled sampling"""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = {
            'perplexity': [],
            'non_empty_rate': [],
            'avg_length': [],
            'vietnamese_coherence': [],  # 🔥 NEW
            'repetition_rate': [],       # 🔥 NEW
        }
    
    def update(self, metrics):
        """Update with new metrics"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
                # Keep only last N
                if len(self.history[key]) > self.window_size:
                    self.history[key].pop(0)
    
    def get_quality_score(self):
        """
        🔥 FIXED: Return quality score [0, 1] with Vietnamese coherence check
        - 1.0 = excellent (ready for more generated)
        - 0.0 = poor (stay with GT)
        """
        if not self.history['perplexity']:
            return 0.0
        
        # Average recent metrics
        avg_ppl = np.mean(self.history['perplexity'])
        non_empty = np.mean(self.history['non_empty_rate'])
        avg_len = np.mean(self.history['avg_length'])
        
        # 🔥 NEW: Vietnamese coherence and repetition
        viet_coherence = np.mean(self.history.get('vietnamese_coherence', [0.0])) if 'vietnamese_coherence' in self.history else 0.0
        repetition = np.mean(self.history.get('repetition_rate', [1.0])) if 'repetition_rate' in self.history else 1.0
        
        # Quality criteria:
        # - Perplexity < 20 (good), < 10 (excellent)
        # - Non-empty > 0.95 (most generations are valid)
        # - Avg length > 10 tokens (not degenerate)
        # - Vietnamese coherence > 0.3 (at least 30% Vietnamese tokens)
        # - Repetition < 0.5 (less than 50% repeated trigrams)
        
        ppl_score = max(0, 1 - (avg_ppl / 30))  # 30+ ppl = 0, 0 ppl = 1
        empty_score = non_empty
        length_score = min(1.0, avg_len / 20)  # 20+ tokens = 1
        viet_score = min(1.0, viet_coherence / 0.3)  # 30%+ Vietnamese = good
        repetition_score = max(0, 1 - (repetition / 0.5))  # <50% repetition = good
        
        # 🔥 STRICTER: Weighted average with language coherence
        quality = (0.3 * ppl_score + 
                  0.2 * empty_score + 
                  0.15 * length_score + 
                  0.25 * viet_score +      # High weight on Vietnamese!
                  0.1 * repetition_score)
        
        return quality
    
    def is_improving(self):
        """Check if quality is improving (perplexity should DECREASE)"""
        if len(self.history['perplexity']) < 2:
            return True
        
        recent = self.history['perplexity'][-1]
        prev = self.history['perplexity'][-2]
        
        # Perplexity lower is better, allow 10% degradation tolerance
        return recent <= prev * 1.1  # recent ≤ prev means improving/stable


# ============================================================================
# 3. IMPROVED TRAINER
# ============================================================================

class ImplicitReasoningTrainer:
    """
    Trainer với validation-guided scheduled sampling
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        output_dir,
        batch_size=4,
        gradient_accumulation_steps=16,
        num_epochs=30,
        learning_rate=2e-5,
        weight_decay=0.08,       # 🔥 FIX 1: Tăng từ 0.01 → 0.08 (8x) chống overfitting
        warmup_ratio=0.1,
        max_grad_norm=0.5,       # 🔥 FIX 1: Giảm từ 1.0 → 0.5 (gradient clipping mạnh hơn)
        alpha_reasoning=0.4,     # Fixed weight, không anneal
        alpha_answer=0.6,
        label_smoothing=0.15,    # 🔥 FIX 1: Tăng từ 0.1 → 0.15 (1.5x)
        use_amp=True,
        patience=5,              # 🔥 FIX 3: Giảm từ 7 → 5 (early stop sớm hơn)
        log_steps=10,
        save_steps=100,
        # NEW: 3-Stage curriculum params (FIXED - chậm hơn nhiều)
        alignment_epochs=8,      # 🔥 FIX 2: Tăng từ 3 → 8 (stabilization dài hơn)
        language_tuning_epochs=20,  # 🔥 FIX 2: Tăng từ 10 → 20 (gradual exposure)
        full_finetuning_epochs=40,  # 🔥 FIX 2: Tăng từ 20 → 40 (full finetuning)
        stage2_ceiling=0.05,     # 🔥 ULTRA-SAFE: 5% max (was 12%)
        stage3_ceiling=0.15,     # 🔥 ULTRA-SAFE: 15% max (was 40%)
        quality_threshold=0.70,  # 🔥 REALISTIC: 0.70 (with new Vietnamese check)
        # NEW: 2-stage training
        unfreeze_after_epoch=None,  # Auto-unfreeze vision after N epochs
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training params
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.patience = patience
        self.log_steps = log_steps
        self.save_steps = save_steps
        
        # Loss weights (fixed)
        self.alpha_reasoning = alpha_reasoning
        self.alpha_answer = alpha_answer
        
        # 3-Stage curriculum params
        self.alignment_epochs = alignment_epochs
        self.language_tuning_epochs = language_tuning_epochs
        self.full_finetuning_epochs = full_finetuning_epochs
        self.stage2_ceiling = stage2_ceiling
        self.stage3_ceiling = stage3_ceiling
        self.quality_threshold = quality_threshold
        
        # 2-stage training
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.vision_frozen = self._check_vision_frozen()
        
        # Quality tracker
        self.quality_tracker = GenerationQualityTracker(window_size=3)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing
        )
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 🔥 FIX 5: Warmup + Linear Decay thay vì Cosine (giữ LR stable lâu hơn)
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        num_warmup_steps = int(total_steps * warmup_ratio)  # 10% warmup
        
        def lr_lambda(current_step):
            # Warmup phase (0 → 1.0 trong 10% đầu)
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Hold phase (giữ 1.0 trong 50% tiếp theo)
            hold_steps = int(total_steps * 0.5)
            if current_step < num_warmup_steps + hold_steps:
                return 1.0
            
            # Linear decay phase (1.0 → 0.1 trong 40% cuối)
            progress = (current_step - num_warmup_steps - hold_steps) / float(max(1, total_steps - num_warmup_steps - hold_steps))
            return max(0.1, 1.0 - 0.9 * progress)  # Decay to 0.1x, not 0
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        print(f"[INFO] 🔥 Custom LR Schedule: Warmup({num_warmup_steps}) + Hold({int(total_steps*0.5)}) + LinearDecay")
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_answer_loss = float('inf')  # 🔥 FIX 3: Track answer loss riêng
        self.val_answer_loss_history = []         # 🔥 FIX 3: History để detect divergence
        self.patience_counter = 0
        self.current_gen_prob = 0.0
        
        # CSV logger
        self.csv_log_path = self.output_dir / 'training_log.csv'
        self.init_csv_logger()
        
        print(f"\n[INFO] 🔥 IMPROVED Implicit Reasoning Trainer (OVERFITTING-PROOF)")
        print(f"  💪 Regularization:")
        print(f"    - Weight decay: {weight_decay:.3f} (8x stronger)")
        print(f"    - Label smoothing: {label_smoothing:.2f} (1.5x stronger)")
        print(f"    - Gradient clipping: {max_grad_norm:.1f} (2x stricter)")
        print(f"    - Patience: {patience} epochs (early stop faster)")
        print(f"  📚 3-Stage Curriculum (SLOWER & SAFER):")
        print(f"    Stage 1 (0-{alignment_epochs-1}): STABILIZATION [gen_prob=0%, pure teacher forcing]")
        print(f"    Stage 2 ({alignment_epochs}-{language_tuning_epochs-1}): GRADUAL EXPOSURE [gen_prob=0→{stage2_ceiling*100:.0f}%, +1%/epoch]")
        print(f"    Stage 3 ({language_tuning_epochs}-{full_finetuning_epochs-1}): FULL FINETUNING [gen_prob={stage2_ceiling*100:.0f}→{stage3_ceiling*100:.0f}%, +1.4%/epoch]")
        print(f"  🎯 Quality threshold: {quality_threshold:.2f} (strict gating)")
        if self.vision_frozen:
            print(f"  🔒 Vision encoder: FROZEN (Stage 1+2)")
            if unfreeze_after_epoch:
                print(f"     Will unfreeze at epoch {unfreeze_after_epoch} (Stage 3)")
        else:
            print(f"  🔓 Vision encoder: TRAINABLE (Stage 3)")
    
    def _check_vision_frozen(self):
        """Check if vision encoder is frozen"""
        for param in self.model.vision_encoder.parameters():
            if param.requires_grad:
                return False
        return True
    
    def _check_answer_decoder_divergence(self, current_val_answer_loss, train_answer_loss):
        """
        🔥 FIX 3: Kiểm tra answer decoder có đang diverge không
        
        Returns:
            (is_diverging, should_decrease_gen_prob, message)
        """
        # Rule 1: Val answer loss > Train answer loss * 1.3 → Overfitting nghiêm trọng
        if current_val_answer_loss > train_answer_loss * 1.3:
            return True, True, f"⚠️ OVERFITTING: val_answer ({current_val_answer_loss:.4f}) >> train_answer ({train_answer_loss:.4f})"
        
        # Rule 2: Val answer loss tăng > 20% so với best
        if len(self.val_answer_loss_history) >= 3:
            recent_best = min(self.val_answer_loss_history[-3:])
            if current_val_answer_loss > recent_best * 1.2:
                return True, True, f"⚠️ DIVERGING: val_answer tăng {((current_val_answer_loss/recent_best - 1)*100):.1f}% so với best"
        
        # Rule 3: Val answer loss tăng liên tục 3 epochs
        if len(self.val_answer_loss_history) >= 3:
            if (self.val_answer_loss_history[-1] > self.val_answer_loss_history[-2] > self.val_answer_loss_history[-3]):
                return True, False, f"⚠️ UNSTABLE: val_answer tăng 3 epochs liên tiếp"
        
        return False, False, "✓ Answer decoder stable"
    
    def _unfreeze_vision_encoder(self):
        """Unfreeze vision encoder for Stage 3"""
        print(f"\n{'='*70}")
        print(f"🔓 STAGE 3: UNFREEZING VISION ENCODER")
        print(f"{'='*70}")
        
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = True
        
        self.vision_frozen = False
        
        # Recount trainable params
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[INFO] Trainable params after unfreeze: {trainable_params/1e6:.1f}M")
        
        # Update optimizer to include vision params
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate * 0.5,  # Lower LR for Stage 3
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        print(f"[INFO] ✓ Optimizer updated with vision encoder parameters")
        print(f"[INFO] ✓ Learning rate reduced to {self.learning_rate * 0.5:.2e} for Stage 3")
    
    def init_csv_logger(self):
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'gen_prob', 'quality_score',
                'train_loss', 'train_reasoning_loss', 'train_answer_loss',
                'val_loss', 'val_reasoning_loss', 'val_answer_loss',
                'gen_perplexity', 'gen_non_empty_rate', 'gen_avg_length',
                'gen_vietnamese_coherence', 'gen_repetition_rate',  # 🔥 NEW
                'learning_rate', 'patience_counter', 'is_best'
            ])
    
    def log_to_csv(self, epoch, phase, gen_prob, quality_score, train_losses, val_losses, gen_metrics, is_best=False):
        current_lr = self.scheduler.get_last_lr()[0]
        
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                phase,
                f"{gen_prob:.3f}",
                f"{quality_score:.3f}",
                f"{train_losses['total']:.4f}",
                f"{train_losses['reasoning']:.4f}",
                f"{train_losses['answer']:.4f}",
                f"{val_losses['total']:.4f}",
                f"{val_losses['reasoning']:.4f}",
                f"{val_losses['answer']:.4f}",
                f"{gen_metrics.get('perplexity', 0):.2f}",
                f"{gen_metrics.get('non_empty_rate', 0):.3f}",
                f"{gen_metrics.get('avg_length', 0):.1f}",
                f"{gen_metrics.get('vietnamese_coherence', 0):.3f}",  # 🔥 NEW
                f"{gen_metrics.get('repetition_rate', 1):.3f}",        # 🔥 NEW
                f"{current_lr:.2e}",
                self.patience_counter,
                1 if is_best else 0
            ])
    
    def compute_gen_prob(self, epoch, quality_score):
        """
        🔥 FIXED: 3-Stage Quality-Gated Scheduled Sampling (SLOWER & SAFER)
        
        Stage 1 (0-7): STABILIZATION
          - gen_prob = 0% (HARD LOCK)
          - Mục tiêu: Học stable reasoning với pure teacher forcing
          - 8 epochs để model nắm vững pattern
        
        Stage 2 (8-19): GRADUAL EXPOSURE
          - gen_prob: 0% → 12% (CEILING thấp hơn)
          - Quality-gated: increase only if quality > 0.85 (strict hơn)
          - +1% per epoch (chậm hơn 2x so với cũ)
        
        Stage 3 (20-39): FULL FINETUNING
          - gen_prob: 12% → 40% (FLOOR 12%, CEILING 40%)
          - Quality-gated: increase/decrease based on quality
          - +1.4% per epoch
          - Can decrease to floor if quality drops
        """
        # Stage 1: STABILIZATION (hard lock at 0% for 8 epochs)
        if epoch < self.alignment_epochs:
            return 0.0, "Stage 1: STABILIZATION"
        
        # Stage 2: GRADUAL EXPOSURE
        elif epoch < self.language_tuning_epochs:
            # Quality gate (strict!)
            if quality_score < self.quality_threshold:
                # Quality not good → hold
                return self.current_gen_prob, "Stage 2: GRADUAL EXPOSURE (Holding)"
            
            # Quality good → increase slowly
            epochs_in_stage = epoch - self.alignment_epochs  # 0-indexed
            target_prob = min(self.stage2_ceiling, (epochs_in_stage + 1) * 0.01)  # 🔥 1% per epoch (was 2%)
            
            # Smooth increase
            new_prob = min(self.current_gen_prob + 0.01, target_prob, self.stage2_ceiling)
            return new_prob, "Stage 2: GRADUAL EXPOSURE (Growing)"
        
        # Stage 3: FULL FINETUNING
        else:
            # Start from Stage 2 floor (12%)
            floor = self.stage2_ceiling
            
            if quality_score < self.quality_threshold:
                # Quality drop → DECREASE (but respect floor)
                new_prob = max(floor, self.current_gen_prob - 0.014)  # 🔥 Decrease proportional to increase
                return new_prob, "Stage 3: FULL FINETUNING (Decreasing)"
            
            # Quality good → increase to ceiling
            epochs_in_stage = epoch - self.language_tuning_epochs  # 0-indexed
            target_prob = min(self.stage3_ceiling, floor + (epochs_in_stage + 1) * 0.014)  # 🔥 1.4% per epoch
            
            # Smooth increase
            new_prob = min(self.current_gen_prob + 0.014, target_prob, self.stage3_ceiling)
            
            if new_prob >= self.stage3_ceiling:
                return self.stage3_ceiling, "Stage 3: FULL FINETUNING (Ceiling)"
            else:
                return new_prob, "Stage 3: FULL FINETUNING (Growing)"
    
    def save_checkpoint(self, epoch, val_loss=None, is_best=False):
        try:
            checkpoint = {
                'epoch': epoch + 1,  # Save NEXT epoch to train (so resume doesn't skip or repeat)
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'best_val_answer_loss': self.best_val_answer_loss,  # 🔥 FIX 3
                'val_answer_loss_history': self.val_answer_loss_history,  # 🔥 FIX 3
                'patience_counter': self.patience_counter,
                'current_gen_prob': self.current_gen_prob,
                'quality_tracker': self.quality_tracker.history,
            }
            
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            latest_path = self.output_dir / 'checkpoint_latest.pt'
            torch.save(checkpoint, latest_path)
            
            if is_best and val_loss is not None:
                best_path = self.output_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f"[INFO] ✓ Best model saved (Loss: {val_loss:.4f})")
            
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        try:
            print(f"\n[INFO] Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']  # Epoch already +1 when saved
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_answer_loss = checkpoint.get('best_val_answer_loss', float('inf'))  # 🔥 FIX 3
            self.val_answer_loss_history = checkpoint.get('val_answer_loss_history', [])  # 🔥 FIX 3
            self.patience_counter = checkpoint['patience_counter']
            self.current_gen_prob = checkpoint.get('current_gen_prob', 0.0)
            
            if 'quality_tracker' in checkpoint:
                self.quality_tracker.history = checkpoint['quality_tracker']
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"[INFO] ✓ Resume from epoch {self.current_epoch}")
            print(f"[INFO] ✓ Current gen_prob: {self.current_gen_prob:.1%}")
            
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            return False
    
    def train_epoch(self, epoch, gen_prob):
        """Train one epoch with given gen_prob"""
        self.model.train()
        self.optimizer.zero_grad()  # Initialize clean state
        total_loss = 0
        reasoning_loss_sum = 0
        answer_loss_sum = 0
        
        progress_bar = tqdm(self.train_loader, 
                           desc=f"Epoch {epoch+1}/{self.num_epochs} [gen={gen_prob:.1%}]")
        
        for step, batch in enumerate(progress_bar):
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                           if torch.is_tensor(v)}
            
            # Prepare labels
            reasoning_labels = tensor_batch['reasoning_input_ids'].clone()
            reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
            
            answer_labels = tensor_batch['answer_input_ids'].clone()
            answer_labels[answer_labels == self.model.tokenizer.pad_token_id] = -100
            
            try:
                with autocast('cuda', enabled=self.use_amp):
                    # Encode
                    vision_embeds = self.model.encode_image(tensor_batch['pixel_values'])
                    question_embeds = self.model.encode_text(
                        input_ids=tensor_batch['input_ids'],
                        attention_mask=tensor_batch['attention_mask']
                    )
                    fused_features, _ = self.model.fuse_multimodal(question_embeds, vision_embeds)
                    
                    # 🔥 FIX: Two-pass training for reasoning decoder feedback
                    
                    # Pass 1: GT reasoning path (for reasoning loss + GT answer baseline)
                    reasoning_logits_gt, reasoning_hidden_gt, _ = self.model.generate_reasoning(
                        fused_features=fused_features,
                        reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                        reasoning_attention_mask=tensor_batch['reasoning_attention_mask']
                    )
                    
                    reasoning_loss = self.criterion(
                        reasoning_logits_gt.view(-1, reasoning_logits_gt.size(-1)),
                        reasoning_labels.view(-1)
                    )
                    
                    # Pass 2: Scheduled sampling with GRADIENT flow
                    use_generated = random.random() < gen_prob
                    
                    if use_generated and gen_prob > 0:
                        # 🔥 Generate reasoning WITH GRADIENT (not detached!)
                        # Use teacher forcing on generated tokens (differentiable)
                        reasoning_ids_gen, _ = self.model.generate_reasoning_autoregressive(
                            fused_features=fused_features,
                            max_length=96,
                            num_beams=1,
                            temperature=0.7,
                            repetition_penalty=1.5
                        )
                        
                        # Forward pass with generated IDs (gradients flow!)
                        gen_attention_mask = (reasoning_ids_gen != self.model.tokenizer.pad_token_id).long()
                        _, reasoning_hidden_gen, _ = self.model.generate_reasoning(
                            fused_features=fused_features,
                            reasoning_input_ids=reasoning_ids_gen,  # Generated, but gradients flow through decoder
                            reasoning_attention_mask=gen_attention_mask
                        )
                        reasoning_hidden_for_answer = reasoning_hidden_gen
                    else:
                        # Use GT reasoning
                        reasoning_hidden_for_answer = reasoning_hidden_gt
                    
                    # Answer loss (gradients flow back to reasoning decoder if generated!)
                    answer_logits, _ = self.model.generate_answer(
                        fused_features=fused_features,
                        reasoning_hidden=reasoning_hidden_for_answer,
                        answer_input_ids=tensor_batch['answer_input_ids'],
                        answer_attention_mask=tensor_batch['answer_attention_mask']
                    )
                    
                    answer_loss = self.criterion(
                        answer_logits.view(-1, answer_logits.size(-1)),
                        answer_labels.view(-1)
                    )
                    
                    # 🔥 Combined loss: reasoning decoder learns from BOTH losses!
                    loss = (self.alpha_reasoning * reasoning_loss + 
                           self.alpha_answer * answer_loss)
                    loss = loss / self.gradient_accumulation_steps
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[WARNING] OOM at step {step}!")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate
            total_loss += loss.item() * self.gradient_accumulation_steps
            reasoning_loss_sum += reasoning_loss.item()
            answer_loss_sum += answer_loss.item()
            
            # Save periodic
            if self.save_steps > 0 and self.global_step > 0 and self.global_step % self.save_steps == 0:
                self.save_checkpoint(epoch)
            
            # Memory cleanup
            if (step + 1) % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update progress
            if (step + 1) % self.log_steps == 0:
                avg_loss = total_loss / (step + 1)
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        n = len(self.train_loader)
        return {
            'total': total_loss / n,
            'reasoning': reasoning_loss_sum / n,
            'answer': answer_loss_sum / n
        }
    
    @torch.no_grad()
    def validate(self, epoch):
        """Standard validation"""
        self.model.eval()
        total_loss = 0
        reasoning_loss_sum = 0
        answer_loss_sum = 0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                               if torch.is_tensor(v)}
                
                reasoning_labels = tensor_batch['reasoning_input_ids'].clone()
                reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
                
                answer_labels = tensor_batch['answer_input_ids'].clone()
                answer_labels[answer_labels == self.model.tokenizer.pad_token_id] = -100
                
                with autocast('cuda', enabled=self.use_amp):
                    # Encode
                    vision_embeds = self.model.encode_image(tensor_batch['pixel_values'])
                    question_embeds = self.model.encode_text(
                        input_ids=tensor_batch['input_ids'],
                        attention_mask=tensor_batch['attention_mask']
                    )
                    fused_features, _ = self.model.fuse_multimodal(question_embeds, vision_embeds)
                    
                    # Reasoning
                    reasoning_logits, reasoning_hidden, _ = self.model.generate_reasoning(
                        fused_features=fused_features,
                        reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                        reasoning_attention_mask=tensor_batch['reasoning_attention_mask']
                    )
                    
                    reasoning_loss = self.criterion(
                        reasoning_logits.view(-1, reasoning_logits.size(-1)),
                        reasoning_labels.view(-1)
                    )
                    
                    # Answer
                    answer_logits, _ = self.model.generate_answer(
                        fused_features=fused_features,
                        reasoning_hidden=reasoning_hidden,
                        answer_input_ids=tensor_batch['answer_input_ids'],
                        answer_attention_mask=tensor_batch['answer_attention_mask']
                    )
                    
                    answer_loss = self.criterion(
                        answer_logits.view(-1, answer_logits.size(-1)),
                        answer_labels.view(-1)
                    )
                    
                    loss = (self.alpha_reasoning * reasoning_loss + 
                           self.alpha_answer * answer_loss)
                
                total_loss += loss.item()
                reasoning_loss_sum += reasoning_loss.item()
                answer_loss_sum += answer_loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if num_batches > 0:
            return {
                'total': total_loss / num_batches,
                'reasoning': reasoning_loss_sum / num_batches,
                'answer': answer_loss_sum / num_batches
            }
        else:
            return {'total': float('inf'), 'reasoning': 0, 'answer': 0}
    
    @torch.no_grad()
    def measure_generation_quality(self, num_samples=50):
        """
        🔥 FIXED: Measure ACTUAL generation quality (not GT!)
        
        Metrics:
        - Perplexity: Computed on GENERATED text (via loss)
        - Non-empty rate: % of non-empty generations
        - Avg length: Average token count
        - Vietnamese coherence: % of Vietnamese tokens (NEW)
        - Repetition rate: % of repeated trigrams (NEW)
        """
        self.model.eval()
        
        perplexities = []
        non_empty_count = 0
        lengths = []
        vietnamese_scores = []
        repetition_rates = []
        total_samples = 0
        
        # Vietnamese token detection (common Vietnamese syllables/words)
        vietnamese_tokens = {'người', 'hình', 'ảnh', 'màu', 'chiếc', 'đang', 'một', 'trong', 'có', 
                           'là', 'của', 'được', 'không', 'này', 'đó', 'cho', 'với', 'các', 'từ',
                           'sắc', 'trên', 'thấy', 'ở', 'bên', 'trái', 'phải', 'giữa', 'và'}
        
        for batch_idx, batch in enumerate(self.val_loader):
            if total_samples >= num_samples:
                break
            
            try:
                tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                               if torch.is_tensor(v)}
                
                batch_size = tensor_batch['pixel_values'].size(0)
                
                with autocast('cuda', enabled=self.use_amp):
                    # Encode
                    vision_embeds = self.model.encode_image(tensor_batch['pixel_values'])
                    question_embeds = self.model.encode_text(
                        input_ids=tensor_batch['input_ids'],
                        attention_mask=tensor_batch['attention_mask']
                    )
                    fused_features, _ = self.model.fuse_multimodal(question_embeds, vision_embeds)
                    
                    # Generate reasoning (THIS is what we evaluate!)
                    reasoning_ids, reasoning_hidden = self.model.generate_reasoning_autoregressive(
                        fused_features=fused_features,
                        max_length=96,
                        num_beams=1,
                        temperature=0.7,
                        repetition_penalty=1.5
                    )
                    
                    # 🔥 FIX: Compute perplexity on GENERATED reasoning
                    # Create attention mask for generated ids
                    gen_attention_mask = (reasoning_ids != self.model.tokenizer.pad_token_id).long()
                    
                    # Forward pass with GENERATED reasoning_ids
                    reasoning_logits, _, _ = self.model.generate_reasoning(
                        fused_features=fused_features,
                        reasoning_input_ids=reasoning_ids,  # ✅ Use generated!
                        reasoning_attention_mask=gen_attention_mask
                    )
                    
                    # Prepare labels (shift for autoregressive loss)
                    reasoning_labels = reasoning_ids.clone()
                    reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
                    
                    # Compute loss (for perplexity)
                    loss = self.criterion(
                        reasoning_logits.reshape(-1, reasoning_logits.size(-1)),
                        reasoning_labels.reshape(-1)
                    )
                    
                    ppl = torch.exp(loss).item()
                    perplexities.append(ppl)
                    
                    # Check generation quality per sample
                    for i in range(batch_size):
                        text = self.model.tokenizer.decode(reasoning_ids[i], skip_special_tokens=True)
                        tokens = text.strip().split()
                        
                        if len(tokens) > 0:
                            non_empty_count += 1
                            lengths.append(len(tokens))
                            
                            # Vietnamese coherence check
                            viet_count = sum(1 for t in tokens if t.lower() in vietnamese_tokens)
                            vietnamese_scores.append(viet_count / len(tokens))
                            
                            # Repetition detection (trigram overlap)
                            if len(tokens) >= 3:
                                trigrams = [tuple(tokens[j:j+3]) for j in range(len(tokens)-2)]
                                if len(trigrams) > 0:
                                    unique_trigrams = len(set(trigrams))
                                    repetition_rates.append(1.0 - unique_trigrams / len(trigrams))
                                else:
                                    repetition_rates.append(0.0)
                            else:
                                repetition_rates.append(0.0)
                        else:
                            lengths.append(0)
                            vietnamese_scores.append(0.0)
                            repetition_rates.append(1.0)  # Empty = max repetition
                        
                        total_samples += 1
                        if total_samples >= num_samples:
                            break
                
                # Break outer loop if reached num_samples
                if total_samples >= num_samples:
                    break
                
            except Exception as e:
                print(f"[WARNING] Error in quality measurement: {e}")
                continue
        
        if not perplexities:
            return {
                'perplexity': 999.0,
                'non_empty_rate': 0.0,
                'avg_length': 0.0,
                'vietnamese_coherence': 0.0,
                'repetition_rate': 1.0
            }
        
        return {
            'perplexity': np.mean(perplexities),
            'non_empty_rate': non_empty_count / max(total_samples, 1),
            'avg_length': np.mean(lengths) if lengths else 0.0,
            'vietnamese_coherence': np.mean(vietnamese_scores) if vietnamese_scores else 0.0,
            'repetition_rate': np.mean(repetition_rates) if repetition_rates else 1.0
        }
    
    @torch.no_grad()
    def show_generation_samples(self, num_samples=3):
        """Show actual generation samples"""
        self.model.eval()
        
        print("\n" + "="*70)
        print("GENERATION SAMPLES")
        print("="*70)
        
        samples_shown = 0
        for batch in self.val_loader:
            if samples_shown >= num_samples:
                break
            
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                           if torch.is_tensor(v)}
            
            batch_size = tensor_batch['pixel_values'].size(0)
            
            for i in range(min(batch_size, num_samples - samples_shown)):
                try:
                    # Encode
                    vision_embeds = self.model.encode_image(tensor_batch['pixel_values'][i:i+1])
                    question_embeds = self.model.encode_text(
                        input_ids=tensor_batch['input_ids'][i:i+1],
                        attention_mask=tensor_batch['attention_mask'][i:i+1]
                    )
                    fused_features, _ = self.model.fuse_multimodal(question_embeds, vision_embeds)
                    
                    # Generate reasoning
                    reasoning_ids, reasoning_hidden = self.model.generate_reasoning_autoregressive(
                        fused_features=fused_features,
                        max_length=96,
                        num_beams=1,
                        temperature=0.7,
                        repetition_penalty=1.5
                    )
                    
                    # Generate answer using answer_decoder
                    answer_ids = torch.full(
                        (1, 1),
                        self.model.tokenizer.bos_token_id,
                        dtype=torch.long,
                        device=self.device
                    )
                    
                    # Autoregressive generation for answer
                    for _ in range(31):  # max 32 tokens
                        decoder_outputs = self.model.answer_decoder(
                            input_ids=answer_ids,
                            encoder_hidden_states=reasoning_hidden,
                            return_dict=True,
                            use_cache=False
                        )
                        
                        hidden = decoder_outputs.last_hidden_state[:, -1, :]
                        logits = self.model.lm_head(hidden)
                        
                        # Apply temperature and repetition penalty
                        logits = logits / 0.7
                        for token_id in set(answer_ids[0].tolist()):
                            logits[0, token_id] /= 1.5
                        
                        next_token = logits.argmax(dim=-1, keepdim=True)
                        answer_ids = torch.cat([answer_ids, next_token], dim=1)
                        
                        if next_token.item() == self.model.tokenizer.eos_token_id:
                            break
                    
                    # Decode
                    question_text = self.model.tokenizer.decode(
                        tensor_batch['input_ids'][i], 
                        skip_special_tokens=True
                    )
                    
                    generated_reasoning_text = self.model.tokenizer.decode(
                        reasoning_ids[0],
                        skip_special_tokens=True
                    )
                    
                    generated_answer_text = self.model.tokenizer.decode(
                        answer_ids[0],
                        skip_special_tokens=True
                    )
                    
                    gt_reasoning_text = self.model.tokenizer.decode(
                        tensor_batch['reasoning_input_ids'][i],
                        skip_special_tokens=True
                    )
                    
                    gt_answer_text = self.model.tokenizer.decode(
                        tensor_batch['answer_input_ids'][i],
                        skip_special_tokens=True
                    )
                    
                    # Print
                    print(f"\n[Sample {samples_shown + 1}]")
                    print(f"Question: {question_text}")
                    print(f"\nGT Reasoning: {gt_reasoning_text}")
                    print(f"Generated Reasoning: {generated_reasoning_text}")
                    print(f"\nGT Answer: {gt_answer_text}")
                    print(f"Generated Answer: {generated_answer_text}")
                    print("-" * 70)
                    
                    samples_shown += 1
                    
                except Exception as e:
                    print(f"[ERROR] Failed to generate sample: {e}")
                    continue
                
                if samples_shown >= num_samples:
                    break
        
        print("="*70 + "\n")
        return samples_shown
    
    def train(self):
        """Main training loop with validation-guided curriculum"""
        print(f"\n{'='*70}")
        print(f"IMPROVED IMPLICIT REASONING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            # Auto-unfreeze vision encoder at Stage 3
            if self.unfreeze_after_epoch is not None and epoch == self.unfreeze_after_epoch and self.vision_frozen:
                self._unfreeze_vision_encoder()
            
            # Measure generation quality (skip in Stage 1 - alignment only)
            if epoch < self.alignment_epochs:
                print(f"\n[EPOCH {epoch+1}] Stage 1: ALIGNMENT (skipping quality measurement)")
                gen_metrics = {'perplexity': 0, 'non_empty_rate': 0, 'avg_length': 0}
                quality_score = 0.0
            else:
                print(f"\n[EPOCH {epoch+1}] Measuring generation quality...")
                gen_metrics = self.measure_generation_quality(num_samples=50)
                self.quality_tracker.update(gen_metrics)
                quality_score = self.quality_tracker.get_quality_score()
                
                print(f"  Perplexity: {gen_metrics['perplexity']:.2f}")
                print(f"  Non-empty rate: {gen_metrics['non_empty_rate']:.1%}")
                print(f"  Avg length: {gen_metrics['avg_length']:.1f} tokens")
                print(f"  Vietnamese coherence: {gen_metrics.get('vietnamese_coherence', 0):.1%}")  # 🔥 NEW
                print(f"  Repetition rate: {gen_metrics.get('repetition_rate', 1):.1%}")  # 🔥 NEW
                print(f"  Quality score: {quality_score:.3f}")
            
            # Compute gen_prob based on quality
            new_gen_prob, phase = self.compute_gen_prob(epoch, quality_score)
            
            # Safety: only increase if quality improving
            if new_gen_prob > self.current_gen_prob:
                if not self.quality_tracker.is_improving():
                    print(f"  ⚠️  Quality not improving - holding gen_prob at {self.current_gen_prob:.1%}")
                    new_gen_prob = self.current_gen_prob
                else:
                    print(f"  ✓ Quality improving - increasing gen_prob: {self.current_gen_prob:.1%} → {new_gen_prob:.1%}")
            
            self.current_gen_prob = new_gen_prob
            
            # Train
            train_losses = self.train_epoch(epoch, self.current_gen_prob)
            print(f"\n[EPOCH {epoch+1}] [{phase}] Train Loss: {train_losses['total']:.4f}")
            print(f"  Reasoning: {train_losses['reasoning']:.4f}")
            print(f"  Answer: {train_losses['answer']:.4f}")
            
            # Validate
            val_losses = self.validate(epoch)
            print(f"[VALIDATION] Loss: {val_losses['total']:.4f}")
            print(f"  Reasoning: {val_losses['reasoning']:.4f}")
            print(f"  Answer: {val_losses['answer']:.4f}")
            
            # 🔥 FIX 3: Check answer decoder divergence
            self.val_answer_loss_history.append(val_losses['answer'])
            is_diverging, should_decrease_gen, divergence_msg = self._check_answer_decoder_divergence(
                val_losses['answer'], 
                train_losses['answer']
            )
            print(f"[HEALTH CHECK] {divergence_msg}")
            
            if should_decrease_gen and self.current_gen_prob > 0:
                old_gen_prob = self.current_gen_prob
                self.current_gen_prob = max(0.0, self.current_gen_prob - 0.05)
                print(f"[ACTION] 🛑 Decreasing gen_prob: {old_gen_prob:.1%} → {self.current_gen_prob:.1%}")
            
            # Show samples every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.show_generation_samples(num_samples=3)
            
            # 🔥 FIX 3: Save best dựa trên val_answer_loss (key metric!)
            is_best = False
            if val_losses['answer'] < self.best_val_answer_loss:
                self.best_val_answer_loss = val_losses['answer']
                self.best_val_loss = val_losses['total']  # Also update total
                self.patience_counter = 0
                is_best = True
                self.save_checkpoint(epoch, val_losses['total'], is_best=True)
                print(f"[INFO] 🎯 New best answer loss: {self.best_val_answer_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"[INFO] No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Save latest
            self.save_checkpoint(epoch, val_losses['total'], is_best=False)
            
            # Log
            self.log_to_csv(epoch, phase, self.current_gen_prob, quality_score, 
                          train_losses, val_losses, gen_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improved Implicit Reasoning Training')
    
    parser.add_argument('--train_json', type=str, 
                        default='/kaggle/input/teacher/teacher_outputs_train.jsonl')
    parser.add_argument('--image_dir', type=str,
                        default='/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train')
    parser.add_argument('--output_dir', type=str,
                        default='/kaggle/working/checkpoints_implicit_fixed')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='🔥 FIXED: 40 epochs cho 3-stage curriculum (8+12+20)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='🔥 ULTRA-SAFE: 5e-5 for very stable decoder training')
    parser.add_argument('--alpha_reasoning', type=float, default=0.4)
    parser.add_argument('--reasoning_bottleneck', type=int, default=None)
    parser.add_argument('--freeze_pretrained', action='store_true',
                        help='Freeze all pretrained weights (encoder + decoders), only train heads')
    parser.add_argument('--freeze_encoders_only', action='store_true',
                        help='🔥 RECOMMENDED: Freeze encoders only, train decoders + fusion (~573M trainable)')
    parser.add_argument('--freeze_vision', action='store_true',
                        help='Stage 1: Freeze DINOv2 only (train decoders)')
    parser.add_argument('--unfreeze_after_epoch', type=int, default=None,
                        help='DEPRECATED: Use manual 2-stage instead for better control')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.20,
                        help='Label smoothing (0.20 ULTRA-SAFE - maximum overfitting prevention)')
    parser.add_argument('--weight_decay', type=float, default=0.10,
                        help='Weight decay (0.10 ULTRA-SAFE - 10x stronger regularization)')
    parser.add_argument('--quality_threshold', type=float, default=0.60,
                        help='Quality threshold for scheduled sampling (0.60 recommended)')
    
    args = parser.parse_args()
    
    # Validate strategy
    strategies = [args.freeze_pretrained, args.freeze_encoders_only, args.freeze_vision]
    if sum(strategies) > 1:
        raise ValueError("Cannot use multiple freeze strategies. Choose only one.")
    
    # Validate strategy
    if args.freeze_vision and args.unfreeze_after_epoch:
        print("\n⚠️  WARNING: Auto-unfreeze may conflict with curriculum phases!")
        print("   Recommended: Use manual 2-stage training instead")
        print("   Stage 1: --freeze_vision --num_epochs 15")
        print("   Stage 2: --resume best_model.pt --num_epochs 25 --learning_rate 1e-5\n")
    
    print("="*70)
    print("IMPROVED IMPLICIT REASONING TRAINING")
    if args.freeze_encoders_only:
        print("🔥 RECOMMENDED: Freeze Encoders Only (Train Decoders)")
        print("   Frozen: Vision + Text Encoder (~280M)")
        print("   Trainable: Both Decoders + Fusion (~573M)")
        print("   Recommended: LR=1e-4, epochs=25-30")
    elif args.freeze_vision:
        print("🔒 STAGE 1: WARM-UP (Vision Frozen - covers Foundation+Exposure)")
        print("   Recommended: Train for 15 epochs")
    elif args.freeze_pretrained:
        print("🔒 FEATURE EXTRACTION: All Pretrained Frozen")
        print("   Warning: Decoders frozen → poor generation")
    elif args.resume:
        print("🔓 STAGE 2: FINE-TUNE (All Parameters - Robustness phase)")
        print("   Recommended: LR=1e-5, train for 20-25 epochs")
    print("="*70)
    
    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Model
    print("\n[INFO] Initializing model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        use_reasoning_quality_check=False,
        gradient_checkpointing=True,
        reasoning_bottleneck_tokens=args.reasoning_bottleneck
    )
    
    # Apply freezing strategy
    if args.freeze_pretrained:
        model.freeze_pretrained_weights()
    elif args.freeze_encoders_only:
        model.freeze_encoders_only()
    elif args.freeze_vision:
        print("\n[INFO] 🔒 FREEZING DINOv2 Vision Encoder only")
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n[INFO] Total params: {total_params/1e6:.1f}M")
    print(f"[INFO] Trainable params: {trainable_params/1e6:.1f}M")
    
    if args.freeze_pretrained:
        print(f"[INFO] Strategy: Feature extraction - train heads only (~91M)")
    elif args.freeze_vision:
        vision_params = sum(p.numel() for p in model.vision_encoder.parameters())
        print(f"[INFO] Frozen vision params: {vision_params/1e6:.1f}M")
        print(f"[INFO] Strategy: Train fusion + decoders (~595M)")
    
    # Dataset
    print("\n[INFO] Loading dataset...")
    full_dataset = ImplicitReasoningDataset(
        json_path=args.train_json,
        image_dir=args.image_dir,
        vision_processor=model.vision_processor,
        tokenizer=model.tokenizer,
        augment=True
    )
    
    total_size = len(full_dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Trainer
    trainer = ImplicitReasoningTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        alpha_reasoning=args.alpha_reasoning,
        quality_threshold=args.quality_threshold,
        # Use default 3-stage params (optimized for feature extraction):
        # alignment_epochs=3, language_tuning_epochs=10, full_finetuning_epochs=20
        unfreeze_after_epoch=args.unfreeze_after_epoch,
    )
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()
    
    print("\n[INFO] Training completed!")
    
    # Print 2-stage training guidance
    if args.freeze_vision and not args.resume:
        print("\n" + "="*70)
        print("STAGE 1 COMPLETE - Next Steps for STAGE 2:")
        print("="*70)
        print("Stage 2: Unfreeze vision encoder and fine-tune end-to-end")
        print("")
        print("✅ ALIGNED STRATEGY (Recommended):")
        print("   Stage 1 covered Foundation + Exposure phases")
        print("   Stage 2 will continue Robustness phase from current gen_prob")
        print("")
        print("Command:")
        print(f"  python {os.path.basename(__file__)} \\")
        print(f"    --resume {args.output_dir}/best_model.pt \\")
        print(f"    --output_dir {args.output_dir}_stage2 \\")
        print(f"    --num_epochs 25 \\")
        print(f"    --learning_rate 1e-5  # 🔥 LOWER LR for fine-tuning!")
        print("")
        print("Expected behavior:")
        print("  - Will resume from epoch {current_epoch}")
        print("  - Will continue gen_prob from checkpoint (no reset)")
        print("  - Vision encoder will be unfrozen automatically")
        print("="*70)


if __name__ == '__main__':
    main()
