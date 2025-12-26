"""
PROFESSIONAL VQA TRAINING PIPELINE
===================================
Đầy đủ techniques như chuyên gia:
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Learning Rate Scheduling (Cosine with Warmup)
- Gradient Clipping
- EMA (Exponential Moving Average)
- Multi-task Learning (Answer + Reasoning)
- Confidence-weighted Loss
- Early Stopping với patience
- Checkpoint Management
- Logging & Monitoring
- Data Augmentation
- Label Smoothing

Author: VQA Training Expert
Target: 70%+ accuracy on ViVQA
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from tqdm.auto import tqdm
import wandb
from pathlib import Path
import numpy as np
from collections import defaultdict
import copy
import random


# ============================================================================
# 1. DATASET WITH AUGMENTATION
# ============================================================================

class VQADistillationDataset(Dataset):
    """Dataset với offline distillation data + augmentation"""
    
    def __init__(
        self, 
        json_path, 
        image_dir,
        clip_processor,
        text_tokenizer,
        decoder_tokenizer,
        max_question_len=64,
        max_answer_len=32,
        max_reasoning_len=128,
        augment=False
    ):
        self.data = self.load_data(json_path)
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
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
        """Simple image augmentation"""
        import torchvision.transforms as T
        
        aug = T.Compose([
            T.RandomHorizontalFlip(p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])
        return aug(image)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, item['image_path'].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        
        # Augment if training
        if self.augment:
            image = self.augment_image(image)
        
        # Process image
        pixel_values = self.clip_processor(images=image, return_tensors="pt")['pixel_values'][0]
        
        # Tokenize question
        question_encoding = self.text_tokenizer(
            item['question'],
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize teacher answer
        teacher_answer_encoding = self.decoder_tokenizer(
            item['teacher_answer'],
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize teacher reasoning (if available)
        if 'teacher_reasoning' in item and item['teacher_reasoning']:
            reasoning_encoding = self.decoder_tokenizer(
                item['teacher_reasoning'],
                max_length=self.max_reasoning_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            # Dummy reasoning if not available
            reasoning_encoding = teacher_answer_encoding
        
        return {
            'pixel_values': pixel_values,
            'input_ids': question_encoding['input_ids'][0],
            'attention_mask': question_encoding['attention_mask'][0],
            'labels': teacher_answer_encoding['input_ids'][0],
            'reasoning_labels': reasoning_encoding['input_ids'][0],
            'reasoning_weight': item.get('reasoning_weight', 3.0),
            'reasoning_type': item.get('reasoning_type', 'OTHER'),
            'img_id': item['img_id']
        }


# ============================================================================
# 2. ADVANCED LOSS FUNCTIONS
# ============================================================================

class ChainOfThoughtLoss(nn.Module):
    """
    Multi-task loss with Chain-of-Thought reasoning
    Model phải suy nghĩ (reasoning) trước khi trả lời (answer)
    Giống như con người: "Tôi thấy cái bình màu xanh lá → Answer: màu xanh lá"
    """
    
    def __init__(
        self,
        alpha_reasoning=0.6,  # Reasoning loss weight (higher priority)
        alpha_answer=0.4,     # Answer loss weight
        temperature=2.0,
        label_smoothing=0.1,
        max_weight=3.0  # Maximum reasoning_weight in data
    ):
        super().__init__()
        self.alpha_reasoning = alpha_reasoning
        self.alpha_answer = alpha_answer
        self.temperature = temperature
        self.max_weight = max_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
    
    def forward(self, outputs, answer_labels, reasoning_labels, reasoning_weight=1.0):
        """
        outputs: dict with 'reasoning_logits' and 'answer_logits'
        answer_labels: teacher answer ids - [batch_size, seq_len]
        reasoning_labels: teacher reasoning ids - [batch_size, seq_len]
        reasoning_weight: confidence score from teacher
        
        NEW: Model outputs [batch_size, seq_len, vocab_size] from ViT5 decoder
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. REASONING LOSS (priority - model must learn to think first)
        if 'reasoning_logits' in outputs and reasoning_labels is not None and outputs['reasoning_logits'] is not None:
            reasoning_logits = outputs['reasoning_logits']  # [B, L_r, vocab_size]
            
            # Shift for teacher forcing: predict next token
            # Flatten for CrossEntropyLoss
            shift_logits = reasoning_logits[:, :-1, :].contiguous()  # [B, L-1, V]
            shift_labels = reasoning_labels[:, 1:].contiguous()  # [B, L-1]
            
            # Reshape for loss computation
            reasoning_loss = self.ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)),  # [B*(L-1), V]
                shift_labels.view(-1)  # [B*(L-1)]
            )
            total_loss += self.alpha_reasoning * reasoning_loss
            loss_dict['reasoning_loss'] = reasoning_loss.item()
        else:
            loss_dict['reasoning_loss'] = 0.0
        
        # 2. ANSWER LOSS (based on reasoning context)
        if 'answer_logits' in outputs and outputs['answer_logits'] is not None:
            answer_logits = outputs['answer_logits']  # [B, L_a, vocab_size]
            
            # Shift for teacher forcing
            shift_logits = answer_logits[:, :-1, :].contiguous()  # [B, L-1, V]
            shift_labels = answer_labels[:, 1:].contiguous()  # [B, L-1]
            
            # Reshape for loss computation
            answer_loss = self.ce_loss(
                shift_logits.view(-1, shift_logits.size(-1)),  # [B*(L-1), V]
                shift_labels.view(-1)  # [B*(L-1)]
            )
            total_loss += self.alpha_answer * answer_loss
            loss_dict['answer_loss'] = answer_loss.item()
        
        # 3. Weight by teacher confidence
        confidence_scale = reasoning_weight / self.max_weight
        weighted_loss = total_loss * confidence_scale
        
        loss_dict['confidence_scale'] = confidence_scale
        loss_dict['total_loss'] = weighted_loss.item()
        loss_dict['unweighted_total'] = total_loss.item()
        
        return weighted_loss, loss_dict


# ============================================================================
# 3. EMA (EXPONENTIAL MOVING AVERAGE)
# ============================================================================

class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# 4. TRAINER CLASS (CHUYÊN NGHIỆP)
# ============================================================================

class VQATrainer:
    """Professional trainer with all best practices"""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        output_dir='./checkpoints',
        # Training hyperparameters
        batch_size=16,
        gradient_accumulation_steps=4,
        num_epochs=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        # Loss configuration
        alpha_reasoning=0.6,
        alpha_answer=0.4,
        label_smoothing=0.1,
        max_reasoning_weight=3.0,
        # Advanced features
        use_amp=True,
        use_ema=True,
        ema_decay=0.999,
        # Early stopping
        patience=5,
        # Logging
        log_steps=50,
        eval_steps=500,
        save_steps=1000,
        use_wandb=False,
        wandb_project='vqa-distillation',
        # Device
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        
        # Device
        self.device = device
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Reduced to save memory
            pin_memory=True  # Disable to save memory
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,  # Same as train to avoid OOM
            shuffle=False,
            num_workers=2,  # Reduced to save memory
            pin_memory=False  # Disable to save memory
        )
        
        # Loss function
        self.criterion = ChainOfThoughtLoss(
            alpha_reasoning=alpha_reasoning,
            alpha_answer=alpha_answer,
            label_smoothing=label_smoothing,
            max_weight=max_reasoning_weight
        )
        
        # Optimizer (AdamW with weight decay + Nesterov momentum)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Note: AdamW already uses momentum-like behavior via beta1
        # For explicit Nesterov, could use SGD with nesterov=True
        # But AdamW is SOTA for transformers
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision training
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # Early stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = None
        
        # Logging
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project=wandb_project, config={
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'model_params': sum(p.numel() for p in model.parameters()),
            })
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        print(f"[INFO] Trainer initialized")
        print(f"  Device: {device}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Total steps: {num_training_steps}")
        print(f"  Warmup steps: {num_warmup_steps}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  EMA: {use_ema}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_losses = defaultdict(float)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    reasoning_labels=batch.get('reasoning_labels'),  # CRITICAL: Pass reasoning labels
                    labels=batch['labels']
                )
                
                # Prepare outputs dict for loss calculation
                # Support both single-task (answer only) and multi-task (reasoning + answer)
                outputs_dict = {}
                
                if hasattr(outputs, 'reasoning_logits'):
                    # Multi-task model with reasoning head
                    outputs_dict['reasoning_logits'] = outputs.reasoning_logits
                    outputs_dict['answer_logits'] = outputs.answer_logits
                else:
                    # Single-task model (fallback)
                    outputs_dict['answer_logits'] = outputs.logits
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    outputs=outputs_dict,
                    answer_labels=batch['labels'],
                    reasoning_labels=batch.get('reasoning_labels'),
                    reasoning_weight=batch['reasoning_weight'].mean().item()
                )
                
                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                self.scheduler.step()
                
                # EMA update
                if self.use_ema:
                    self.ema.update()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Clear CUDA cache periodically to avoid fragmentation
                if self.global_step % 100 == 0:
                    torch.cuda.empty_cache()
                
                self.global_step += 1
            
            # Accumulate losses
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Logging
            if self.global_step % self.log_steps == 0:
                log_dict = {
                    'train/loss': loss.item() * self.gradient_accumulation_steps,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                }
                
                if self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)
            
            # Evaluation
            if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                val_loss, val_metrics = self.evaluate()
                self.model.train()  # Back to training mode
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(f'best_model_step_{self.global_step}.pt', is_best=True)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    print(f"[INFO] Early stopping triggered at step {self.global_step}")
                    return True  # Signal to stop training
            
            # Save checkpoint
            if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        # Epoch metrics
        avg_loss = epoch_loss / len(self.train_loader)
        avg_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}
        
        print(f"\n[EPOCH {epoch+1}] Train Loss: {avg_loss:.4f}")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")
        
        return False  # Continue training
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        
        # Apply EMA weights if available
        if self.use_ema:
            self.ema.apply_shadow()
        
        total_loss = 0.0
        all_losses = defaultdict(float)
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    reasoning_labels=batch.get('reasoning_labels'),  # Pass reasoning labels for loss
                    labels=batch['labels']
                )
                
                # Prepare outputs dict
                outputs_dict = {}
                if hasattr(outputs, 'reasoning_logits'):
                    outputs_dict['reasoning_logits'] = outputs.reasoning_logits
                    outputs_dict['answer_logits'] = outputs.answer_logits
                else:
                    outputs_dict['answer_logits'] = outputs.logits
                
                loss, loss_dict = self.criterion(
                    outputs=outputs_dict,
                    answer_labels=batch['labels'],
                    reasoning_labels=batch.get('reasoning_labels'),
                    reasoning_weight=batch['reasoning_weight'].mean().item()
                )
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                all_losses[k] += v
        
        # Restore original weights
        if self.use_ema:
            self.ema.restore()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_losses = {k: v / len(self.val_loader) for k, v in all_losses.items()}
        
        print(f"\n[VALIDATION] Loss: {avg_loss:.4f}")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")
        
        if self.use_wandb:
            log_dict = {'val/loss': avg_loss}
            log_dict.update({f'val/{k}': v for k, v in avg_losses.items()})
            wandb.log(log_dict, step=self.global_step)
        
        return avg_loss, avg_losses
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint - ONLY SAVE BEST to save disk space"""
        if not is_best:
            # Skip non-best checkpoints to save disk space (each ~2.5GB)
            return
        
        checkpoint_path = self.output_dir / 'best_model.pt'  # Always overwrite
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
        }
        
        if self.use_ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # Save with atomic write to prevent corruption
        temp_path = self.output_dir / 'best_model_temp.pt'
        torch.save(checkpoint, temp_path)
        
        # Atomic rename
        import shutil
        shutil.move(str(temp_path), str(checkpoint_path))
        
        self.best_model_path = checkpoint_path
        print(f"[INFO] ✓ Best model saved: {checkpoint_path} (Epoch {self.current_epoch+1}, Loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resume training"""
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        
        if self.use_ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f"[INFO] Checkpoint loaded successfully!")
        print(f"  Resuming from Epoch: {self.current_epoch + 1}")
        print(f"  Global Step: {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Patience counter: {self.patience_counter}/{self.patience}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            should_stop = self.train_epoch(epoch)
            
            # Evaluate at end of epoch
            val_loss, val_metrics = self.evaluate()
            
            # Check early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f'best_model_epoch_{epoch+1}.pt', is_best=True)
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break
            
            # Don't save every epoch checkpoint to save disk space
            # Only best model is saved
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model: {self.best_model_path}")
        
        if self.use_wandb:
            wandb.finish()


# ============================================================================
# 5. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""
    
    # Config
    CONFIG = {
        # Paths - Kaggle format
        'train_json': '/kaggle/input/teacher-5-12/teacher_outputs_train.jsonl',
        'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
        'output_dir': '/kaggle/working/checkpoints',
        
        # Train/Val split (nếu không có val_json riêng)
        'val_split': 0.1,  # 10% for validation
        'random_seed': 42,
        
        # Resume training
        'resume_checkpoint': None,  # Path to checkpoint để resume, hoặc None
        
        # Training - EXTREME MEMORY OPTIMIZATION (15GB GPU)
        'batch_size': 2,  # Further reduced to 2 to save memory
        'gradient_accumulation_steps': 32,  # Increased to keep effective batch = 64
        'num_epochs': 20,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        
        # Loss weights (Chain-of-Thought)
        'alpha_reasoning': 0.6,  # Reasoning first (higher priority)
        'alpha_answer': 0.4,     # Answer based on reasoning
        'label_smoothing': 0.1,
        
        # Advanced features
        'use_amp': True,
        'use_ema': False,  # DISABLED - EMA uses too much memory (duplicate weights)
        'ema_decay': 0.999,
        
        # Early stopping
        'patience': 5,
        
        # Logging - ADJUSTED FOR GRADIENT ACCUMULATION
        'log_steps': 10,     # Log mỗi 10 accumulation steps (was 50)
        'eval_steps': 0,     # DISABLE mid-epoch eval (too slow), only eval at epoch end
        'save_steps': 0,     # DISABLE mid-epoch save (too slow), only save at epoch end
        'use_wandb': False,
    }
    
    print("="*70)
    print("PROFESSIONAL VQA TRAINING (with Resume Support)")
    print("="*70)
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()
    
    # Set random seed for reproducibility
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['random_seed'])
    
    # Import model
    print("[INFO] Importing Chain-of-Thought model...")
    from model_cot import create_cot_model
    
    # Initialize model
    print("[INFO] Initializing model...")
    model = create_cot_model(
        clip_model='openai/clip-vit-base-patch32',
        text_encoder='vinai/phobert-base',
        decoder='VietAI/vit5-base',
        hidden_dim=768,
        fusion='cross_attention',  # SOTA: Bidirectional cross-attention (better than concat)
        use_reasoning_attention=True
    )
    
    # ===== STAGED UNFREEZING STRATEGY =====
    def freeze_encoder(encoder, name):
        """Freeze all parameters of an encoder"""
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"[INFO] ✓ Frozen {name}")
    
    def unfreeze_last_n_layers(encoder, name, n_layers=2):
        """Unfreeze last n layers of encoder"""
        # For transformers, unfreeze last n layers
        if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
            # BERT-like (PhoBERT)
            total_layers = len(encoder.encoder.layer)
            for i, layer in enumerate(encoder.encoder.layer):
                if i >= total_layers - n_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            # Also unfreeze pooler if exists
            if hasattr(encoder, 'pooler'):
                for param in encoder.pooler.parameters():
                    param.requires_grad = True
            print(f"[INFO] ✓ Unfrozen last {n_layers} layers of {name}")
        elif hasattr(encoder, 'vision_model'):
            # CLIP vision model
            if hasattr(encoder.vision_model, 'encoder') and hasattr(encoder.vision_model.encoder, 'layers'):
                total_layers = len(encoder.vision_model.encoder.layers)
                for i, layer in enumerate(encoder.vision_model.encoder.layers):
                    if i >= total_layers - n_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                # Unfreeze post_layernorm and projection
                if hasattr(encoder, 'visual_projection'):
                    for param in encoder.visual_projection.parameters():
                        param.requires_grad = True
                print(f"[INFO] ✓ Unfrozen last {n_layers} layers of {name}")
    
    # Stage 1: Freeze ALL encoders (only train fusion + heads)
    print("\n[STAGE 1] Freezing all encoders, training fusion + heads only...")
    freeze_encoder(model.clip_model, "CLIP")
    freeze_encoder(model.text_encoder, "PhoBERT")
    # Decoder frozen by default (we only use it as vocabulary reference)
    
    # Enable gradient checkpointing to save memory
    print("[INFO] Enabling gradient checkpointing to save memory...")
    if hasattr(model.clip_model, 'gradient_checkpointing_enable'):
        model.clip_model.gradient_checkpointing_enable()
    if hasattr(model.text_encoder, 'gradient_checkpointing_enable'):
        model.text_encoder.gradient_checkpointing_enable()
    if hasattr(model.decoder_backbone, 'gradient_checkpointing_enable'):
        model.decoder_backbone.gradient_checkpointing_enable()
    
    # Load full dataset
    print("[INFO] Loading full dataset...")
    full_dataset = VQADistillationDataset(
        json_path=CONFIG['train_json'],
        image_dir=CONFIG['image_dir'],
        clip_processor=model.clip_processor,
        text_tokenizer=model.text_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        augment=False  # Will enable for train split only
    )
    
    # Split into train and val
    print(f"[INFO] Splitting dataset with {CONFIG['val_split']*100:.0f}% for validation...")
    total_size = len(full_dataset)
    val_size = int(total_size * CONFIG['val_split'])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    # Enable augmentation for train split
    # Note: random_split returns Subset, so we modify the base dataset
    print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    
    # Create augmented train dataset
    train_dataset_aug = VQADistillationDataset(
        json_path=CONFIG['train_json'],
        image_dir=CONFIG['image_dir'],
        clip_processor=model.clip_processor,
        text_tokenizer=model.text_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        augment=True  # Enable augmentation
    )
    # Use same indices as train_dataset
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_aug, train_dataset.indices)
    
    # ===== STAGED TRAINING =====
    print("\n" + "="*70)
    print("STAGED UNFREEZING TRAINING STRATEGY")
    print("="*70)
    print("Stage 1: Train fusion + heads only (5 epochs)")
    print("Stage 2: Unfreeze PhoBERT last 2 layers (5 epochs)")
    print("Stage 3: Unfreeze CLIP last 2 layers (remaining epochs)")
    print("="*70 + "\n")
    
    # === STAGE 1: Fusion + Heads only ===
    print("\n[STAGE 1/3] Training fusion + heads (encoders frozen)...")
    trainer_stage1 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=5,  # Stage 1: 5 epochs
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        use_ema=CONFIG['use_ema'],
        ema_decay=CONFIG['ema_decay'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb']
    )
    trainer_stage1.train()
    
    # === STAGE 2: Unfreeze PhoBERT last layers ===
    print("\n[STAGE 2/3] Unfreezing PhoBERT last 2 layers...")
    unfreeze_last_n_layers(model.text_encoder, "PhoBERT", n_layers=2)
    
    trainer_stage2 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=5,  # Stage 2: 5 epochs
        learning_rate=CONFIG['learning_rate'] * 0.5,  # Lower LR for fine-tuning
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        use_ema=CONFIG['use_ema'],
        ema_decay=CONFIG['ema_decay'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb']
    )
    # Load best from stage 1
    stage1_best = os.path.join(CONFIG['output_dir'], 'best_model.pt')
    if os.path.exists(stage1_best):
        trainer_stage2.load_checkpoint(stage1_best)
    trainer_stage2.train()
    
    # === STAGE 3: Unfreeze CLIP last layers ===
    print("\n[STAGE 3/3] Unfreezing CLIP last 2 layers...")
    unfreeze_last_n_layers(model.clip_model, "CLIP", n_layers=2)
    
    remaining_epochs = CONFIG['num_epochs'] - 10  # Remaining after stage 1+2
    trainer_stage3 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=remaining_epochs,  # Stage 3: rest of epochs
        learning_rate=CONFIG['learning_rate'] * 0.3,  # Even lower LR
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        use_ema=CONFIG['use_ema'],
        ema_decay=CONFIG['ema_decay'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb']
    )
    # Load best from stage 2
    stage2_best = os.path.join(CONFIG['output_dir'], 'best_model.pt')
    if os.path.exists(stage2_best):
        trainer_stage3.load_checkpoint(stage2_best)
    trainer_stage3.train()
    
    print("\n[INFO] Training completed successfully!")


if __name__ == '__main__':
    main()