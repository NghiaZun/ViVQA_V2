"""
PROFESSIONAL TRAINING PIPELINE: DINOv2 + BARTpho VQA
======================================================
SOTA training với full features:
✅ Resume training từ bất kỳ checkpoint nào
✅ Optimizer state + scheduler state persistence
✅ Momentum scheduling (AdamW với cosine warmup)
✅ Staged training với reasoning quality validation
✅ Gradient checkpointing + Mixed Precision
✅ Chain-of-Thought với quality gating

Target: 70%+ accuracy trên ViVQA
"""

import os
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import copy
import random
from collections import defaultdict

# Import model mới
from model_dinov2_bartpho import DINOv2BARTphoVQA, count_parameters


# ============================================================================
# 1. DATASET
# ============================================================================

class VQADistillationDataset(Dataset):
    """Dataset cho DINOv2 + BARTpho"""
    
    def __init__(
        self, 
        json_path, 
        image_dir,
        vision_processor,  # DINOv2 AutoImageProcessor
        tokenizer,  # BARTpho tokenizer
        max_question_len=64,
        max_answer_len=32,
        max_reasoning_len=96,  # Reduced from 128 to make reasoning easier to learn
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
        """Image augmentation"""
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
        
        if self.augment:
            image = self.augment_image(image)
        
        # Process image với DINOv2 processor
        pixel_values = self.vision_processor(images=image, return_tensors='pt')['pixel_values'][0]
        
        # Tokenize question
        question_enc = self.tokenizer(
            item['question'],
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize reasoning (teacher output)
        reasoning_enc = self.tokenizer(
            item.get('reasoning', item.get('teacher_reasoning', '')),
            max_length=self.max_reasoning_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer (handle different field names)
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
            'labels': answer_enc['input_ids'][0],  # For evaluation
            'reasoning_labels': reasoning_enc['input_ids'][0],  # For evaluation
            'img_id': item.get('image_id', item.get('img_id', f"img_{idx}")),  # For evaluation
            'question': item['question'],  # For evaluation (already string)
        }


# ============================================================================
# 2. LOSS FUNCTIONS
# ============================================================================

class ChainOfThoughtLoss(nn.Module):
    """
    Multi-task loss với confidence weighting
    
    Loss = α * L_reasoning + β * L_answer + γ * L_quality
    
    L_quality: Encourage high confidence when correct, low when wrong
    """
    
    def __init__(
        self, 
        alpha_reasoning=0.6, 
        alpha_answer=0.4,
        alpha_quality=0.1,
        label_smoothing=0.1
    ):
        super().__init__()
        self.alpha_reasoning = alpha_reasoning
        self.alpha_answer = alpha_answer
        self.alpha_quality = alpha_quality
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100, 
            label_smoothing=label_smoothing
        )
        
    def forward(self, outputs, reasoning_labels, answer_labels):
        """
        Args:
            outputs: CoTOutput from model
            reasoning_labels: [batch, seq_len]
            answer_labels: [batch, seq_len]
        Returns:
            loss, loss_dict
        """
        # Reasoning loss
        reasoning_logits = outputs.reasoning_logits
        reasoning_loss = self.criterion(
            reasoning_logits.view(-1, reasoning_logits.size(-1)),
            reasoning_labels.view(-1)
        )
        
        # Answer loss
        answer_logits = outputs.answer_logits
        answer_loss = self.criterion(
            answer_logits.view(-1, answer_logits.size(-1)),
            answer_labels.view(-1)
        )
        
        # Quality loss: Calibrate confidence based on actual correctness
        quality_loss = 0.0
        if outputs.reasoning_confidence is not None:
            # Compute accuracy for each sample (simplified: check if argmax matches)
            reasoning_preds = reasoning_logits.argmax(dim=-1)
            reasoning_correct = (reasoning_preds == reasoning_labels).float().mean(dim=1)  # [batch]
            
            # Target confidence = actual correctness
            # If reasoning is correct, confidence should be high (1.0)
            # If reasoning is wrong, confidence should be low (0.0)
            target_confidence = reasoning_correct
            
            # Calibration loss: confidence should match correctness
            quality_loss = F.mse_loss(outputs.reasoning_confidence, target_confidence)
        
        # Total loss
        total_loss = (
            self.alpha_reasoning * reasoning_loss + 
            self.alpha_answer * answer_loss +
            self.alpha_quality * quality_loss
        )
        
        # Confidence scale for adaptive weighting
        if outputs.reasoning_confidence is not None:
            confidence_scale = outputs.reasoning_confidence.mean().item()
        else:
            confidence_scale = 1.0
        
        return total_loss, {
            'reasoning_loss': reasoning_loss.item(),
            'answer_loss': answer_loss.item(),
            'quality_loss': quality_loss if isinstance(quality_loss, float) else quality_loss.item(),
            'confidence_scale': confidence_scale,
            'total_loss': total_loss.item(),
            'unweighted_total': (reasoning_loss + answer_loss).item()
        }


# ============================================================================
# 3. TRAINER WITH FULL RESUME SUPPORT
# ============================================================================

class VQATrainer:
    """
    Professional trainer với:
    - Full checkpoint resume (optimizer, scheduler, scaler, epoch)
    - Momentum scheduling
    - Reasoning quality validation
    - Stage-based training
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        output_dir,
        batch_size=2,
        gradient_accumulation_steps=32,
        num_epochs=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        alpha_reasoning=0.6,
        alpha_answer=0.4,
        alpha_quality=0.1,
        label_smoothing=0.1,
        use_amp=True,
        patience=5,
        log_steps=10,
        eval_steps=0,
        save_steps=0,
        use_wandb=False,
        resume_checkpoint=None,  # Path to checkpoint for resume
        load_optimizer=True,  # Whether to load optimizer state (False for stage transition)
        stage_name="main"  # Training stage name
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
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.use_wandb = use_wandb
        self.stage_name = stage_name
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Reduced to save RAM
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Reduced to save RAM
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Loss function
        self.criterion = ChainOfThoughtLoss(
            alpha_reasoning=alpha_reasoning,
            alpha_answer=alpha_answer,
            alpha_quality=alpha_quality,
            label_smoothing=label_smoothing
        )
        
        # Optimizer với momentum
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # Standard momentum values
            eps=1e-8
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = self.output_dir / f'best_model_{stage_name}.pt'
        
        # CSV logger for metrics tracking
        self.csv_log_path = self.output_dir / f'training_log_{stage_name}.csv'
        self.init_csv_logger()
        
        # Resume từ checkpoint nếu có
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint, load_optimizer=load_optimizer)
        
        print(f"\n[INFO] Trainer initialized for stage: {stage_name}")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Training steps: {num_training_steps}")
        print(f"  Warmup steps: {num_warmup_steps}")
        if resume_checkpoint:
            print(f"  Resumed from epoch {self.current_epoch}")
    
    def init_csv_logger(self):
        """Initialize CSV logger for tracking metrics"""
        # Check if file exists and has content (resume case)
        if self.csv_log_path.exists() and self.current_epoch > 0:
            print(f"[INFO] Resuming CSV logging to {self.csv_log_path}")
            return
        
        # Create new CSV with headers
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'train_reasoning_loss',
                'train_answer_loss',
                'train_quality_loss',
                'train_confidence_scale',
                'val_loss',
                'val_reasoning_loss',
                'val_answer_loss',
                'val_quality_loss',
                'val_confidence_scale',
                'val_reasoning_conf_mean',
                'val_reasoning_conf_std',
                'learning_rate',
                'patience_counter',
                'is_best'
            ])
        print(f"[INFO] CSV logging initialized: {self.csv_log_path}")
    
    def log_to_csv(self, epoch, train_components, val_components, is_best):
        """Log epoch metrics to CSV"""
        current_lr = self.scheduler.get_last_lr()[0]
        
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_components.get('total_loss', 0),
                train_components.get('reasoning_loss', 0),
                train_components.get('answer_loss', 0),
                train_components.get('quality_loss', 0),
                train_components.get('confidence_scale', 0),
                val_components.get('total_loss', 0),
                val_components.get('reasoning_loss', 0),
                val_components.get('answer_loss', 0),
                val_components.get('quality_loss', 0),
                val_components.get('confidence_scale', 0),
                val_components.get('reasoning_confidence_mean', 0),
                val_components.get('reasoning_confidence_std', 0),
                f"{current_lr:.2e}",
                self.patience_counter,
                1 if is_best else 0
            ])
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save full training state for resume"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'stage_name': self.stage_name,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"[INFO] ✓ Best model saved: {self.best_model_path} (Epoch {epoch+1}, Loss: {val_loss:.4f})")
        
        # Also save latest checkpoint
        latest_path = self.output_dir / f'checkpoint_{self.stage_name}_latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """Load training state for resume
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: If False, only load model weights (for stage transition)
                          If True, load full state including optimizer (for same-stage resume)
        """
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Always load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] ✓ Model weights loaded")
        
        if load_optimizer:
            # Load optimizer state (for same-stage resume)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                self.global_step = checkpoint['global_step']
                self.best_val_loss = checkpoint['best_val_loss']
                self.patience_counter = checkpoint.get('patience_counter', 0)
                
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                print(f"[INFO] ✓ Full state loaded. Resuming from epoch {self.current_epoch}")
                print(f"[INFO]   Best val loss: {self.best_val_loss:.4f}")
                print(f"[INFO]   Global step: {self.global_step}")
            except ValueError as e:
                print(f"[WARNING] Failed to load optimizer state: {e}")
                print(f"[INFO] Continuing with fresh optimizer (this is normal for stage transition)")
        else:
            # Only load model weights (for stage transition)
            print(f"[INFO] ✓ Model-only load (fresh optimizer for new stage)")
            # Reset training state for new stage
            self.current_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move only tensor fields to device (skip string fields like img_id, question)
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                           if torch.is_tensor(v)}
            
            # Prepare labels (mask padding tokens with -100)
            reasoning_labels = tensor_batch['reasoning_input_ids'].clone()
            reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
            
            answer_labels = tensor_batch['answer_input_ids'].clone()
            answer_labels[answer_labels == self.model.tokenizer.pad_token_id] = -100
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=tensor_batch['pixel_values'],
                    input_ids=tensor_batch['input_ids'],
                    attention_mask=tensor_batch['attention_mask'],
                    reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                    reasoning_attention_mask=tensor_batch['reasoning_attention_mask'],
                    answer_input_ids=tensor_batch['answer_input_ids'],
                    answer_attention_mask=tensor_batch['answer_attention_mask']
                )
                
                loss, loss_dict = self.criterion(outputs, reasoning_labels, answer_labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
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
            
            # Accumulate losses
            total_loss += loss.item() * self.gradient_accumulation_steps
            for k, v in loss_dict.items():
                loss_components[k] += v
            
            # Clear cache periodically to prevent memory buildup
            if (step + 1) % (self.gradient_accumulation_steps * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update progress bar
            if (step + 1) % self.log_steps == 0:
                avg_loss = total_loss / (step + 1)
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_components:
            loss_components[k] /= len(self.train_loader)
        
        return avg_loss, dict(loss_components)
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate với reasoning quality metrics"""
        self.model.eval()
        total_loss = 0
        loss_components = defaultdict(float)
        reasoning_confidences = []
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            # Move only tensor fields to device (skip string fields like img_id, question)
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                           if torch.is_tensor(v)}
            
            # Prepare labels (mask padding tokens with -100)
            reasoning_labels = tensor_batch['reasoning_input_ids'].clone()
            reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
            
            answer_labels = tensor_batch['answer_input_ids'].clone()
            answer_labels[answer_labels == self.model.tokenizer.pad_token_id] = -100
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=tensor_batch['pixel_values'],
                    input_ids=tensor_batch['input_ids'],
                    attention_mask=tensor_batch['attention_mask'],
                    reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                    reasoning_attention_mask=tensor_batch['reasoning_attention_mask'],
                    answer_input_ids=tensor_batch['answer_input_ids'],
                    answer_attention_mask=tensor_batch['answer_attention_mask']
                )
                
                loss, loss_dict = self.criterion(outputs, reasoning_labels, answer_labels)
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v
            
            # Collect reasoning confidences
            if outputs.reasoning_confidence is not None:
                reasoning_confidences.extend(outputs.reasoning_confidence.cpu().numpy())
        
        # Average
        avg_loss = total_loss / len(self.val_loader)
        for k in loss_components:
            loss_components[k] /= len(self.val_loader)
        
        # Reasoning quality stats
        if reasoning_confidences:
            loss_components['reasoning_confidence_mean'] = np.mean(reasoning_confidences)
            loss_components['reasoning_confidence_std'] = np.std(reasoning_confidences)
        
        return avg_loss, dict(loss_components)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"TRAINING STAGE: {self.stage_name}")
        print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            # Train
            train_loss, train_components = self.train_epoch(epoch)
            
            # Log training
            print(f"\n[EPOCH {epoch+1}] Train Loss: {train_loss:.4f}")
            for k, v in train_components.items():
                print(f"  {k}: {v:.4f}")
            
            # Validate
            val_loss, val_components = self.validate(epoch)
            
            # Log validation
            print(f"\n[VALIDATION] Loss: {val_loss:.4f}")
            for k, v in val_components.items():
                print(f"  {k}: {v:.4f}")
            
            # Save best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                print(f"[INFO] No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Save latest checkpoint
            self.save_checkpoint(epoch, val_loss, is_best=False)
            
            # Log to CSV
            self.log_to_csv(epoch, train_components, val_components, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print(f"STAGE {self.stage_name} COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best model: {self.best_model_path}")


# ============================================================================
# 4. STAGE TRAINING STRATEGY
# ============================================================================

def freeze_module(module, name):
    """Freeze all parameters in module"""
    for param in module.parameters():
        param.requires_grad = False
    print(f"[INFO] ✓ Frozen {name}")


def unfreeze_module(module, name):
    """Unfreeze all parameters in module"""
    for param in module.parameters():
        param.requires_grad = True
    print(f"[INFO] ✓ Unfrozen {name}")


def unfreeze_last_n_layers(module, name, n_layers=2):
    """Unfreeze last N transformer layers"""
    # For DINOv2
    if hasattr(module, 'encoder') and hasattr(module.encoder, 'layer'):
        total_layers = len(module.encoder.layer)
        for i, layer in enumerate(module.encoder.layer):
            if i >= total_layers - n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"[INFO] ✓ Unfrozen last {n_layers} layers of {name}")
    # For BARTpho encoder
    elif hasattr(module, 'layers'):
        total_layers = len(module.layers)
        for i, layer in enumerate(module.layers):
            if i >= total_layers - n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"[INFO] ✓ Unfrozen last {n_layers} layers of {name}")


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    """Main training với staged unfreezing"""
    
    CONFIG = {
        # Paths
        'train_json': '/kaggle/input/teacher-3-12/teacher_outputs_train.jsonl',
        'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
        'output_dir': '/kaggle/working/checkpoints_dinov2_bartpho',
        
        # Data
        'val_split': 0.1,
        'random_seed': 42,
        
        # Training
        'batch_size': 1,  # Reduced to save GPU memory
        'gradient_accumulation_steps': 64,  # Increased to maintain effective batch size
        'num_epochs': 54,  # Total epochs across all stages
        'learning_rate': 3e-5,  # Lower LR cho large model
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        
        # Loss weights
        'alpha_reasoning': 0.6,
        'alpha_answer': 0.4,
        'alpha_quality': 0.1,
        'label_smoothing': 0.0,  # Removed to lower loss (was causing artificially high loss)
        
        # Advanced
        'use_amp': True,
        'patience': 8,  # Increased to allow more training
        'log_steps': 10,
        'eval_steps': 0,
        'save_steps': 0,
        'use_wandb': False,
        
        # Resume
        'resume_from': None,  # Stage 1 complete, resume at stage 2 level
    }
    
    print("="*70)
    print("SOTA VQA TRAINING: DINOv2 + BARTpho")
    print("="*70)
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Seed
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    # Initialize model
    print("\n[INFO] Initializing model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        num_heads=16,  # Must be divisible: 1024 ÷ 16 = 64
        use_reasoning_quality_check=True,
        gradient_checkpointing=True
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"[INFO] Total params: {total_params/1e6:.1f}M")
    print(f"[INFO] Trainable params: {trainable_params/1e6:.1f}M")
    
    # Load dataset
    print("\n[INFO] Loading dataset...")
    full_dataset = VQADistillationDataset(
        json_path=CONFIG['train_json'],
        image_dir=CONFIG['image_dir'],
        vision_processor=model.vision_processor,
        tokenizer=model.tokenizer,
        augment=False
    )
    
    # Split
    total_size = len(full_dataset)
    val_size = int(total_size * CONFIG['val_split'])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create augmented train dataset
    train_dataset_aug = VQADistillationDataset(
        json_path=CONFIG['train_json'],
        image_dir=CONFIG['image_dir'],
        vision_processor=model.vision_processor,
        tokenizer=model.tokenizer,
        augment=True
    )
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset_aug, train_dataset.indices)
    
    # ===== STAGED TRAINING =====
    print("\n" + "="*70)
    print("STAGED TRAINING STRATEGY")
    print("="*70)
    print("Stage 1: Train fusion + heads only (15 epochs)")
    print("Stage 2: Unfreeze DINOv2 last 4 layers (12 epochs)")
    print("Stage 3: Unfreeze BARTpho encoder last 6 layers (12 epochs)")
    print("Stage 4: Full fine-tuning (15 epochs)")
    print("Total: 54 epochs")
    print("="*70 + "\n")
    
    # === STAGE 1: Fusion + Heads only ===
    print("\n[STAGE 1/4] Training fusion + heads (encoders frozen)...")
    freeze_module(model.vision_encoder, "DINOv2")
    freeze_module(model.bartpho.model.encoder, "BARTpho Encoder")
    freeze_module(model.bartpho.model.decoder, "BARTpho Decoder")
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    
    trainer_s1 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=15,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        alpha_quality=CONFIG['alpha_quality'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb'],
        resume_checkpoint=CONFIG['resume_from'],
        stage_name="stage1"
    )
    trainer_s1.train()
    
    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] GPU Memory after Stage 1: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # === STAGE 2: Unfreeze DINOv2 last layers ===
    print("\n[STAGE 2/4] Unfreezing DINOv2 last 4 layers...")
    unfreeze_last_n_layers(model.vision_encoder, "DINOv2", n_layers=4)
    
    # Resume checkpoint for stage 2 (use latest to continue from last epoch)
    stage2_resume = '/kaggle/input/s2/transformers/default/1/checkpoint_stage2_latest.pt'
    
    trainer_s2 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=12,
        learning_rate=CONFIG['learning_rate'] * 0.8,  # Increased from 0.5x to learn faster
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        alpha_quality=CONFIG['alpha_quality'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb'],
        resume_checkpoint=stage2_resume,  # Resume from stage 2 latest
        load_optimizer=True,  # Load optimizer state to continue same stage
        stage_name="stage2"
    )
    trainer_s2.train()
    
    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] GPU Memory after Stage 2: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # === STAGE 3: Unfreeze BARTpho encoder last layers ===
    print("\n[STAGE 3/4] Unfreezing BARTpho encoder last 6 layers...")
    unfreeze_last_n_layers(model.bartpho.model.encoder, "BARTpho Encoder", n_layers=6)
    
    trainer_s3 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=12,
        learning_rate=CONFIG['learning_rate'] * 0.5,  # Increased from 0.3x to learn faster
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        alpha_quality=CONFIG['alpha_quality'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb'],
        resume_checkpoint=trainer_s2.best_model_path,
        load_optimizer=False,  # Stage transition - model weights only
        stage_name="stage3"
    )
    trainer_s3.train()
    
    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] GPU Memory after Stage 3: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # === STAGE 4: Full fine-tuning ===
    print("\n[STAGE 4/4] Full fine-tuning (all layers unfrozen)...")
    unfreeze_module(model.vision_encoder, "DINOv2")
    unfreeze_module(model.bartpho.model.encoder, "BARTpho Encoder")
    unfreeze_last_n_layers(model.bartpho.model.decoder, "BARTpho Decoder", n_layers=6)
    
    trainer_s4 = VQATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=15,
        learning_rate=CONFIG['learning_rate'] * 0.2,  # Increased from 0.1x for better final tuning
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        alpha_quality=CONFIG['alpha_quality'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        use_wandb=CONFIG['use_wandb'],
        resume_checkpoint=trainer_s3.best_model_path,
        load_optimizer=False,  # Stage transition - model weights only
        stage_name="stage4"
    )
    trainer_s4.train()
    
    print("\n[INFO] Training completed successfully!")
    print(f"[INFO] Final best model: {trainer_s4.best_model_path}")


if __name__ == '__main__':
    main()
