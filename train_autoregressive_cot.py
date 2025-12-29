"""
AUTOREGRESSIVE COT TRAINING: DINOv2 + BARTpho VQA
=================================================
Training với autoregressive reasoning generation:
✅ Generate reasoning first (no teacher forcing)
✅ Use generated reasoning as input for answer
✅ Scheduled sampling (mix teacher forcing & generation)
✅ Better inference alignment

Flow:
1. Generate reasoning: reasoning = model.generate(image, question)
2. Generate answer: answer = model.generate(image, question, reasoning)
3. Loss: L = α * L_reasoning + β * L_answer
"""

import os
import json
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

# Import model
from model_dinov2_bartpho import DINOv2BARTphoVQA, count_parameters


# ============================================================================
# 1. DATASET (Same as before)
# ============================================================================

class VQADistillationDataset(Dataset):
    """Dataset cho DINOv2 + BARTpho"""
    
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
        
        # Tokenize reasoning (ground truth)
        reasoning_enc = self.tokenizer(
            item.get('reasoning', item.get('teacher_reasoning', '')),
            max_length=self.max_reasoning_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer (ground truth)
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
            'img_id': item.get('image_id', item.get('img_id', f"img_{idx}")),
            'question': item['question'],
        }


# ============================================================================
# 2. AUTOREGRESSIVE COT LOSS
# ============================================================================

class AutoregressiveCOTLoss(nn.Module):
    """
    Loss for autoregressive CoT training
    
    Loss = α * L_reasoning + β * L_answer
    
    Both reasoning and answer are generated, then compared to ground truth
    """
    
    def __init__(
        self, 
        alpha_reasoning=0.6, 
        alpha_answer=0.4,
        label_smoothing=0.0
    ):
        super().__init__()
        self.alpha_reasoning = alpha_reasoning
        self.alpha_answer = alpha_answer
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100, 
            label_smoothing=label_smoothing
        )
        
    def forward(self, reasoning_logits, answer_logits, reasoning_labels, answer_labels):
        """
        Args:
            reasoning_logits: [batch, reasoning_len, vocab]
            answer_logits: [batch, answer_len, vocab]
            reasoning_labels: [batch, reasoning_len]
            answer_labels: [batch, answer_len]
        """
        # Reasoning loss
        reasoning_loss = self.criterion(
            reasoning_logits.view(-1, reasoning_logits.size(-1)),
            reasoning_labels.view(-1)
        )
        
        # Answer loss
        answer_loss = self.criterion(
            answer_logits.view(-1, answer_logits.size(-1)),
            answer_labels.view(-1)
        )
        
        # Total loss
        total_loss = self.alpha_reasoning * reasoning_loss + self.alpha_answer * answer_loss
        
        return total_loss, {
            'reasoning_loss': reasoning_loss.item(),
            'answer_loss': answer_loss.item(),
            'total_loss': total_loss.item(),
        }


# ============================================================================
# 3. AUTOREGRESSIVE COT TRAINER
# ============================================================================

class AutoregressiveCOTTrainer:
    """
    Trainer với autoregressive reasoning generation
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
        label_smoothing=0.0,
        use_amp=True,
        patience=5,
        log_steps=10,
        save_steps=100,  # Save checkpoint every N steps (OOM safety)
        validate_every_n_epochs=1,  # Reduce validation frequency to save memory
        scheduled_sampling_start=0.0,  # Start with 0% teacher forcing
        scheduled_sampling_end=0.0,    # End with 0% teacher forcing
        scheduled_sampling_anneal_epochs=10,  # Anneal over N epochs
        resume_checkpoint=None,
        load_optimizer=True,
        stage_name="main",
        stage_milestones=None,  # Dict: {epoch: (stage_name, lr_scale, unfreeze_actions)}
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
        self.validate_every_n_epochs = validate_every_n_epochs
        self.stage_name = stage_name
        self.stage_milestones = stage_milestones or {}
        self.base_learning_rate = learning_rate
        
        # Scheduled sampling params
        self.scheduled_sampling_start = scheduled_sampling_start
        self.scheduled_sampling_end = scheduled_sampling_end
        self.scheduled_sampling_anneal_epochs = scheduled_sampling_anneal_epochs
        
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
        self.criterion = AutoregressiveCOTLoss(
            alpha_reasoning=alpha_reasoning,
            alpha_answer=alpha_answer,
            label_smoothing=label_smoothing
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Training state (initialize before loading checkpoint)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = self.output_dir / f'best_model_{stage_name}.pt'
        
        # CSV logger
        self.csv_log_path = self.output_dir / f'training_log_{stage_name}.csv'
        
        # Resume (BEFORE creating scheduler)
        if resume_checkpoint:
            # Auto-detect epoch offset from old stage-based checkpoints
            epoch_offset = 0
            load_opt = load_optimizer
            
            if 'stage2' in str(resume_checkpoint):
                epoch_offset = 15
                load_opt = False
            elif 'stage3' in str(resume_checkpoint):
                epoch_offset = 27
                load_opt = False
            elif 'stage4' in str(resume_checkpoint):
                epoch_offset = 39
                load_opt = False
            
            if epoch_offset > 0:
                # Old checkpoint - load model only, manually set epoch
                self.load_checkpoint(resume_checkpoint, load_optimizer=False)
                try:
                    checkpoint = torch.load(resume_checkpoint, map_location=self.device)
                    checkpoint_epoch = checkpoint.get('epoch', 0)
                    self.current_epoch = checkpoint_epoch + epoch_offset + 1
                    print(f"[INFO] Manually set global epoch: {self.current_epoch} (checkpoint epoch {checkpoint_epoch} + offset {epoch_offset} + 1)")
                except:
                    pass
            else:
                # New progressive checkpoint
                self.load_checkpoint(resume_checkpoint, load_optimizer=load_opt)
        
        print(f"\n[INFO] Autoregressive COT Trainer initialized")
        print(f"  Stage: {stage_name}")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Scheduled sampling: {scheduled_sampling_start} → {scheduled_sampling_end}")
    
    def init_csv_logger(self):
        """Initialize CSV logger"""
        if self.csv_log_path.exists() and self.current_epoch > 0:
            print(f"[INFO] Resuming CSV logging to {self.csv_log_path}")
            return
        
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss',
                'train_reasoning_loss',
                'train_answer_loss',
                'val_loss',
                'val_reasoning_loss',
                'val_answer_loss',
                'learning_rate',
                'teacher_forcing_ratio',
                'patience_counter',
                'is_best'
            ])
    
    def log_to_csv(self, epoch, train_components, val_components, teacher_forcing_ratio, is_best):
        """Log metrics to CSV"""
        current_lr = self.scheduler.get_last_lr()[0]
        
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_components.get('total_loss', 0),
                train_components.get('reasoning_loss', 0),
                train_components.get('answer_loss', 0),
                val_components.get('total_loss', 0),
                val_components.get('reasoning_loss', 0),
                val_components.get('answer_loss', 0),
                f"{current_lr:.2e}",
                f"{teacher_forcing_ratio:.3f}",
                self.patience_counter,
                1 if is_best else 0
            ])
    
    def get_teacher_forcing_ratio(self, epoch):
        """Get teacher forcing ratio for scheduled sampling"""
        if epoch >= self.scheduled_sampling_anneal_epochs:
            return self.scheduled_sampling_end
        
        # Linear annealing
        progress = epoch / self.scheduled_sampling_anneal_epochs
        ratio = self.scheduled_sampling_start + (self.scheduled_sampling_end - self.scheduled_sampling_start) * progress
        return ratio
    
    def save_checkpoint(self, epoch, val_loss=None, is_best=False, force_save=False):
        """Save checkpoint with OOM safety"""
        try:
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
            
            # Always save latest for OOM recovery
            latest_path = self.output_dir / f'checkpoint_{self.stage_name}_latest.pt'
            torch.save(checkpoint, latest_path)
            
            if is_best and val_loss is not None:
                torch.save(checkpoint, self.best_model_path)
                print(f"[INFO] ✓ Best model saved (Loss: {val_loss:.4f})")
            elif force_save:
                print(f"[INFO] ✓ Emergency checkpoint saved (Step: {self.global_step})")
            
            # Clear memory after saving
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            # Try minimal save
            try:
                torch.save({'model_state_dict': self.model.state_dict()}, 
                          self.output_dir / f'emergency_{self.stage_name}.pt')
                print(f"[INFO] ✓ Emergency model weights saved")
            except:
                print(f"[ERROR] Could not save emergency checkpoint")
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """Load checkpoint"""
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] ✓ Model weights loaded")
        
        if load_optimizer:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_epoch = checkpoint['epoch'] + 1
                # Keep global_step for smooth LR schedule continuation
                self.global_step = checkpoint['global_step']
                self.best_val_loss = checkpoint['best_val_loss']
                self.patience_counter = checkpoint.get('patience_counter', 0)
                
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                print(f"[INFO] ✓ Full state loaded (smooth LR continuation). Resuming from epoch {self.current_epoch}")
            except Exception as e:
                print(f"[WARNING] Failed to load optimizer: {e}")
        else:
            print(f"[INFO] ✓ Model-only load")
            self.current_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.patience_counter = 0
        
        # Initialize CSV logger AFTER loading checkpoint
        self.init_csv_logger()
        
        # Create scheduler AFTER loading checkpoint (based on remaining epochs)
        remaining_epochs = num_epochs - self.current_epoch
        num_training_steps = len(self.train_loader) * remaining_epochs // gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * warmup_ratio) if self.current_epoch == 0 else 0
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"[INFO] Scheduler: {remaining_epochs} remaining epochs, {num_training_steps} steps, {num_warmup_steps} warmup")
    
    def handle_stage_transition(self, epoch):
        """Check and handle stage transitions based on epoch milestones"""
        if epoch in self.stage_milestones:
            stage_name, lr_scale, unfreeze_actions = self.stage_milestones[epoch]
            print(f"\n{'='*70}")
            print(f"[STAGE TRANSITION] Epoch {epoch+1}: {stage_name}")
            print(f"{'='*70}")
            
            # Unfreeze modules
            for action, module_name, *args in unfreeze_actions:
                if action == 'unfreeze_last_n':
                    n_layers = args[0] if args else 2
                    module = self.get_module_by_name(module_name)
                    if module:
                        unfreeze_last_n_layers(module, module_name, n_layers)
                elif action == 'unfreeze_all':
                    module = self.get_module_by_name(module_name)
                    if module:
                        unfreeze_module(module, module_name)
            
            # Update learning rate
            new_lr = self.base_learning_rate * lr_scale
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"[INFO] Learning rate updated: {new_lr:.2e}")
            print(f"{'='*70}\n")
    
    def get_module_by_name(self, name):
        """Get module by string name"""
        if name == 'vision_encoder':
            return self.model.vision_encoder
        elif name == 'bartpho_encoder':
            return self.model.bartpho.model.encoder
        elif name == 'bartpho_decoder':
            return self.model.bartpho.model.decoder
        return None
    
    def train_epoch(self, epoch):
        """Train one epoch with autoregressive generation"""
        # Handle stage transitions at the beginning of epoch
        self.handle_stage_transition(epoch)
        
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(epoch)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                           if torch.is_tensor(v)}
            
            # Prepare labels
            reasoning_labels = tensor_batch['reasoning_input_ids'].clone()
            reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
            
            answer_labels = tensor_batch['answer_input_ids'].clone()
            answer_labels[answer_labels == self.model.tokenizer.pad_token_id] = -100
            
            # Decide: use teacher forcing or generation
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            try:
                with autocast('cuda', enabled=self.use_amp):
                    if use_teacher_forcing:
                        # Teacher forcing: use ground truth reasoning
                        outputs = self.model(
                            pixel_values=tensor_batch['pixel_values'],
                            input_ids=tensor_batch['input_ids'],
                            attention_mask=tensor_batch['attention_mask'],
                            reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                            reasoning_attention_mask=tensor_batch['reasoning_attention_mask'],
                            answer_input_ids=tensor_batch['answer_input_ids'],
                            answer_attention_mask=tensor_batch['answer_attention_mask']
                        )
                        reasoning_logits = outputs.reasoning_logits
                        answer_logits = outputs.answer_logits
                    else:
                        # Autoregressive: generate reasoning first
                        with torch.no_grad():  # No gradients for generation
                            # First encode and fuse
                            visual_features = self.model.encode_image(tensor_batch['pixel_values'])
                            text_features = self.model.encode_text(
                                tensor_batch['input_ids'],
                                tensor_batch['attention_mask']
                            )
                            fused_features, _ = self.model.fuse_multimodal(text_features, visual_features)
                            
                            # Then generate reasoning using bartpho.generate
                            from transformers.modeling_outputs import BaseModelOutput
                            encoder_outputs_wrapped = BaseModelOutput(last_hidden_state=fused_features)
                            reasoning_outputs = self.model.bartpho.generate(
                                encoder_outputs=encoder_outputs_wrapped,
                                max_length=96,
                                num_beams=1,  # Greedy for memory efficiency
                                pad_token_id=self.model.tokenizer.pad_token_id,
                                eos_token_id=self.model.tokenizer.eos_token_id,
                                bos_token_id=self.model.tokenizer.bos_token_id,
                            )
                            # Detach to save memory
                            reasoning_outputs = reasoning_outputs.detach()
                            fused_features = fused_features.detach()
                            visual_features = None
                            text_features = None
                        
                        # Clear cache after generation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Forward pass with generated reasoning
                        reasoning_forward = self.model(
                            pixel_values=tensor_batch['pixel_values'],
                            input_ids=tensor_batch['input_ids'],
                            attention_mask=tensor_batch['attention_mask'],
                            reasoning_input_ids=reasoning_outputs,
                            reasoning_attention_mask=(reasoning_outputs != self.model.tokenizer.pad_token_id).long(),
                            answer_input_ids=tensor_batch['answer_input_ids'],
                            answer_attention_mask=tensor_batch['answer_attention_mask']
                        )
                        reasoning_logits = reasoning_forward.reasoning_logits
                        answer_logits = reasoning_forward.answer_logits
                    
                    # Compute loss
                    loss, loss_dict = self.criterion(
                        reasoning_logits, answer_logits, 
                        reasoning_labels, answer_labels
                    )
                    loss = loss / self.gradient_accumulation_steps
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[WARNING] OOM at step {step}! Saving checkpoint and clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Save emergency checkpoint
                    self.save_checkpoint(epoch, force_save=True)
                    # Skip this batch
                    continue
                else:
                    raise e
            
            # Backward
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
            
            # Save periodic checkpoint (OOM safety)
            if self.save_steps > 0 and self.global_step > 0 and self.global_step % self.save_steps == 0:
                print(f"\n[INFO] Saving periodic checkpoint at step {self.global_step}...")
                self.save_checkpoint(epoch, force_save=True)
            
            # Aggressive cache clearing for autoregressive generation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update progress
            if (step + 1) % self.log_steps == 0:
                avg_loss = total_loss / (step + 1)
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'tf': f'{teacher_forcing_ratio:.2f}'
                })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for k in loss_components:
            loss_components[k] /= len(self.train_loader)
        
        return avg_loss, dict(loss_components)
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate with full generation (memory-efficient)"""
        self.model.eval()
        total_loss = 0
        loss_components = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                tensor_batch = {k: v.to(self.device) for k, v in batch.items() 
                               if torch.is_tensor(v)}
                
                # Prepare labels
                reasoning_labels = tensor_batch['reasoning_input_ids'].clone()
                reasoning_labels[reasoning_labels == self.model.tokenizer.pad_token_id] = -100
                
                answer_labels = tensor_batch['answer_input_ids'].clone()
                answer_labels[answer_labels == self.model.tokenizer.pad_token_id] = -100
                
                with autocast('cuda', enabled=self.use_amp):
                    # Encode and fuse first
                    visual_features = self.model.encode_image(tensor_batch['pixel_values'])
                    text_features = self.model.encode_text(
                        tensor_batch['input_ids'],
                        tensor_batch['attention_mask']
                    )
                    fused_features, _ = self.model.fuse_multimodal(text_features, visual_features)
                    
                    # Generate reasoning using bartpho.generate
                    from transformers.modeling_outputs import BaseModelOutput
                    encoder_outputs_wrapped = BaseModelOutput(last_hidden_state=fused_features)
                    reasoning_outputs = self.model.bartpho.generate(
                        encoder_outputs=encoder_outputs_wrapped,
                        max_length=96,
                        num_beams=1,
                        pad_token_id=self.model.tokenizer.pad_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                        bos_token_id=self.model.tokenizer.bos_token_id,
                    )
                    
                    # Clear cache after generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Forward pass to get logits
                    outputs = self.model(
                        pixel_values=tensor_batch['pixel_values'],
                        input_ids=tensor_batch['input_ids'],
                        attention_mask=tensor_batch['attention_mask'],
                        reasoning_input_ids=reasoning_outputs,
                        reasoning_attention_mask=(reasoning_outputs != self.model.tokenizer.pad_token_id).long(),
                        answer_input_ids=tensor_batch['answer_input_ids'],
                        answer_attention_mask=tensor_batch['answer_attention_mask']
                    )
                    
                    loss, loss_dict = self.criterion(
                        outputs.reasoning_logits, outputs.answer_logits,
                        reasoning_labels, answer_labels
                    )
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_components[k] += v
                num_batches += 1
                
                # Periodic cache clearing
                if (batch_idx + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[WARNING] OOM during validation at batch {batch_idx}! Skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Average
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            for k in loss_components:
                loss_components[k] /= num_batches
        else:
            print("[WARNING] No batches completed in validation!")
            avg_loss = float('inf')
            loss_components = {'reasoning_loss': 0, 'answer_loss': 0, 'total_loss': 0}
        
        return avg_loss, dict(loss_components)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"AUTOREGRESSIVE COT TRAINING: {self.stage_name}")
        print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            # Get teacher forcing ratio
            tf_ratio = self.get_teacher_forcing_ratio(epoch)
            
            # Train
            try:
                train_loss, train_components = self.train_epoch(epoch)
                
                print(f"\n[EPOCH {epoch+1}] Train Loss: {train_loss:.4f} (TF: {tf_ratio:.2f})")
                for k, v in train_components.items():
                    print(f"  {k}: {v:.4f}")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[ERROR] OOM during training epoch {epoch+1}!")
                    self.save_checkpoint(epoch, force_save=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
            
            # Validate (skip if not validation epoch)
            should_validate = (epoch + 1) % self.validate_every_n_epochs == 0
            
            if should_validate:
                try:
                    val_loss, val_components = self.validate(epoch)
                    
                    print(f"\n[VALIDATION] Loss: {val_loss:.4f}")
                    for k, v in val_components.items():
                        print(f"  {k}: {v:.4f}")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n[WARNING] OOM during validation! Skipping validation this epoch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        val_loss = float('inf')
                        val_components = {'reasoning_loss': 0, 'answer_loss': 0, 'total_loss': 0}
                    else:
                        raise e
            else:
                print(f"\n[INFO] Skipping validation (every {self.validate_every_n_epochs} epochs)")
                val_loss = self.best_val_loss  # Use previous best
                val_components = {'reasoning_loss': 0, 'answer_loss': 0, 'total_loss': 0}
            
            # Save best
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                print(f"[INFO] No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Save latest
            self.save_checkpoint(epoch, val_loss, is_best=False)
            
            # Log to CSV
            self.log_to_csv(epoch, train_components, val_components, tf_ratio, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# ============================================================================
# 4. STAGE FUNCTIONS
# ============================================================================

def freeze_module(module, name):
    for param in module.parameters():
        param.requires_grad = False
    print(f"[INFO] ✓ Frozen {name}")

def unfreeze_module(module, name):
    for param in module.parameters():
        param.requires_grad = True
    print(f"[INFO] ✓ Unfrozen {name}")

def unfreeze_last_n_layers(module, name, n_layers=2):
    if hasattr(module, 'encoder') and hasattr(module.encoder, 'layer'):
        total_layers = len(module.encoder.layer)
        for i, layer in enumerate(module.encoder.layer):
            if i >= total_layers - n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"[INFO] ✓ Unfrozen last {n_layers} layers of {name}")
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
    """Main training with autoregressive CoT"""
    
    CONFIG = {
        # Paths
        'train_json': '/kaggle/input/teacher-3-12/teacher_outputs_train.jsonl',
        'image_dir': '/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train',
        'output_dir': '/kaggle/working/checkpoints_autoregressive_cot',
        
        # Data
        'val_split': 0.1,
        'random_seed': 42,
        
        # Training
        'batch_size': 1,
        'gradient_accumulation_steps': 64,
        'num_epochs': 54,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        
        # Loss weights
        'alpha_reasoning': 0.6,
        'alpha_answer': 0.4,
        'label_smoothing': 0.0,
        
        # Scheduled sampling
        'scheduled_sampling_start': 1.0,  # Start with 100% teacher forcing
        'scheduled_sampling_end': 0.0,    # End with 0% teacher forcing (full generation)
        'scheduled_sampling_anneal_epochs': 10,  # Anneal over first 10 epochs
        
        # Advanced
        'use_amp': True,
        'patience': 8,
        'log_steps': 10,
        
        # Resume
        'resume_from': None,
    }
    
    print("="*70)
    print("AUTOREGRESSIVE COT TRAINING: DINOv2 + BARTpho")
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
        num_heads=16,
        use_reasoning_quality_check=False,  # Not used in autoregressive training
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
    
    # ===== STAGED TRAINING WITH AUTOMATIC UNFREEZING =====
    print("\n" + "="*70)
    print("AUTOREGRESSIVE COT STAGED TRAINING (Automatic)")
    print("="*70)
    print("Epoch 0-14: Stage 1 - Fusion + heads only (15 epochs)")
    print("Epoch 15-26: Stage 2 - Unfreeze DINOv2 last 4 layers (12 epochs)")
    print("Epoch 27-38: Stage 3 - Unfreeze BARTpho encoder last 6 layers (12 epochs)")
    print("Epoch 39-53: Stage 4 - Full fine-tuning (15 epochs)")
    print("Total: 54 epochs")
    print("="*70 + "\n")
    
    # Freeze all encoders initially (Stage 1 setup)
    print("[INFO] Initial setup: Freezing all encoders...")
    freeze_module(model.vision_encoder, "DINOv2")
    freeze_module(model.bartpho.model.encoder, "BARTpho Encoder")
    freeze_module(model.bartpho.model.decoder, "BARTpho Decoder")
    
    # Define stage milestones
    stage_milestones = {
        15: ('Stage 2: Unfreeze DINOv2 last 4 layers', 0.8, [
            ('unfreeze_last_n', 'vision_encoder', 4)
        ]),
        27: ('Stage 3: Unfreeze BARTpho encoder last 6 layers', 0.5, [
            ('unfreeze_last_n', 'bartpho_encoder', 6)
        ]),
        39: ('Stage 4: Full fine-tuning', 0.2, [
            ('unfreeze_all', 'vision_encoder'),
            ('unfreeze_all', 'bartpho_encoder'),
            ('unfreeze_last_n', 'bartpho_decoder', 6)
        ])
    }
    
    # Single trainer for all 54 epochs
    trainer = AutoregressiveCOTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=54,  # Total epochs
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning=CONFIG['alpha_reasoning'],
        alpha_answer=CONFIG['alpha_answer'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        save_steps=100,  # Save every 100 steps for OOM safety
        validate_every_n_epochs=1,
        scheduled_sampling_start=CONFIG['scheduled_sampling_start'],
        scheduled_sampling_end=CONFIG['scheduled_sampling_end'],
        scheduled_sampling_anneal_epochs=CONFIG['scheduled_sampling_anneal_epochs'],
        resume_checkpoint=CONFIG['resume_from'],
        load_optimizer=True,
        stage_name="progressive",
        stage_milestones=stage_milestones
    )
    trainer.train()
    
    print("\n[INFO] Training completed!")
    print(f"[INFO] Final best model: {trainer.best_model_path}")


if __name__ == '__main__':
    main()
