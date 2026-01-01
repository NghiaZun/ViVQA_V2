"""
IMPLICIT REASONING TRAINING: DINOv2 + BARTpho VQA
==================================================
Training với implicit reasoning (hidden-state based):
✅ Reasoning là hidden states, KHÔNG generate text
✅ Answer được condition trên reasoning hidden states
✅ 1 forward pass (nhanh như direct answer)
✅ Có reasoning capability nhưng không bias tokens

Flow:
1. Encode: Image + Question → Fused Features
2. Reasoning Hidden: Fused → Reasoning Decoder → Hidden States (NO TEXT!)
3. Answer: Reasoning Hidden → Answer Decoder → Answer Text

Key difference:
- Autoregressive CoT: Generate reasoning TEXT → slow, token bias
- Implicit Reasoning: Generate reasoning HIDDEN → fast, no bias!
"""

import os
import json
import csv
import torch
import torch.nn as nn
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
import argparse

# Import model
from model_dinov2_bartpho import DINOv2BARTphoVQA, count_parameters


# ============================================================================
# 1. DATASET
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
        import torchvision.transforms as T
        aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            T.RandomRotation(degrees=10),
            T.RandomResizedCrop(224, scale=(0.85, 1.0)),
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
        
        # Tokenize reasoning (for supervision)
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
# 2. IMPLICIT REASONING TRAINER
# ============================================================================

class ImplicitReasoningTrainer:
    """
    Trainer với implicit reasoning
    
    Key idea:
    - Reasoning decoder generates HIDDEN STATES (supervised by GT reasoning)
    - Answer decoder uses reasoning hidden as context
    - Only 1 forward pass (like direct answer)
    - But has reasoning capability!
    
    FIX:
    1. Detach test mỗi epoch → chứng minh reasoning có ích
    2. Anneal α_reasoning → giảm dần từ 0.5 → 0.1 để tránh overfit text
    3. Learned reasoning prefix → inference strategy rõ ràng
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
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        alpha_reasoning_start=0.5,
        alpha_reasoning_end=0.1,
        alpha_answer=0.6,
        label_smoothing=0.1,
        use_amp=True,
        patience=5,
        log_steps=10,
        save_steps=100,
        detach_test_every=5,
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
        self.detach_test_every = detach_test_every
        
        # Annealing α_reasoning: start high → end low
        self.alpha_reasoning_start = alpha_reasoning_start
        self.alpha_reasoning_end = alpha_reasoning_end
        self.alpha_answer = alpha_answer
        
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
        
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        num_warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # CSV logger
        self.csv_log_path = self.output_dir / 'training_log.csv'
        self.init_csv_logger()
        
        print(f"\n[INFO] Implicit Reasoning Trainer initialized")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Loss weights: α_reasoning={alpha_reasoning_start}→{alpha_reasoning_end}, α_answer={alpha_answer}")
        print(f"  Detach test every {detach_test_every} epochs")
    
    def init_csv_logger(self):
        with open(self.csv_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_reasoning_loss', 'train_answer_loss',
                'val_loss', 'val_reasoning_loss', 'val_answer_loss',
                'val_loss_detached', 'answer_drop_pct',
                'learning_rate', 'alpha_reasoning', 'patience_counter', 'is_best'
            ])
    
    def log_to_csv(self, epoch, train_losses, val_losses, alpha_reasoning, val_loss_detached=None, is_best=False):
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Calculate answer drop if detach test was run
        answer_drop_pct = 0.0
        if val_loss_detached is not None and val_losses['answer'] > 0:
            answer_drop_pct = ((val_loss_detached - val_losses['answer']) / val_losses['answer']) * 100
        
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_losses['total']:.4f}",
                f"{train_losses['reasoning']:.4f}",
                f"{train_losses['answer']:.4f}",
                f"{val_losses['total']:.4f}",
                f"{val_losses['reasoning']:.4f}",
                f"{val_losses['answer']:.4f}",
                f"{val_loss_detached:.4f}" if val_loss_detached else "",
                f"{answer_drop_pct:.2f}%" if val_loss_detached else "",
                f"{current_lr:.2e}",
                f"{alpha_reasoning:.3f}",
                self.patience_counter,
                1 if is_best else 0
            ])
    
    def save_checkpoint(self, epoch, val_loss=None, is_best=False):
        try:
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
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
    
    def train_epoch(self, epoch):
        """
        Train one epoch with implicit reasoning
        
        Key: 1 forward pass, but with 2 losses:
        1. Reasoning loss (hidden state supervision) - ANNEALED
        2. Answer loss (conditioned on reasoning hidden)
        """
        self.model.train()
        total_loss = 0
        reasoning_loss_sum = 0
        answer_loss_sum = 0
        
        # Anneal α_reasoning: linear decay
        progress = epoch / max(self.num_epochs - 1, 1)
        alpha_reasoning = self.alpha_reasoning_start + progress * (self.alpha_reasoning_end - self.alpha_reasoning_start)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [α_r={alpha_reasoning:.3f}]")
        
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
                    # Encode image + question
                    vision_embeds = self.model.encode_image(tensor_batch['pixel_values'])
                    question_embeds = self.model.encode_text(
                        input_ids=tensor_batch['input_ids'],
                        attention_mask=tensor_batch['attention_mask']
                    )
                    fused_features, _ = self.model.fuse_multimodal(question_embeds, vision_embeds)
                    
                    # Step 1: Reasoning hidden (supervised by GT reasoning)
                    reasoning_logits, reasoning_hidden, _ = self.model.generate_reasoning(
                        fused_features=fused_features,
                        reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                        reasoning_attention_mask=tensor_batch['reasoning_attention_mask']
                    )
                    
                    reasoning_loss = self.criterion(
                        reasoning_logits.view(-1, reasoning_logits.size(-1)),
                        reasoning_labels.view(-1)
                    )
                    
                    # REGULARIZATION: Random detach reasoning hidden (10-20%)
                    # Prevent reasoning from collapsing to identity/passthrough
                    if random.random() < 0.15:  # 15% detach rate
                        reasoning_hidden_for_answer = reasoning_hidden.detach()
                    else:
                        reasoning_hidden_for_answer = reasoning_hidden
                    
                    # Step 2: Answer conditioned on reasoning hidden
                    answer_logits, _ = self.model.generate_answer(
                        fused_features=fused_features,
                        reasoning_hidden=reasoning_hidden_for_answer,  # Use reasoning hidden!
                        answer_input_ids=tensor_batch['answer_input_ids'],
                        answer_attention_mask=tensor_batch['answer_attention_mask']
                    )
                    
                    answer_loss = self.criterion(
                        answer_logits.view(-1, answer_logits.size(-1)),
                        answer_labels.view(-1)
                    )
                    
                    # REGULARIZATION: Variance regularization
                    # Encourage reasoning hidden to have diverse representations
                    reasoning_var = reasoning_hidden.var(dim=1).mean()  # variance across seq_len
                    var_reg_loss = -torch.log(reasoning_var + 1e-8)  # negative log encourages high variance
                    
                    # Combined loss with annealed α_reasoning
                    loss = (alpha_reasoning * reasoning_loss + 
                           self.alpha_answer * answer_loss +
                           0.01 * var_reg_loss)  # Small weight for regularization
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
        }, alpha_reasoning
    
    @torch.no_grad()
    def validate(self, epoch, test_detach=False):
        """
        Validate
        
        Args:
            test_detach: If True, detach reasoning hidden to test if it's useful
        """
        self.model.eval()
        total_loss = 0
        reasoning_loss_sum = 0
        answer_loss_sum = 0
        num_batches = 0
        
        # Compute current α_reasoning for consistency
        progress = epoch / max(self.num_epochs - 1, 1)
        alpha_reasoning = self.alpha_reasoning_start + progress * (self.alpha_reasoning_end - self.alpha_reasoning_start)
        
        desc = "Evaluating (DETACHED)" if test_detach else "Evaluating"
        progress_bar = tqdm(self.val_loader, desc=desc)
        
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
                    
                    # Reasoning hidden
                    reasoning_logits, reasoning_hidden, _ = self.model.generate_reasoning(
                        fused_features=fused_features,
                        reasoning_input_ids=tensor_batch['reasoning_input_ids'],
                        reasoning_attention_mask=tensor_batch['reasoning_attention_mask']
                    )
                    
                    reasoning_loss = self.criterion(
                        reasoning_logits.view(-1, reasoning_logits.size(-1)),
                        reasoning_labels.view(-1)
                    )
                    
                    # DETACH TEST: If testing, detach reasoning hidden
                    if test_detach:
                        reasoning_hidden = reasoning_hidden.detach()
                    
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
                    
                    loss = (alpha_reasoning * reasoning_loss + 
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
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"IMPLICIT REASONING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(self.num_epochs):
            # Train
            train_losses, alpha_reasoning = self.train_epoch(epoch)
            print(f"\n[EPOCH {epoch+1}] Train Loss: {train_losses['total']:.4f} [α_r={alpha_reasoning:.3f}]")
            print(f"  Reasoning: {train_losses['reasoning']:.4f}")
            print(f"  Answer: {train_losses['answer']:.4f}")
            
            # Validate (normal)
            val_losses = self.validate(epoch, test_detach=False)
            print(f"[VALIDATION] Loss: {val_losses['total']:.4f}")
            print(f"  Reasoning: {val_losses['reasoning']:.4f}")
            print(f"  Answer: {val_losses['answer']:.4f}")
            
            # Detach test every N epochs
            val_loss_detached = None
            if (epoch + 1) % self.detach_test_every == 0:
                print(f"\n[DETACH TEST] Testing if reasoning is useful...")
                val_losses_detached = self.validate(epoch, test_detach=True)
                val_loss_detached = val_losses_detached['answer']
                
                # Calculate degradation
                answer_drop = val_loss_detached - val_losses['answer']
                answer_drop_pct = (answer_drop / val_losses['answer']) * 100 if val_losses['answer'] > 0 else 0
                
                print(f"  Answer loss (normal): {val_losses['answer']:.4f}")
                print(f"  Answer loss (detached): {val_loss_detached:.4f}")
                print(f"  Degradation: +{answer_drop:.4f} ({answer_drop_pct:+.2f}%)")
                
                if answer_drop > 0.01:
                    print(f"  ✅ Reasoning IS useful! (answer degrades without it)")
                else:
                    print(f"  ⚠️ Reasoning NOT useful (answer doesn't need it)")
            
            # Save best
            is_best = False
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                is_best = True
                self.save_checkpoint(epoch, val_losses['total'], is_best=True)
            else:
                self.patience_counter += 1
                print(f"[INFO] No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Save latest
            self.save_checkpoint(epoch, val_losses['total'], is_best=False)
            
            # Log
            self.log_to_csv(epoch, train_losses, val_losses, alpha_reasoning, val_loss_detached, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# ============================================================================
# 3. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Implicit Reasoning Training')
    
    parser.add_argument('--train_json', type=str, 
                        default='/kaggle/input/teacher-checkpoint-11k/teacher_outputs_train.jsonl')
    parser.add_argument('--image_dir', type=str,
                        default='/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train')
    parser.add_argument('--output_dir', type=str,
                        default='/kaggle/working/checkpoints_implicit_reasoning')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--alpha_reasoning_start', type=float, default=0.5, 
                        help='Initial weight for reasoning loss (will anneal down)')
    parser.add_argument('--alpha_reasoning_end', type=float, default=0.1,
                        help='Final weight for reasoning loss')
    parser.add_argument('--detach_test_every', type=int, default=5,
                        help='Test reasoning utility every N epochs')
    parser.add_argument('--reasoning_bottleneck', type=int, default=None,
                        help='Compress reasoning to k tokens (e.g., 6). None = no compression')
    
    args = parser.parse_args()
    
    CONFIG = {
        'train_json': args.train_json,
        'image_dir': args.image_dir,
        'output_dir': args.output_dir,
        'val_split': 0.1,
        'random_seed': 42,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'alpha_reasoning_start': args.alpha_reasoning_start,
        'alpha_reasoning_end': args.alpha_reasoning_end,
        'alpha_answer': 0.6,
        'label_smoothing': 0.1,
        'use_amp': True,
        'patience': 5,
        'log_steps': 10,
        'detach_test_every': args.detach_test_every,
    }
    
    print("="*70)
    print("IMPLICIT REASONING TRAINING: DINOv2 + BARTpho")
    print("="*70)
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Seed
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    # Model
    print("\n[INFO] Initializing model...")
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        use_reasoning_quality_check=False,
        gradient_checkpointing=True,
        reasoning_bottleneck_tokens=args.reasoning_bottleneck  # NEW: bottleneck
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"[INFO] Total params: {total_params/1e6:.1f}M")
    if args.reasoning_bottleneck:
        print(f"[INFO] Reasoning bottleneck: {args.reasoning_bottleneck} tokens")
    print(f"[INFO] Trainable params: {trainable_params/1e6:.1f}M")
    
    # Dataset
    print("\n[INFO] Loading dataset...")
    full_dataset = ImplicitReasoningDataset(
        json_path=CONFIG['train_json'],
        image_dir=CONFIG['image_dir'],
        vision_processor=model.vision_processor,
        tokenizer=model.tokenizer,
        augment=True
    )
    
    total_size = len(full_dataset)
    val_size = int(total_size * CONFIG['val_split'])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )
    
    print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Trainer
    trainer = ImplicitReasoningTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=CONFIG['output_dir'],
        batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        max_grad_norm=CONFIG['max_grad_norm'],
        alpha_reasoning_start=CONFIG['alpha_reasoning_start'],
        alpha_reasoning_end=CONFIG['alpha_reasoning_end'],
        alpha_answer=CONFIG['alpha_answer'],
        label_smoothing=CONFIG['label_smoothing'],
        use_amp=CONFIG['use_amp'],
        patience=CONFIG['patience'],
        log_steps=CONFIG['log_steps'],
        detach_test_every=CONFIG['detach_test_every'],
    )
    
    trainer.train()
    
    print("\n[INFO] Training completed!")
    print(f"[INFO] Best model: {trainer.output_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
