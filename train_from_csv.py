"""
Training Script cho DINOv2-BARTpho VQA từ CSV
==============================================
Train model từ CSV file với columns: stt, question, answer, img_id, type

🔥 KEY FEATURES:
================
1. Resume Training:
   - Sử dụng --resume để load checkpoint và continue training
   - Tự động restore: model, optimizer, scheduler, training state
   - Resume từ đúng stage và epoch bị gián đoạn

2. Error Handling:
   - Catch OOM errors → Save checkpoint + Recommendations
   - Catch KeyboardInterrupt (Ctrl+C) → Save checkpoint
   - Catch unexpected errors → Save checkpoint + Traceback
   - Always save "last_checkpoint" để có thể resume

3. 3-Stage Progressive Training:
   - Stage 1: Fusion only (~20M params)
   - Stage 2: + Answer decoder + LM head (~220M params)
   - Stage 3: + Encoder last 3 layers (~260M params)
   - Checkpoint sau mỗi epoch + khi bị gián đoạn

Usage:
======
# First run:
python train_from_csv.py \
    --csv_path data.csv \
    --image_folder /path/to/images \
    --output_dir ./outputs \
    --batch_size 8 \
    --stage1_epochs 3 \
    --stage2_epochs 3 \
    --stage3_epochs 4

# Nếu bị gián đoạn (OOM, Ctrl+C, crash), resume bằng:
python train_from_csv.py \
    --csv_path data.csv \
    --image_folder /path/to/images \
    --output_dir ./outputs \
    --resume ./outputs/last_checkpoint \
    --batch_size 4 \
    --gradient_accumulation_steps 8

# Script sẽ:
# - Load checkpoint từ path chỉ định
# - Restore model + optimizer + scheduler + training state
# - Resume từ đúng stage và epoch
# - Continue training như bình thường

Advanced:
=========
# Resume from specific checkpoint (không phải last_checkpoint):
python train_from_csv.py \
    --resume ./outputs/stage2_epoch3 \
    ...
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
import logging

from model_dinov2_bartpho_2 import DINOv2BARTphoVQA


# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET
# ============================================================================
class ViVQADataset(Dataset):
    """
    Dataset cho ViVQA từ CSV
    
    CSV format:
        stt, question, answer, img_id, type
        0, "màu của chiếc bình là gì", "màu xanh lá", 68857, 2
    """
    
    def __init__(
        self, 
        csv_path, 
        image_folder, 
        model,
        max_question_len=64,  # 🔥 Giảm từ 128 → 64 (Vietnamese questions ngắn)
        max_answer_len=16,  # 🔥 Short answer (3-5 từ)
        image_ext='.jpg'
    ):
        """
        Args:
            csv_path: Path to CSV file
            image_folder: Folder chứa images
            model: DINOv2BARTphoVQA model (để lấy tokenizer và processor)
            max_question_len: Max length cho question
            max_answer_len: Max length cho answer
            image_ext: Extension của image files (.jpg, .png, etc.)
        """
        self.df = pd.read_csv(csv_path)
        self.image_folder = Path(image_folder)
        self.tokenizer = model.tokenizer
        self.vision_processor = model.vision_processor
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.image_ext = image_ext
        
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        logger.info(f"Image folder: {image_folder}")
        
        # Validate image files
        self._validate_images()
    
    def _validate_images(self):
        """Check if all images exist"""
        missing = []
        for idx, row in self.df.iterrows():
            img_path = self.image_folder / f"{row['img_id']}{self.image_ext}"
            if not img_path.exists():
                missing.append(str(img_path))
        
        if missing:
            logger.warning(f"Missing {len(missing)} images:")
            for path in missing[:5]:
                logger.warning(f"  - {path}")
            if len(missing) > 5:
                logger.warning(f"  ... and {len(missing) - 5} more")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_folder / f"{row['img_id']}{self.image_ext}"
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return black image if error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process image
        pixel_values = self.vision_processor(
            images=image, 
            return_tensors='pt'
        )['pixel_values'].squeeze(0)  # [3, 224, 224]
        
        # Tokenize question
        question_encoding = self.tokenizer(
            row['question'],
            max_length=self.max_question_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer
        answer_encoding = self.tokenizer(
            row['answer'],
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': question_encoding['input_ids'].squeeze(0),
            'attention_mask': question_encoding['attention_mask'].squeeze(0),
            'answer_input_ids': answer_encoding['input_ids'].squeeze(0),
            'answer_attention_mask': answer_encoding['attention_mask'].squeeze(0),
            'question_text': row['question'],
            'answer_text': row['answer'],
            'img_id': row['img_id'],
            'type': row['type']
        }


# ============================================================================
# TRAINER
# ============================================================================
class VQATrainer:
    """
    Trainer cho VQA model với 3-stage progressive unfreezing:
    
    Stage 1: Freeze all pretrained, train ONLY fusion (projection + cross-attention)
    Stage 2: Unfreeze Answer Decoder + LM head (language generation adaptation)
    Stage 3: Unfreeze Encoder last 3 layers (Vietnamese semantic adaptation)
    
    Note: Vision encoder ALWAYS frozen (11K samples không đủ, DINOv2 đã tốt)
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        output_dir,
        learning_rate=2e-5,
        num_epochs_per_stage=(3, 3, 4),  # 🔥 Epochs cho mỗi stage
        warmup_steps=500,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        save_steps=1000,
        eval_steps=500,
        device='cuda',
        resume_checkpoint=None  # 🔥 Path to checkpoint để resume
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.num_epochs_per_stage = num_epochs_per_stage
        self.total_epochs = sum(num_epochs_per_stage)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        # Optimizer (sẽ được tạo lại cho mỗi stage)
        self.optimizer = None
        self.scheduler = None
        self.learning_rate = learning_rate
        
        # Scheduler params
        self.warmup_steps = warmup_steps
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
        
        # Metrics
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.current_stage = 1
        self.current_epoch_in_stage = 0
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Stage 1 epochs: {num_epochs_per_stage[0]} (fusion only)")
        logger.info(f"  - Stage 2 epochs: {num_epochs_per_stage[1]} (+ answer decoder + LM head)")
        logger.info(f"  - Stage 3 epochs: {num_epochs_per_stage[2]} (+ encoder last 3 layers)")
        logger.info(f"  - Total epochs: {self.total_epochs}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
        logger.info(f"  - Vision encoder: ALWAYS FROZEN (11K samples insufficient)")
        
        # 🔥 Load checkpoint nếu có --resume
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
    
    def setup_stage(self, stage):
        """
        Setup freezing/unfreezing cho từng stage
        
        Stage 1: Train ONLY fusion (projection + cross-attention)
        Stage 2: + Unfreeze Answer Decoder + LM head (language generation)
        Stage 3: + Unfreeze Encoder last 3 layers (Vietnamese semantics)
        
        Note: Vision encoder ALWAYS frozen (11K samples insufficient)
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"SETTING UP STAGE {stage}")
        logger.info("=" * 80)
        
        if stage == 1:
            # Stage 1: Freeze ALL pretrained, train ONLY fusion
            logger.info("[Stage 1] Freeze all pretrained, train ONLY fusion")
            logger.info("  Goal: Learn vision-language alignment")
            
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze fusion components
            for param in self.model.vision_proj.parameters():
                param.requires_grad = True
            for param in self.model.cross_attention_fusion.parameters():
                param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"  ✅ Trainable: {trainable/1e6:.1f}M params (fusion only)")
            
        elif stage == 2:
            # Stage 2: + Unfreeze Answer Decoder + LM head
            logger.info("[Stage 2] + Unfreeze Answer Decoder + LM head")
            logger.info("  Goal: Adapt Vietnamese VQA answer generation")
            
            # Unfreeze answer decoder (199M params)
            for param in self.model.answer_decoder.parameters():
                param.requires_grad = True
            
            # Unfreeze LM head (vocabulary projection)
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"  ✅ Trainable: {trainable/1e6:.1f}M params (fusion + answer decoder + LM head)")
            
        elif stage == 3:
            # Stage 3: + Unfreeze Encoder last 3 layers
            logger.info("[Stage 3] + Unfreeze Encoder last 3 layers")
            logger.info("  Goal: Fine-tune Vietnamese semantic understanding")
            
            # Unfreeze last 3 layers của encoder
            total_layers = len(self.model.encoder.layers)
            for i, layer in enumerate(self.model.encoder.layers):
                if i >= total_layers - 3:  # Last 3 layers
                    for param in layer.parameters():
                        param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"  ✅ Trainable: {trainable/1e6:.1f}M params (fusion + decoder + LM head + encoder last 3 layers)")
        
        # Create optimizer cho stage này
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Create scheduler
        num_epochs_this_stage = self.num_epochs_per_stage[stage - 1]
        total_steps = len(self.train_loader) * num_epochs_this_stage // self.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"  ✅ Optimizer & scheduler created for {num_epochs_this_stage} epochs")
        logger.info("=" * 80 + "\n")
    
    def train(self):
        """
        🔥 Main training loop với 3 stages + Error handling + Auto-resume
        """
        logger.info("=" * 80)
        logger.info("Starting 3-Stage Training...")
        logger.info("=" * 80)
        
        try:
            # Stage 1: Fusion only
            if self.current_stage <= 1:
                logger.info(f"\n🔥 STAGE 1: FUSION ONLY")
                if self.current_stage < 1 or self.current_epoch_in_stage == 0:
                    self.setup_stage(1)
                    self.current_stage = 1
                
                for epoch in range(self.current_epoch_in_stage, self.num_epochs_per_stage[0]):
                    self.current_epoch_in_stage = epoch
                    logger.info(f"\n[Stage 1] Epoch {epoch + 1}/{self.num_epochs_per_stage[0]}")
                    
                    self.train_epoch(epoch + 1)
                    val_loss, _ = self.validate()
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model_stage1')
                        logger.info(f"✓ Best Stage 1 model saved! (val_loss={val_loss:.4f})")
                    
                    # 🔥 Save last_checkpoint sau mỗi epoch (để có thể resume)
                    self.save_checkpoint('last_checkpoint')
                    # Save checkpoint cụ thể
                    self.save_checkpoint(f'stage1_epoch{epoch + 1}')
                
                self.save_checkpoint('stage1_final')
                self.current_stage = 2
                self.current_epoch_in_stage = 0
            
            # Stage 2: + Answer Decoder + LM head
            if self.current_stage <= 2:
                logger.info(f"\n🔥 STAGE 2: + ANSWER DECODER + LM HEAD")
                if self.current_epoch_in_stage == 0:
                    self.setup_stage(2)
                
                for epoch in range(self.current_epoch_in_stage, self.num_epochs_per_stage[1]):
                    self.current_epoch_in_stage = epoch
                    logger.info(f"\n[Stage 2] Epoch {epoch + 1}/{self.num_epochs_per_stage[1]}")
                    
                    self.train_epoch(epoch + 1)
                    val_loss, _ = self.validate()
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model_stage2')
                        logger.info(f"✓ Best Stage 2 model saved! (val_loss={val_loss:.4f})")
                    
                    # 🔥 Save last_checkpoint sau mỗi epoch
                    self.save_checkpoint('last_checkpoint')
                    self.save_checkpoint(f'stage2_epoch{epoch + 1}')
                
                self.save_checkpoint('stage2_final')
                self.current_stage = 3
                self.current_epoch_in_stage = 0
            
            # Stage 3: + Encoder last 3 layers
            if self.current_stage <= 3:
                logger.info(f"\n🔥 STAGE 3: + ENCODER LAST 3 LAYERS")
                if self.current_epoch_in_stage == 0:
                    self.setup_stage(3)
                
                for epoch in range(self.current_epoch_in_stage, self.num_epochs_per_stage[2]):
                    self.current_epoch_in_stage = epoch
                    logger.info(f"\n[Stage 3] Epoch {epoch + 1}/{self.num_epochs_per_stage[2]}")
                    
                    self.train_epoch(epoch + 1)
                    val_loss, _ = self.validate()
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model_final')
                        logger.info(f"✓ Best final model saved! (val_loss={val_loss:.4f})")
                    
                    # 🔥 Save last_checkpoint sau mỗi epoch
                    self.save_checkpoint('last_checkpoint')
                    self.save_checkpoint(f'stage3_epoch{epoch + 1}')
                
                self.save_checkpoint('stage3_final')
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ 3-Stage Training completed!")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            logger.info("=" * 80)
            
        except KeyboardInterrupt:
            logger.warning("\n" + "=" * 80)
            logger.warning("⚠️  Training interrupted by user (Ctrl+C)")
            logger.warning("=" * 80)
            self._save_interrupted_checkpoint()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("\n" + "=" * 80)
                logger.error("❌ OUT OF MEMORY ERROR")
                logger.error("=" * 80)
                logger.error(f"Error: {e}")
                logger.error("\nRecommendations:")
                logger.error("  1. Reduce --batch_size (try batch_size=2 or 1)")
                logger.error("  2. Increase --gradient_accumulation_steps (try 16 or 32)")
                logger.error("  3. Check GPU memory: nvidia-smi")
                self._save_interrupted_checkpoint()
            else:
                logger.error(f"\n❌ Runtime Error: {e}")
                self._save_interrupted_checkpoint()
                raise
            
        except Exception as e:
            logger.error("\n" + "=" * 80)
            logger.error(f"❌ UNEXPECTED ERROR: {e}")
            logger.error("=" * 80)
            import traceback
            logger.error(traceback.format_exc())
            self._save_interrupted_checkpoint()
            raise
    
    def _save_interrupted_checkpoint(self):
        """
        🔥 Save checkpoint khi training bị gián đoạn
        Luôn save vào "last_checkpoint" để dễ resume
        """
        try:
            # Save vào last_checkpoint để dễ tìm
            self.save_checkpoint('last_checkpoint')
            
            # Save thêm checkpoint với tên cụ thể
            checkpoint_name = f'interrupted_stage{self.current_stage}_epoch{self.current_epoch_in_stage}'
            self.save_checkpoint(checkpoint_name)
            
            logger.warning(f"\n{'='*80}")
            logger.warning(f"✓ Saved checkpoint for resume:")
            logger.warning(f"  - {self.output_dir}/last_checkpoint")
            logger.warning(f"  - {self.output_dir}/{checkpoint_name}")
            logger.warning(f"{'='*80}")
            logger.warning(f"\n📌 To resume training, use:")
            logger.warning(f"   --resume {self.output_dir}/last_checkpoint")
            logger.warning(f"\n   Training will continue from Stage {self.current_stage}, Epoch {self.current_epoch_in_stage + 1}")
            logger.warning(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"❌ Could not save interrupted checkpoint: {e}")
    
    def train_epoch(self, global_epoch):
        """Train one epoch với error handling"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {global_epoch}")
        for step, batch in enumerate(pbar):
            try:
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                answer_input_ids = batch['answer_input_ids'].to(self.device)
                answer_attention_mask = batch['answer_attention_mask'].to(self.device)
                
                # 🔥 Clear cache every 50 steps
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                
                # Forward pass với mixed precision
                with autocast():
                    # Encode image + question
                    visual_features = self.model.encode_image(pixel_values)
                    text_features = self.model.encode_text(input_ids, attention_mask)
                    fused_features, _ = self.model.fuse_multimodal(text_features, visual_features)
                    
                    # Generate answer (NO reasoning - direct answer)
                    # Use dummy reasoning_hidden = fused_features
                    answer_logits, _ = self.model.generate_answer(
                        fused_features=fused_features,
                        reasoning_hidden=fused_features,  # Direct: no reasoning stage
                        answer_input_ids=answer_input_ids,
                        answer_attention_mask=answer_attention_mask,
                        use_reasoning_only=False  # Use fused_features directly
                    )
                    
                    # Compute loss
                    loss = self.criterion(
                        answer_logits.view(-1, answer_logits.size(-1)),
                        answer_input_ids.view(-1)
                    )
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward
                self.scaler.scale(loss).backward()
                
                # 🔥 Delete intermediate tensors to free memory
                del visual_features, text_features, fused_features, answer_logits
                
                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
                
                # Eval
                if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    val_loss, _ = self.validate()
                    logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}")
                    self.model.train()
                
                # Save checkpoint
                if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"\n❌ OOM at step {step}/{len(self.train_loader)}")
                    logger.error("Skipping this batch and clearing cache...")
                    torch.cuda.empty_cache()
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad()
                    continue
                else:
                    raise
        
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {global_epoch} - Avg Train Loss: {avg_loss:.4f}")
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        # 🔥 Clear cache before validation
        torch.cuda.empty_cache()
        
        pbar = tqdm(self.val_loader, desc="Validating")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answer_input_ids = batch['answer_input_ids'].to(self.device)
            answer_attention_mask = batch['answer_attention_mask'].to(self.device)
            
            # Forward
            visual_features = self.model.encode_image(pixel_values)
            text_features = self.model.encode_text(input_ids, attention_mask)
            fused_features, _ = self.model.fuse_multimodal(text_features, visual_features)
            
            answer_logits, _ = self.model.generate_answer(
                fused_features=fused_features,
                reasoning_hidden=fused_features,
                answer_input_ids=answer_input_ids,
                answer_attention_mask=answer_attention_mask,
                use_reasoning_only=False
            )
            
            loss = self.criterion(
                answer_logits.view(-1, answer_logits.size(-1)),
                answer_input_ids.view(-1)
            )
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'batches': num_batches})
            
            # 🔥 Delete intermediate tensors
            del visual_features, text_features, fused_features, answer_logits, loss
        
        avg_loss = total_loss / num_batches
        logger.info(f"✓ Validation completed: {num_batches} batches, avg_loss={avg_loss:.4f}")
        return avg_loss, {}
    
    def save_checkpoint(self, name):
        """
        Save checkpoint với training state để có thể resume
        """
        try:
            checkpoint_dir = self.output_dir / name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            torch.save(self.model.state_dict(), checkpoint_dir / 'model.pt')
            
            # 🔥 Save training state để có thể resume
            training_state = {
                'global_step': self.global_step,
                'best_val_loss': self.best_val_loss,
                'current_stage': self.current_stage,
                'current_epoch_in_stage': self.current_epoch_in_stage,
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict(),
            }
            torch.save(training_state, checkpoint_dir / 'training_state.pt')
            
            logger.info(f"✓ Checkpoint saved to {checkpoint_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint {name}: {e}")
            logger.warning("⚠️  Training will continue but checkpoint not saved")
    
    def load_checkpoint(self, checkpoint_path):
        """
        🔥 Load checkpoint từ path được chỉ định
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            logger.warning("⚠️  Starting training from scratch")
            return
        
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 LOADING CHECKPOINT: {checkpoint_path}")
            logger.info(f"{'='*80}")
            
            # Load model
            model_path = checkpoint_path / 'model.pt'
            if not model_path.exists():
                logger.error(f"❌ Model file not found: {model_path}")
                return
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"✓ Loaded model from {model_path}")
            
            # Load training state
            state_path = checkpoint_path / 'training_state.pt'
            if not state_path.exists():
                logger.warning(f"⚠️  Training state not found: {state_path}")
                logger.warning("⚠️  Will use model only, starting from Stage 1")
                return
            
            state = torch.load(state_path, map_location=self.device)
            
            self.global_step = state.get('global_step', 0)
            self.best_val_loss = state.get('best_val_loss', float('inf'))
            self.current_stage = state.get('current_stage', 1)
            self.current_epoch_in_stage = state.get('current_epoch_in_stage', 0)
            
            logger.info(f"✓ Loaded training state:")
            logger.info(f"  - Global step: {self.global_step}")
            logger.info(f"  - Best val loss: {self.best_val_loss:.4f}")
            logger.info(f"  - Current stage: {self.current_stage}")
            logger.info(f"  - Epoch in stage: {self.current_epoch_in_stage}")
            
            # Setup stage trước khi load optimizer/scheduler
            self.setup_stage(self.current_stage)
            
            # Load optimizer/scheduler nếu có
            if state.get('optimizer') and self.optimizer:
                try:
                    self.optimizer.load_state_dict(state['optimizer'])
                    logger.info(f"✓ Loaded optimizer state")
                except Exception as e:
                    logger.warning(f"⚠️  Could not load optimizer state: {e}")
            
            if state.get('scheduler') and self.scheduler:
                try:
                    self.scheduler.load_state_dict(state['scheduler'])
                    logger.info(f"✓ Loaded scheduler state")
                except Exception as e:
                    logger.warning(f"⚠️  Could not load scheduler state: {e}")
            
            if state.get('scaler'):
                try:
                    self.scaler.load_state_dict(state['scaler'])
                    logger.info(f"✓ Loaded scaler state")
                except Exception as e:
                    logger.warning(f"⚠️  Could not load scaler state: {e}")
            
            logger.info(f"{'='*80}")
            logger.info(f"✅ RESUME FROM CHECKPOINT SUCCESSFUL")
            logger.info(f"   Will continue from Stage {self.current_stage}, Epoch {self.current_epoch_in_stage + 1}")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("⚠️  Starting training from scratch")
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.current_stage = 1
            self.current_epoch_in_stage = 0


# ============================================================================
# INFERENCE
# ============================================================================
@torch.no_grad()
def generate_answers(
    model,
    csv_path,
    image_folder,
    output_path,
    batch_size=8,
    max_answer_len=16,  # 🔥 Giảm xuống 16 cho short answer
    num_beams=1,  # 🔥 Greedy cho answer ngắn
    device='cuda'
):
    """
    Generate answers cho toàn bộ dataset và save ra CSV
    
    Args:
        model: Trained model
        csv_path: Input CSV
        image_folder: Folder chứa images
        output_path: Output CSV path
        batch_size: Batch size
        max_answer_len: Max answer length
        num_beams: Beam search width
        device: Device
    """
    logger.info("=" * 80)
    logger.info("Generating answers...")
    logger.info("=" * 80)
    
    model.eval()
    model = model.to(device)
    
    # Create dataset (no answer needed for inference)
    dataset = ViVQADataset(
        csv_path=csv_path,
        image_folder=image_folder,
        model=model
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # 🔥 Giảm workers
        pin_memory=True,
        prefetch_factor=2
    )
    
    results = []
    
    pbar = tqdm(loader, desc="Generating")
    for batch in pbar:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Encode
        visual_features = model.encode_image(pixel_values)
        text_features = model.encode_text(input_ids, attention_mask)
        fused_features, _ = model.fuse_multimodal(text_features, visual_features)
        
        # Generate answer (no reasoning stage)
        batch_size_curr = pixel_values.size(0)
        answer_ids = torch.full(
            (batch_size_curr, 1),
            model.tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 🔥 FIX: Thêm repetition penalty và early stopping
        for step in range(max_answer_len - 1):
            decoder_outputs = model.answer_decoder(
                input_ids=answer_ids,
                encoder_hidden_states=fused_features,  # Direct encoding
                return_dict=True,
                use_cache=False
            )
            
            hidden = decoder_outputs.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)
            
            # 🔥 Apply repetition penalty
            for i in range(batch_size_curr):
                for token_id in set(answer_ids[i].tolist()):
                    logits[i, token_id] /= 1.5  # Penalty = 1.5
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            answer_ids = torch.cat([answer_ids, next_token], dim=1)
            
            # 🔥 Early stop if ALL sequences hit EOS
            if (next_token == model.tokenizer.eos_token_id).all():
                break
        
        # 🔥 FIX: Decode và clean output
        generated_answers = []
        for i in range(batch_size_curr):
            tokens = answer_ids[i].tolist()
            
            # Remove BOS token (đầu tiên)
            if tokens and tokens[0] == model.tokenizer.bos_token_id:
                tokens = tokens[1:]
            
            # Remove EOS và padding
            if model.tokenizer.eos_token_id in tokens:
                eos_idx = tokens.index(model.tokenizer.eos_token_id)
                tokens = tokens[:eos_idx]
            
            # Decode
            text = model.tokenizer.decode(tokens, skip_special_tokens=True)
            # Clean whitespace
            text = text.strip()
            generated_answers.append(text)
        
        # Collect results
        for i in range(len(generated_answers)):
            results.append({
                'stt': len(results),
                'question': batch['question_text'][i],
                'answer_gt': batch['answer_text'][i],
                'answer_pred': generated_answers[i],
                'img_id': batch['img_id'][i].item(),
                'type': batch['type'][i].item()
            })
    
    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"✓ Results saved to {output_path}")
    logger.info(f"  Total samples: {len(results)}")
    
    # Preview
    logger.info("\nPreview (first 3 samples):")
    for i in range(min(3, len(results))):
        logger.info(f"\n  [{i+1}] Question: {results[i]['question']}")
        logger.info(f"      GT Answer: {results[i]['answer_gt']}")
        logger.info(f"      Pred Answer: {results[i]['answer_pred']}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train DINOv2-BARTpho VQA from CSV')
    
    # Data args
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--image_folder', type=str, required=True, help='Folder containing images')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio (auto split from csv_path)')
    parser.add_argument('--image_ext', type=str, default='.jpg', help='Image file extension')
    
    # Model args
    parser.add_argument('--dinov2_model', type=str, default='facebook/dinov2-base')
    parser.add_argument('--bartpho_model', type=str, default='vinai/bartpho-syllable')
    parser.add_argument('--num_cross_attn_layers', type=int, default=3)
    
    # Training args
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--stage1_epochs', type=int, default=3, help='Stage 1: Fusion only')
    parser.add_argument('--stage2_epochs', type=int, default=3, help='Stage 2: + Answer decoder + LM head')
    parser.add_argument('--stage3_epochs', type=int, default=4, help='Stage 3: + Encoder last 3 layers')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=500)
    
    # 🔥 Resume training
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint folder để resume training (e.g., ./outputs/last_checkpoint)')
    
    # Generation args
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load for generation')
    parser.add_argument('--max_answer_len', type=int, default=16, help='Max answer length (default 16 for short answers)')
    parser.add_argument('--num_beams', type=int, default=1, help='Beam search width (1=greedy, recommended for short answers)')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    if args.mode == 'train':
        # Initialize model
        logger.info("Initializing model...")
        model = DINOv2BARTphoVQA(
            dinov2_model_name=args.dinov2_model,
            bartpho_model_name=args.bartpho_model,
            num_cross_attn_layers=args.num_cross_attn_layers,
            gradient_checkpointing=True
        )
        
        # 🔥 ALWAYS freeze pretrained weights cho 3-stage training
        logger.info("Freezing all pretrained weights (will unfreeze progressively)...")
        for param in model.parameters():
            param.requires_grad = False
        
        # Prepare datasets - 🔥 AUTO SPLIT from csv_path
        logger.info("Loading dataset and splitting train/val...")
        full_dataset = ViVQADataset(args.csv_path, args.image_folder, model, image_ext=args.image_ext)
        
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 🔥 Seed for reproducibility
        )
        
        # Dataloaders - 🔥 Giảm workers để save memory
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,  # 🔥 Giảm từ 4 → 2
            pin_memory=True,
            prefetch_factor=2  # 🔥 Giảm prefetch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,  # 🔥 Giảm từ 4 → 2
            pin_memory=True,
            prefetch_factor=2
        )
        
        logger.info(f"✓ Train samples: {len(train_dataset)}")
        logger.info(f"✓ Val samples: {len(val_dataset)}")
        
        # Initialize trainer với 3 stages + resume checkpoint
        trainer = VQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs_per_stage=(args.stage1_epochs, args.stage2_epochs, args.stage3_epochs),
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            device=device,
            resume_checkpoint=args.resume  # 🔥 Pass resume checkpoint
        )
        
        # Train với 3 stages
        trainer.train()
    
    # ========================================================================
    # GENERATION MODE
    # ========================================================================
    elif args.mode == 'generate':
        if not args.checkpoint:
            raise ValueError("Must provide --checkpoint for generation mode")
        
        # Initialize model
        logger.info("Initializing model...")
        model = DINOv2BARTphoVQA(
            dinov2_model_name=args.dinov2_model,
            bartpho_model_name=args.bartpho_model,
            num_cross_attn_layers=args.num_cross_attn_layers
        )
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        
        # Generate answers
        output_path = os.path.join(args.output_dir, 'predictions.csv')
        generate_answers(
            model=model,
            csv_path=args.csv_path,
            image_folder=args.image_folder,
            output_path=output_path,
            batch_size=args.batch_size,
            max_answer_len=args.max_answer_len,
            num_beams=args.num_beams,
            device=device
        )


if __name__ == '__main__':
    main()
