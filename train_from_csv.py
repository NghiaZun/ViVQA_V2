"""
Training Script cho DINOv2-BARTpho VQA từ CSV
==============================================
Train model từ CSV file với columns: stt, question, answer, img_id, type

Usage:
    python train_from_csv.py \
        --csv_path data.csv \
        --image_folder /path/to/images \
        --output_dir ./outputs \
        --batch_size 8 \
        --num_epochs 10
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
        max_question_len=128,
        max_answer_len=16,  # 🔥 Giảm xuống 16 cho short answer (3-5 ký tự)
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
    Trainer cho VQA model với single-stage (chỉ train answer, không có reasoning)
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        output_dir,
        learning_rate=2e-5,
        num_epochs=10,
        warmup_steps=500,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        save_steps=1000,
        eval_steps=500,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        # Optimizer
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
        
        # Metrics
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Num epochs: {num_epochs}")
        logger.info(f"  - Total training steps: {total_steps}")
        logger.info(f"  - Warmup steps: {warmup_steps}")
    
    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate()
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model')
                logger.info(f"✓ New best model saved! (val_loss={val_loss:.4f})")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}')
        
        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answer_input_ids = batch['answer_input_ids'].to(self.device)
            answer_attention_mask = batch['answer_attention_mask'].to(self.device)
            
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
        
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch + 1} - Avg Train Loss: {avg_loss:.4f}")
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
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
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, {}
    
    def save_checkpoint(self, name):
        """Save checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), checkpoint_dir / 'model.pt')
        
        # Save optimizer & scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, checkpoint_dir / 'training_state.pt')
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


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
        num_workers=4,
        pin_memory=True
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
    parser.add_argument('--val_csv_path', type=str, default=None, help='Validation CSV (optional)')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split if no val_csv')
    parser.add_argument('--image_ext', type=str, default='.jpg', help='Image file extension')
    
    # Model args
    parser.add_argument('--dinov2_model', type=str, default='facebook/dinov2-base')
    parser.add_argument('--bartpho_model', type=str, default='vinai/bartpho-syllable')
    parser.add_argument('--num_cross_attn_layers', type=int, default=3)
    parser.add_argument('--freeze_pretrained', action='store_true', help='Freeze pretrained weights')
    parser.add_argument('--unfreeze_encoder_layers', type=int, default=3, help='Unfreeze last N encoder layers')
    
    # Training args
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=500)
    
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
        
        # Freeze pretrained weights if requested
        if args.freeze_pretrained:
            model.freeze_pretrained_weights(
                unfreeze_encoder_last_n_layers=args.unfreeze_encoder_layers
            )
        
        # Prepare datasets
        logger.info("Loading datasets...")
        
        if args.val_csv_path:
            # Use separate validation CSV
            train_dataset = ViVQADataset(args.csv_path, args.image_folder, model, image_ext=args.image_ext)
            val_dataset = ViVQADataset(args.val_csv_path, args.image_folder, model, image_ext=args.image_ext)
        else:
            # Split from training CSV
            full_dataset = ViVQADataset(args.csv_path, args.image_folder, model, image_ext=args.image_ext)
            val_size = int(len(full_dataset) * args.val_split)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        # Initialize trainer
        trainer = VQATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            device=device
        )
        
        # Train
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
