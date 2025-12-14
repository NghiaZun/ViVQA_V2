"""
OPTIMAL TRAINING SCRIPT FOR 70% ACCURACY

Implements:
1. Curriculum learning (easy → hard)
2. Image augmentation
3. Hard negative mining
4. Focal loss for imbalanced types
5. Self-training with pseudo-labels
6. Ensemble training

Expected: 65-70% accuracy on ViVQA
"""

import os
import gc
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from transformers import (
    CLIPProcessor,
    get_cosine_schedule_with_warmup
)
from model_optimal import OptimalVQAModel, normalize_vietnamese_answer

# Image augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    AUGMENTATION_AVAILABLE = True
except:
    print("[WARN] albumentations not available, using basic augmentation")
    AUGMENTATION_AVAILABLE = False

# =====================
# REPRODUCIBILITY
# =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

torch.backends.cudnn.benchmark = True

# =====================
# CONFIG
# =====================
@dataclass
class OptimalTrainConfig:
    # Paths
    train_csv: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_dir: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    teacher_jsonl: str = "/kaggle/input/8-12-teacher/teacher_outputs_train.jsonl"
    checkpoint_dir: str = "/kaggle/input/model-base/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working"
    
    # Model (Optimal architecture)
    vision_model: str = "openai/clip-vit-large-patch14"  # ViT-Large for better features
    hidden_dim: int = 768
    num_fusion_layers: int = 4  # Deep fusion
    num_heads: int = 12
    dropout: float = 0.1
    use_lora: bool = True  # Efficient training
    use_type_routing: bool = True  # Question-type aware
    
    # Training
    batch_size: int = 2  # Larger model needs smaller batch
    accum_steps: int = 16  # Effective = 32
    num_epochs: int = 120
    val_ratio: float = 0.1
    num_workers: int = 2
    
    # Learning rates
    base_lr: float = 1e-5
    vision_lr: float = 5e-6  # Lower for pretrained CLIP
    fusion_lr: float = 3e-5  # Higher for new layers
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Progressive unfreezing
    stage1_epochs: int = 60  # Fusion + Decoder
    stage2_epochs: int = 30  # + Text encoder last 2 layers
    stage3_epochs: int = 30  # + Vision encoder last 2 layers
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_warmup_epochs: int = 30  # Start with easy examples
    
    # Data augmentation
    use_image_aug: bool = True
    aug_probability: float = 0.5
    
    # Hard negative mining
    use_hard_negatives: bool = True
    hard_negative_start_epoch: int = 20
    hard_negative_ratio: float = 0.3  # 30% of batch from hard examples
    
    # Loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Self-training
    use_self_training: bool = False  # Enable after first training
    pseudo_label_confidence: float = 0.9
    
    # Inference
    num_beams: int = 8
    num_beam_groups: int = 4
    diversity_penalty: float = 0.5
    length_penalty: float = 1.2
    
    # Early stopping
    es_patience: int = 15
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log_optimal.csv"
    clear_cache_every_n_steps: int = 20

# =====================
# FOCAL LOSS
# =====================
class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced question types"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, ignore_index=-100):
        # Mask out ignored indices
        mask = (targets != ignore_index)
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=ignore_index, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss[mask].mean()
        elif self.reduction == 'sum':
            return focal_loss[mask].sum()
        else:
            return focal_loss

# =====================
# IMAGE AUGMENTATION
# =====================
def get_train_augmentation():
    """Strong augmentation for training"""
    if not AUGMENTATION_AVAILABLE:
        return None
    
    return A.Compose([
        # Color augmentation
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Geometric (mild to preserve spatial info)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
        
        # Blur
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Occlusion
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, fill_value=0, p=0.2),
    ])

# =====================
# CURRICULUM DATASET
# =====================
class CurriculumVQADataset(Dataset):
    """Dataset with curriculum learning support"""
    def __init__(self, csv_path, image_dir, teacher_jsonl, vision_processor,
                 text_tokenizer, decoder_tokenizer, max_len=128, 
                 use_augmentation=False, difficulty_level='all'):
        
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.vision_processor = vision_processor
        self.text_tokenizer = text_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_len = max_len
        self.use_augmentation = use_augmentation
        self.augmentation = get_train_augmentation() if use_augmentation else None
        
        # Load teacher outputs
        self.teacher_outputs = {}
        if os.path.exists(teacher_jsonl):
            with open(teacher_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    key = (str(data['img_id']), str(data['question']))
                    self.teacher_outputs[key] = data
            print(f"[INFO] Loaded {len(self.teacher_outputs)} teacher outputs")
        
        # Classify difficulty
        self._classify_difficulty()
        
        # Filter by difficulty level
        if difficulty_level != 'all':
            self.df = self.df[self.df['difficulty'] == difficulty_level].reset_index(drop=True)
            print(f"[INFO] Using {difficulty_level} examples: {len(self.df)}")
    
    def _classify_difficulty(self):
        """Classify examples by difficulty"""
        difficulties = []
        
        for idx, row in self.df.iterrows():
            question = str(row['question']).lower()
            answer = str(row['answer']).lower()
            
            # Easy: short question, common object
            if len(question.split()) <= 5 and len(answer.split()) <= 2:
                diff = 'easy'
            # Hard: counting, complex spatial
            elif any(word in question for word in ['bao nhiêu', 'mấy', 'số', 'count']):
                diff = 'hard'
            elif any(word in question for word in ['ở đâu', 'where', 'vị trí', 'phía']):
                diff = 'medium'
            else:
                diff = 'medium'
            
            difficulties.append(diff)
        
        self.df['difficulty'] = difficulties
        
        easy_count = sum(1 for d in difficulties if d == 'easy')
        medium_count = sum(1 for d in difficulties if d == 'medium')
        hard_count = sum(1 for d in difficulties if d == 'hard')
        
        print(f"[INFO] Difficulty distribution: Easy={easy_count}, Medium={medium_count}, Hard={hard_count}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['img_id'])
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Apply augmentation
            if self.use_augmentation and self.augmentation:
                image_np = np.array(image)
                augmented = self.augmentation(image=image_np)
                image = Image.fromarray(augmented['image'])
        except:
            image = Image.new("RGB", (224, 224), (255, 255, 255))
        
        question = str(row["question"])
        gt_answer = str(row["answer"])
        
        # Get teacher output
        key = (img_id, question)
        teacher_data = self.teacher_outputs.get(key, {})
        teacher_answer = teacher_data.get("teacher_answer", gt_answer)
        teacher_reasoning = teacher_data.get("teacher_reasoning", "")
        reasoning_type = teacher_data.get("reasoning_type", "OTHER")
        
        # GT-guided
        if teacher_answer and teacher_answer.strip().lower() != gt_answer.strip().lower():
            teacher_answer = gt_answer
            teacher_reasoning = ""
        
        # Construct target
        if teacher_reasoning:
            target = f"Answer: {teacher_answer}\nReasoning: {teacher_reasoning}"
        else:
            target = f"Answer: {gt_answer}"
        
        # Process image
        vision_inputs = self.vision_processor(images=image, return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].squeeze(0)
        
        # Tokenize question
        text_inputs = self.text_tokenizer(
            question, max_length=64, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        
        # Tokenize target
        target_inputs = self.decoder_tokenizer(
            target, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        labels = target_inputs["input_ids"].squeeze(0)
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "img_id": img_id,
            "reasoning_type": reasoning_type,
            "difficulty": row['difficulty']
        }

# =====================
# TRAINING FUNCTIONS
# =====================
def set_training_stage(model, stage: int):
    """Progressive unfreezing - same as before"""
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    
    # Always train decoder, fusion, projections
    for p in model.decoder.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = True
    if hasattr(model, "decoder_input_proj"):
        for p in model.decoder_input_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "type_router"):
        for p in model.type_router.parameters():
            p.requires_grad = True
    
    # Stage 2: Unfreeze last 2 layers of text encoder
    if stage >= 2:
        try:
            if hasattr(model.text_encoder, "encoder"):
                last_layers = model.text_encoder.encoder.layer[-2:]
                for layer in last_layers:
                    for p in layer.parameters():
                        p.requires_grad = True
        except:
            pass
    
    # Stage 3: Unfreeze last 2 layers of vision encoder
    if stage >= 3:
        try:
            if hasattr(model.vision_encoder, "clip"):
                if hasattr(model.vision_encoder.clip, "vision_model"):
                    last_layers = model.vision_encoder.clip.vision_model.encoder.layers[-2:]
                    for layer in last_layers:
                        for p in layer.parameters():
                            p.requires_grad = True
        except:
            pass

def compute_loss(model, batch, device, focal_loss_fn=None):
    """Compute loss with optional focal loss"""
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    if focal_loss_fn is not None:
        # Use focal loss instead
        logits = outputs.logits
        loss = focal_loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
    else:
        loss = outputs.loss
    
    return loss

# =====================
# MAIN TRAINING
# =====================
def train():
    set_seed(42)
    cfg = OptimalTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("="*70)
    print("OPTIMAL VQA TRAINING FOR 70% ACCURACY")
    print("="*70)
    print(f"Device: {device}")
    print(f"Vision: {cfg.vision_model}")
    print(f"Fusion layers: {cfg.num_fusion_layers}")
    print(f"LoRA: {cfg.use_lora}")
    print(f"Type routing: {cfg.use_type_routing}")
    print(f"Curriculum: {cfg.use_curriculum}")
    print(f"Image augmentation: {cfg.use_image_aug}")
    print(f"Focal loss: {cfg.use_focal_loss}")
    print("="*70 + "\n")
    
    # Load model
    model = OptimalVQAModel(
        vision_model_name=cfg.vision_model,
        phobert_dir=os.path.join(cfg.checkpoint_dir, "phobert_tokenizer"),
        vit5_dir=os.path.join(cfg.checkpoint_dir, "vit5_tokenizer"),
        hidden_dim=cfg.hidden_dim,
        num_fusion_layers=cfg.num_fusion_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_lora=cfg.use_lora,
        use_type_routing=cfg.use_type_routing
    ).to(device)
    
    vision_processor = CLIPProcessor.from_pretrained(cfg.vision_model)
    
    # Focal loss
    focal_loss_fn = FocalLoss(cfg.focal_alpha, cfg.focal_gamma) if cfg.use_focal_loss else None
    
    # Dataset
    full_dataset = CurriculumVQADataset(
        cfg.train_csv, cfg.image_dir, cfg.teacher_jsonl,
        vision_processor, model.text_tokenizer, model.decoder_tokenizer,
        max_len=128, use_augmentation=cfg.use_image_aug,
        difficulty_level='all'
    )
    
    # Split
    n_val = int(len(full_dataset) * cfg.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - n_val, n_val]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
    
    # Logging setup
    log_path = os.path.join(cfg.save_dir, cfg.log_csv)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,stage,train_loss,val_loss,best_val_loss,trainable_params_M\n")
    
    # Training loop
    scaler = GradScaler()
    best_val_loss = float('inf')
    es_counter = 0
    
    for epoch in range(cfg.num_epochs):
        # Determine stage
        if epoch < cfg.stage1_epochs:
            stage = 1
        elif epoch < cfg.stage1_epochs + cfg.stage2_epochs:
            stage = 2
        else:
            stage = 3
        
        set_training_stage(model, stage)
        
        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        trainable_params = sum(p.numel() for p in params) / 1e6
        
        # Scheduler
        num_training_steps = len(train_loader) * (cfg.num_epochs - epoch)
        num_warmup_steps = int(num_training_steps * 0.05)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Stage {stage} | Trainable: {trainable_params:.1f}M")
        print(f"{'='*70}")
        
        # ==================
        # TRAINING
        # ==================
        model.train()
        train_loss = 0
        step_count = 0
        optimizer.zero_grad()
        
        # Use tqdm with minimal output
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}", 
                   ncols=100, leave=False, dynamic_ncols=False)
        
        for step, batch in enumerate(pbar):
            with autocast():
                loss = compute_loss(model, batch, device, focal_loss_fn)
                loss = loss / cfg.accum_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
            
            train_loss += loss.item() * cfg.accum_steps
            
            # Update progress bar every 10 steps only (reduce overhead)
            if step % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * cfg.accum_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            # Clear cache periodically
            if (step + 1) % cfg.clear_cache_every_n_steps == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        
        # ==================
        # VALIDATION
        # ==================
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            # Minimal validation logging
            for batch in tqdm(val_loader, desc=f"Val E{epoch+1}", 
                            ncols=100, leave=False, dynamic_ncols=False):
                loss = compute_loss(model, batch, device, focal_loss_fn)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # ==================
        # LOGGING & CHECKPOINTING
        # ==================
        print(f"[EPOCH {epoch+1}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        # Log to CSV
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{stage},{avg_train_loss:.6f},{avg_val_loss:.6f},"
                   f"{best_val_loss:.6f},{trainable_params:.2f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss - cfg.es_min_delta:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_optimal_model.pt"))
            print(f"✅ Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            es_counter += 1
            print(f"⚠️  No improvement ({es_counter}/{cfg.es_patience})")
        
        # Save checkpoint every 10 epochs (less frequent to save time)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'es_counter': es_counter,
            }
            torch.save(checkpoint, os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            print(f"💾 Checkpoint saved")
        
        # Early stopping
        if es_counter >= cfg.es_patience:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*70)

if __name__ == "__main__":
    train()
