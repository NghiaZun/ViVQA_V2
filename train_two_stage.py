"""
TWO-STAGE TRAINING SCRIPT (Hướng 2: Reasoning → Answer)

Key differences from train_optimal.py:
1. Data format: "Reasoning: ... \nAnswer: ..." (reasoning FIRST)
2. Loss: TwoStageCombinedLoss - separate reasoning_loss and answer_loss
3. Inference: Two-stage generation (generate reasoning → generate answer)
4. Evaluation: GT-reasoning upper bound metric

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
class TwoStageTrainConfig:
    # Paths
    train_csv: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_dir: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    teacher_jsonl: str = "/kaggle/input/final/teacher_outputs_train.jsonl"
    checkpoint_dir: str = "/kaggle/input/model-base/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working"
    
    # Model
    vision_model: str = "openai/clip-vit-large-patch14"
    hidden_dim: int = 768
    num_fusion_layers: int = 4
    num_heads: int = 12
    dropout: float = 0.1
    use_lora: bool = True
    use_type_routing: bool = True
    
    # Training
    batch_size: int = 2
    accum_steps: int = 16  # Effective batch = 32
    num_epochs: int = 120  # Stage 1-3 only (no need Stage 4-5 với two-stage)
    val_ratio: float = 0.1
    num_workers: int = 2
    
    # Learning rates
    base_lr: float = 2e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Progressive unfreezing (3 stages)
    stage1_epochs: int = 60  # Fusion + Decoder
    stage2_epochs: int = 30  # + Text encoder last 2 layers
    stage3_epochs: int = 30  # + Vision encoder last 2 layers
    
    # Two-stage loss weights
    lambda_reasoning: float = 0.5  # Reasoning loss weight
    lambda_answer: float = 1.0     # Answer loss weight (prioritize answer)
    
    # Augmentation
    use_image_aug: bool = True
    
    # Loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Inference
    num_beams: int = 4
    length_penalty: float = 1.2
    
    # Two-stage inference
    two_stage_inference: bool = True  # Generate reasoning → answer separately
    eval_with_gt_reasoning: bool = True  # Also eval with GT reasoning (upper bound)
    
    # Early stopping
    es_patience: int = 15
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log_two_stage.csv"
    clear_cache_every_n_steps: int = 20

# =====================
# TWO-STAGE COMBINED LOSS
# =====================
class TwoStageCombinedLoss(nn.Module):
    """
    Combined loss for two-stage training.
    Assumes format: "Reasoning: ... \nAnswer: ..."
    
    Computes:
    - reasoning_loss: CE loss on reasoning tokens
    - answer_loss: CE loss on answer tokens
    - combined: lambda_reasoning * reasoning_loss + lambda_answer * answer_loss
    """
    def __init__(self, lambda_reasoning=0.5, lambda_answer=1.0, ignore_index=-100):
        super().__init__()
        self.lambda_reasoning = lambda_reasoning
        self.lambda_answer = lambda_answer
        self.ignore_index = ignore_index
    
    def forward(self, logits, labels, tokenizer):
        """
        Args:
            logits: [B, L, V]
            labels: [B, L]
            tokenizer: decoder tokenizer
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Delimiter: "\nAnswer:"
        answer_delim = "\nAnswer:"
        answer_ids = tokenizer.encode(answer_delim, add_special_tokens=False)
        
        # Compute per-token CE loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        loss_per_token = F.cross_entropy(
            logits_flat, labels_flat, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        loss_per_token = loss_per_token.view(batch_size, seq_len)
        
        reasoning_losses = []
        answer_losses = []
        reasoning_counts = 0
        answer_counts = 0
        
        for b in range(batch_size):
            # Find delimiter position
            delim_pos = None
            for i in range(seq_len - len(answer_ids) + 1):
                if all(i + j < seq_len and labels[b, i + j] == answer_ids[j] 
                       for j in range(len(answer_ids))):
                    delim_pos = i
                    break
            
            if delim_pos is None:
                # No delimiter found: treat as answer-only
                valid_mask = (labels[b] != self.ignore_index)
                if valid_mask.sum() > 0:
                    answer_losses.append(loss_per_token[b][valid_mask].sum())
                    answer_counts += valid_mask.sum().item()
                continue
            
            # Reasoning tokens: [0 .. delim_pos-1]
            if delim_pos > 0:
                reasoning_mask = (labels[b, :delim_pos] != self.ignore_index)
                if reasoning_mask.sum() > 0:
                    reasoning_losses.append(loss_per_token[b, :delim_pos][reasoning_mask].sum())
                    reasoning_counts += reasoning_mask.sum().item()
            
            # Answer tokens: [delim_pos + len(answer_ids) .. end]
            ans_start = delim_pos + len(answer_ids)
            if ans_start < seq_len:
                answer_mask = (labels[b, ans_start:] != self.ignore_index)
                if answer_mask.sum() > 0:
                    answer_losses.append(loss_per_token[b, ans_start:][answer_mask].sum())
                    answer_counts += answer_mask.sum().item()
        
        # Aggregate
        reasoning_loss = torch.tensor(0.0, device=device)
        answer_loss = torch.tensor(0.0, device=device)
        
        if reasoning_counts > 0:
            reasoning_loss = sum(reasoning_losses) / reasoning_counts
        if answer_counts > 0:
            answer_loss = sum(answer_losses) / answer_counts
        
        combined = self.lambda_reasoning * reasoning_loss + self.lambda_answer * answer_loss
        return combined, reasoning_loss, answer_loss

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
        mask = (targets != ignore_index)
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
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
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, fill_value=0, p=0.2),
    ])

# =====================
# TWO-STAGE DATASET
# =====================
class TwoStageVQADataset(Dataset):
    """
    Dataset for two-stage training.
    Target format: "Reasoning: ... \nAnswer: ..." (reasoning FIRST!)
    """
    def __init__(self, csv_path, image_dir, teacher_jsonl, vision_processor,
                 text_tokenizer, decoder_tokenizer, max_len=128, 
                 use_augmentation=False):
        
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
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['img_id'])
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
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
        
        # GT-guided: if teacher answer wrong, use GT
        if teacher_answer and teacher_answer.strip().lower() != gt_answer.strip().lower():
            teacher_answer = gt_answer
            teacher_reasoning = ""
        
        # ✅ TWO-STAGE FORMAT: "Reasoning: ... \nAnswer: ..."
        if teacher_reasoning:
            target = f"Reasoning: {teacher_reasoning}\nAnswer: {teacher_answer}"
        else:
            # No reasoning: answer-only
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
            "teacher_reasoning": teacher_reasoning,
            "teacher_answer": teacher_answer,
        }

# =====================
# TRAINING FUNCTIONS
# =====================
def set_training_stage(model, stage: int):
    """Progressive unfreezing"""
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

def compute_loss(model, batch, device, criterion=None):
    """Compute loss"""
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
    
    if criterion is not None:
        logits = outputs.logits
        
        if isinstance(criterion, TwoStageCombinedLoss):
            loss, reasoning_loss, answer_loss = criterion(logits, labels, model.decoder_tokenizer)
            return loss, reasoning_loss, answer_loss
        elif isinstance(criterion, FocalLoss):
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss, None, None
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return loss, None, None
    else:
        return outputs.loss, None, None

# =====================
# TWO-STAGE INFERENCE
# =====================
def two_stage_generate(model, pixel_values, input_ids, attention_mask, 
                       max_reasoning_tokens=64, max_answer_tokens=32,
                       num_beams=4, length_penalty=1.2):
    """
    Two-stage generation:
    1. Generate reasoning: "Reasoning: ..."
    2. Generate answer conditioned on reasoning: "Reasoning: ... \nAnswer: ..."
    """
    device = pixel_values.device
    
    # Stage 1: Generate reasoning
    # Force decoder to start with "Reasoning:"
    reasoning_prefix = "Reasoning:"
    reasoning_ids = model.decoder_tokenizer.encode(reasoning_prefix, add_special_tokens=False, return_tensors='pt').to(device)
    
    # Generate reasoning tokens
    reasoning_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_reasoning_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True,
        decoder_input_ids=reasoning_ids.repeat(pixel_values.size(0), 1),
    )
    
    # Decode reasoning
    reasoning_text = model.decoder_tokenizer.batch_decode(reasoning_output, skip_special_tokens=True)
    
    # Stage 2: Generate answer conditioned on reasoning
    # Construct prompt: "Reasoning: ... \nAnswer:"
    full_outputs = []
    for i, reasoning in enumerate(reasoning_text):
        # Create full prompt
        if not reasoning.startswith("Reasoning:"):
            prompt = f"Reasoning: {reasoning}\nAnswer:"
        else:
            prompt = f"{reasoning}\nAnswer:"
        
        # Encode prompt
        prompt_ids = model.decoder_tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
        
        # Generate answer
        answer_output = model.generate(
            pixel_values=pixel_values[i:i+1],
            input_ids=input_ids[i:i+1],
            attention_mask=attention_mask[i:i+1],
            max_new_tokens=max_answer_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            decoder_input_ids=prompt_ids,
        )
        
        # Decode full output
        full_text = model.decoder_tokenizer.decode(answer_output[0], skip_special_tokens=True)
        full_outputs.append(full_text)
    
    return full_outputs

# =====================
# MAIN TRAINING
# =====================
def train():
    set_seed(42)
    cfg = TwoStageTrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("="*70)
    print("TWO-STAGE VQA TRAINING (Hướng 2: Reasoning → Answer)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Vision: {cfg.vision_model}")
    print(f"Two-stage loss: λ_reasoning={cfg.lambda_reasoning}, λ_answer={cfg.lambda_answer}")
    print(f"Two-stage inference: {cfg.two_stage_inference}")
    print(f"\n🎯 TRAINING STAGES:")
    print(f"  Stage 1 (E1-60):   Fusion + Decoder")
    print(f"  Stage 2 (E61-90):  + Text encoder (last 2 layers)")
    print(f"  Stage 3 (E91-120): + Vision encoder (last 2 layers)")
    print(f"\n💡 Format: 'Reasoning: ... \\nAnswer: ...' → Target: 65-70% accuracy!")
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
    
    # Loss functions
    two_stage_loss = TwoStageCombinedLoss(
        lambda_reasoning=cfg.lambda_reasoning,
        lambda_answer=cfg.lambda_answer
    )
    focal_loss = FocalLoss(cfg.focal_alpha, cfg.focal_gamma) if cfg.use_focal_loss else None
    
    # Dataset
    full_dataset = TwoStageVQADataset(
        cfg.train_csv, cfg.image_dir, cfg.teacher_jsonl,
        vision_processor, model.text_tokenizer, model.decoder_tokenizer,
        max_len=128, use_augmentation=cfg.use_image_aug
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
    
    # Logging
    log_path = os.path.join(cfg.save_dir, cfg.log_csv)
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,stage,train_loss,train_reasoning_loss,train_answer_loss,val_loss,val_accuracy,val_accuracy_gt_reasoning,best_val_loss\n")
    
    # Training state
    scaler = GradScaler()
    best_val_loss = float('inf')
    es_counter = 0
    start_epoch = 0
    
    # =====================
    # RESUME FROM CHECKPOINT
    # =====================
    resume_checkpoint = os.path.join(cfg.save_dir, "latest_checkpoint_two_stage.pt")
    if os.path.exists(resume_checkpoint):
        print(f"\n{'='*70}")
        print(f"🔄 RESUMING FROM CHECKPOINT: {resume_checkpoint}")
        print(f"{'='*70}")
        
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        es_counter = checkpoint.get('es_counter', 0)
        
        print(f"✅ Loaded checkpoint from epoch {start_epoch}")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print(f"   ES counter: {es_counter}/{cfg.es_patience}")
        print(f"{'='*70}\n")
    
    # =====================
    # OPTIMIZER & SCHEDULER (PRESERVE MOMENTUM!)
    # =====================
    print(f"🔧 CREATING OPTIMIZER & SCHEDULER")
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    
    total_steps = len(train_loader) * (cfg.num_epochs - start_epoch)
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Load optimizer state if resuming
    if os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Loaded optimizer state (MOMENTUM PRESERVED!)")
            except:
                print(f"⚠️  Could not load optimizer state")
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ Loaded scheduler state")
            except:
                print(f"⚠️  Could not load scheduler state")
    
    print(f"   LR: {cfg.base_lr:.2e}, Total steps: {total_steps}\n")
    
    # =====================
    # TRAINING LOOP
    # =====================
    for epoch in range(start_epoch, cfg.num_epochs):
        # Determine stage
        if epoch < cfg.stage1_epochs:
            stage = 1
        elif epoch < cfg.stage1_epochs + cfg.stage2_epochs:
            stage = 2
        else:
            stage = 3
        
        set_training_stage(model, stage)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Stage {stage} | Trainable: {trainable_params:.1f}M")
        print(f"{'='*70}")
        
        # Select criterion
        criterion = two_stage_loss
        
        # ==================
        # TRAINING
        # ==================
        model.train()
        train_loss = 0
        train_reasoning_loss = 0
        train_answer_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}", ncols=100, leave=False)
        
        for step, batch in enumerate(pbar):
            with autocast():
                loss, r_loss, a_loss = compute_loss(model, batch, device, criterion)
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
            
            train_loss += loss.item() * cfg.accum_steps
            if r_loss is not None:
                train_reasoning_loss += r_loss.item()
            if a_loss is not None:
                train_answer_loss += a_loss.item()
            
            if step % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item() * cfg.accum_steps:.4f}'})
            
            if (step + 1) % cfg.clear_cache_every_n_steps == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        pbar.close()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_r_loss = train_reasoning_loss / len(train_loader)
        avg_train_a_loss = train_answer_loss / len(train_loader)
        
        # ==================
        # VALIDATION
        # ==================
        model.eval()
        val_loss = 0
        val_correct = 0
        val_correct_gt_reasoning = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val E{epoch+1}", ncols=100, leave=False)):
                # Compute loss
                loss, _, _ = compute_loss(model, batch, device, criterion)
                val_loss += loss.item()
                
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Standard generation
                if cfg.two_stage_inference and batch_idx < 5:  # Two-stage for first few batches (slow)
                    predictions = two_stage_generate(
                        model, pixel_values, input_ids, attention_mask,
                        num_beams=cfg.num_beams, length_penalty=cfg.length_penalty
                    )
                else:
                    # Standard generation (faster)
                    output_ids = model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=96,
                        num_beams=cfg.num_beams,
                        length_penalty=cfg.length_penalty,
                        early_stopping=True
                    )
                    predictions = model.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                # GT labels
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = model.decoder_tokenizer.pad_token_id
                gt_answers = model.decoder_tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                
                # Calculate accuracy
                for i, (pred, gt) in enumerate(zip(predictions, gt_answers)):
                    # Extract answer
                    if "Answer:" in pred:
                        pred_answer = pred.split("Answer:")[-1].split("\n")[0].strip().lower()
                    else:
                        pred_answer = pred.split("\n")[0].strip().lower()
                    
                    if "Answer:" in gt:
                        gt_answer = gt.split("Answer:")[-1].split("\n")[0].strip().lower()
                    else:
                        gt_answer = gt.split("\n")[0].strip().lower()
                    
                    # Normalize
                    pred_answer = ' '.join(pred_answer.split())
                    gt_answer = ' '.join(gt_answer.split())
                    
                    if pred_answer == gt_answer:
                        val_correct += 1
                    val_total += 1
                
                # GT reasoning evaluation (upper bound)
                if cfg.eval_with_gt_reasoning and batch_idx < 3:  # Only first few batches
                    teacher_reasoning_list = batch.get("teacher_reasoning", None)
                    teacher_answer_list = batch.get("teacher_answer", None)
                    
                    if teacher_reasoning_list is not None and teacher_answer_list is not None:
                        for i in range(len(teacher_reasoning_list)):
                            if teacher_reasoning_list[i]:
                                # Construct GT reasoning prompt
                                gt_reasoning_prompt = f"Reasoning: {teacher_reasoning_list[i]}\nAnswer:"
                                prompt_ids = model.decoder_tokenizer.encode(
                                    gt_reasoning_prompt, 
                                    add_special_tokens=False, 
                                    return_tensors='pt'
                                ).to(device)
                                
                                # Generate answer conditioned on GT reasoning
                                answer_output = model.generate(
                                    pixel_values=pixel_values[i:i+1],
                                    input_ids=input_ids[i:i+1],
                                    attention_mask=attention_mask[i:i+1],
                                    max_new_tokens=32,
                                    num_beams=2,
                                    decoder_input_ids=prompt_ids,
                                )
                                
                                pred_with_gt = model.decoder_tokenizer.decode(answer_output[0], skip_special_tokens=True)
                                pred_answer_gt = pred_with_gt.split("Answer:")[-1].split("\n")[0].strip().lower()
                                pred_answer_gt = ' '.join(pred_answer_gt.split())
                                
                                if pred_answer_gt == gt_answer:
                                    val_correct_gt_reasoning += 1
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        val_accuracy_gt = 100.0 * val_correct_gt_reasoning / val_total if val_total > 0 else 0.0
        
        # ==================
        # LOGGING
        # ==================
        print(f"[EPOCH {epoch+1}] Train Loss: {avg_train_loss:.4f} (R: {avg_train_r_loss:.4f}, A: {avg_train_a_loss:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val Acc (GT reasoning): {val_accuracy_gt:.2f}%")
        
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{stage},{avg_train_loss:.6f},{avg_train_r_loss:.6f},{avg_train_a_loss:.6f},"
                   f"{avg_val_loss:.6f},{val_accuracy:.2f},{val_accuracy_gt:.2f},{best_val_loss:.6f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss - cfg.es_min_delta:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_two_stage_model.pt"))
            print(f"✅ Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            es_counter += 1
            print(f"⚠️  No improvement ({es_counter}/{cfg.es_patience})")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'es_counter': es_counter,
        }
        torch.save(checkpoint, os.path.join(cfg.save_dir, "latest_checkpoint_two_stage.pt"))
        
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(cfg.save_dir, f"checkpoint_two_stage_epoch_{epoch+1}.pt"))
        
        # Early stopping
        if es_counter >= cfg.es_patience:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*70)

if __name__ == "__main__":
    train()
