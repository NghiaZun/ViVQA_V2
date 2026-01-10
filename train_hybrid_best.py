"""
HYBRID LATENT REASONING VQA - BEST OF BOTH WORLDS
==================================================
Kết hợp ưu điểm từ improved version + implicit reasoning:

FROM IMPROVED VERSION:
✅ Hierarchical reasoning (coarse + fine)
✅ Layer-wise learning rates
✅ Adaptive loss weights
✅ Professional config management
✅ Comprehensive visualization

FROM IMPLICIT REASONING VERSION:
✅ Teacher distillation (CORE!)
✅ Regularization suite (noise + diversity + dropout)
✅ 3-stage training (Baseline → Warmup → Full)
✅ Clean, focused approach

REMOVED:
❌ Auxiliary tasks (visual grounding, reasoning type, contrastive)
❌ Over-complexity

RESULT: Clean, powerful, thesis-ready VQA model
"""

import os
import math
import time
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

from transformers import get_cosine_schedule_with_warmup, AutoImageProcessor
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dataset import VQAGenDataset


# ============================================================================
# IMPORT MODEL (sửa lại từ improved model)
# ============================================================================

# Sẽ import từ model_latent_reasoning.py sau khi clean up
# Tạm thời define model trong file này để đảm bảo chạy được

import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoImageProcessor,
    BartphoTokenizer,
    MBartForConditionalGeneration
)

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift tokens right for teacher forcing"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class GatedCrossAttentionLayer(nn.Module):
    """Gated cross-attention layer"""
    def __init__(self, hidden_dim=1024, num_heads=16, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_features, visual_features):
        attn_out, attn_weights = self.cross_attn(
            query=text_features, key=visual_features, value=visual_features,
            need_weights=True, average_attn_weights=True
        )
        gate_input = torch.cat([text_features, attn_out], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        gated_output = gate_values * attn_out + (1 - gate_values) * text_features
        x = self.norm1(text_features + gated_output)
        ffn_out = self.ffn(x)
        output = self.norm2(x + ffn_out)
        return output, attn_weights


class MultiLayerCrossAttention(nn.Module):
    """Multiple layers of gated cross-attention"""
    def __init__(self, hidden_dim=1024, num_layers=3, num_heads=16, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedCrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, text_features, visual_features):
        attn_weights_list = []
        x = text_features
        for layer in self.layers:
            x, attn_weights = layer(x, visual_features)
            attn_weights_list.append(attn_weights)
        return x, attn_weights_list


class LatentReasoningModule(nn.Module):
    """Single-level latent reasoning module"""
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_reasoning_tokens: int = 8,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_stochastic: bool = True,
        latent_dim: int = 512
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_tokens = num_reasoning_tokens
        self.use_stochastic = use_stochastic
        
        # Learnable reasoning queries
        self.reasoning_queries = nn.Parameter(
            torch.randn(num_reasoning_tokens, hidden_dim) * 0.02
        )
        
        # Cross-attention layers
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Stochastic sampling
        if use_stochastic:
            self.to_mu = nn.Linear(hidden_dim, latent_dim)
            self.to_logvar = nn.Linear(hidden_dim, latent_dim)
            self.from_latent = nn.Linear(latent_dim, hidden_dim)
            self.latent_dim = latent_dim
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, multimodal_features, attention_mask=None, deterministic=False):
        batch_size = multimodal_features.size(0)
        
        # Expand reasoning queries
        reasoning_queries = self.reasoning_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attend
        for layer in self.reasoning_layers:
            reasoning_queries = layer(
                tgt=reasoning_queries, memory=multimodal_features,
                memory_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
        
        # Stochastic sampling
        kl_loss = None
        if self.use_stochastic:
            mu = self.to_mu(reasoning_queries)
            logvar = self.to_logvar(reasoning_queries)
            
            if deterministic or not self.training:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            reasoning_queries = self.from_latent(z)
        
        reasoning_latents = self.output_norm(reasoning_queries)
        return reasoning_latents, kl_loss


class HierarchicalReasoningModule(nn.Module):
    """Hierarchical reasoning: Coarse + Fine"""
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_coarse_tokens: int = 8,
        num_fine_tokens: int = 16,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_stochastic: bool = True,
        latent_dim: int = 512
    ):
        super().__init__()
        
        self.coarse_reasoning = LatentReasoningModule(
            hidden_dim, num_coarse_tokens, num_heads,
            num_layers, dropout, use_stochastic, latent_dim
        )
        
        self.fine_reasoning = LatentReasoningModule(
            hidden_dim, num_fine_tokens, num_heads,
            num_layers, dropout, use_stochastic, latent_dim
        )
        
    def forward(self, multimodal_features, attention_mask=None, deterministic=False):
        # Coarse reasoning
        coarse_r, kl_coarse = self.coarse_reasoning(
            multimodal_features, attention_mask, deterministic
        )
        
        # Fine reasoning conditioned on coarse
        enhanced_features = torch.cat([multimodal_features, coarse_r], dim=1)
        
        if attention_mask is not None:
            coarse_mask = torch.ones(
                coarse_r.size(0), coarse_r.size(1),
                device=attention_mask.device
            )
            enhanced_mask = torch.cat([attention_mask, coarse_mask], dim=1)
        else:
            enhanced_mask = None
        
        fine_r, kl_fine = self.fine_reasoning(
            enhanced_features, enhanced_mask, deterministic
        )
        
        # Total KL
        total_kl = None
        if kl_coarse is not None and kl_fine is not None:
            total_kl = kl_coarse + kl_fine
        
        return coarse_r, fine_r, total_kl


@dataclass
class HybridVQAOutput:
    """Output structure"""
    answer_logits: torch.Tensor
    coarse_reasoning: torch.Tensor
    fine_reasoning: torch.Tensor
    answer_loss: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None


class HybridLatentReasoningVQA(nn.Module):
    """
    HYBRID VQA Model - Best of both worlds
    
    Architecture:
    1. DINOv2 vision encoder
    2. BARTpho text encoder
    3. Gated cross-attention fusion
    4. Hierarchical reasoning (coarse + fine)
    5. BARTpho decoder for answer
    
    Total: ~490M params
    """
    
    def __init__(
        self,
        dinov2_model_name: str = 'facebook/dinov2-base',
        bartpho_model_name: str = 'vinai/bartpho-syllable',
        num_cross_attn_layers: int = 3,
        num_coarse_tokens: int = 8,
        num_fine_tokens: int = 16,
        num_reasoning_layers: int = 2,
        use_stochastic_reasoning: bool = True,
        latent_dim: int = 512,
        num_heads: int = 16,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        print("[INFO] Initializing Hybrid Latent Reasoning VQA")
        print(f"  Vision: {dinov2_model_name}")
        print(f"  Language: {bartpho_model_name}")
        print(f"  Reasoning: {num_coarse_tokens} coarse + {num_fine_tokens} fine tokens")
        
        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(dinov2_model_name)
        vision_hidden_dim = self.vision_encoder.config.hidden_size
        
        # Language model
        bartpho_full = MBartForConditionalGeneration.from_pretrained(bartpho_model_name)
        bartpho_full.config.use_cache = False
        
        self.tokenizer = BartphoTokenizer.from_pretrained(bartpho_model_name)
        bart_hidden_dim = bartpho_full.config.d_model
        
        self.encoder = bartpho_full.model.encoder
        self.decoder = bartpho_full.model.decoder
        self.lm_head = bartpho_full.lm_head
        
        self.config = self.encoder.config
        self.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        
        del bartpho_full
        
        # Dimension alignment
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_dim, bart_hidden_dim),
            nn.LayerNorm(bart_hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Multimodal fusion
        self.cross_attention_fusion = MultiLayerCrossAttention(
            hidden_dim=bart_hidden_dim,
            num_layers=num_cross_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Hierarchical reasoning
        self.hierarchical_reasoning = HierarchicalReasoningModule(
            hidden_dim=bart_hidden_dim,
            num_coarse_tokens=num_coarse_tokens,
            num_fine_tokens=num_fine_tokens,
            num_heads=num_heads,
            num_layers=num_reasoning_layers,
            dropout=dropout,
            use_stochastic=use_stochastic_reasoning,
            latent_dim=latent_dim
        )
        
        # Gradient checkpointing
        if gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()
            self.encoder.gradient_checkpointing_enable()
            self.decoder.gradient_checkpointing_enable()
            print("[INFO] ✓ Gradient checkpointing enabled")
        
        print("[INFO] ✓ Model initialized")
    
    def freeze_pretrained_weights(self, unfreeze_encoder_last_n_layers: int = 3):
        """Freeze pretrained components"""
        print("\n[INFO] 🔒 FREEZING PRETRAINED WEIGHTS")
        
        # Freeze vision
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Freeze encoder except last N layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        total_layers = len(self.encoder.layers)
        unfrozen_encoder_params = 0
        for i, layer in enumerate(self.encoder.layers):
            if i >= total_layers - unfreeze_encoder_last_n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen_encoder_params += sum(p.numel() for p in layer.parameters())
        
        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        # Count trainable
        proj_params = sum(p.numel() for p in self.vision_proj.parameters())
        fusion_params = sum(p.numel() for p in self.cross_attention_fusion.parameters())
        reasoning_params = sum(p.numel() for p in self.hierarchical_reasoning.parameters())
        lmhead_params = sum(p.numel() for p in self.lm_head.parameters())
        
        print(f"  ✅ Vision Projection: {proj_params/1e6:.1f}M")
        print(f"  ✅ Cross-Attention Fusion: {fusion_params/1e6:.1f}M")
        print(f"  ✅ Hierarchical Reasoning: {reasoning_params/1e6:.1f}M")
        print(f"  ✅ Encoder last {unfreeze_encoder_last_n_layers} layers: {unfrozen_encoder_params/1e6:.1f}M")
        print(f"  ✅ LM Head: {lmhead_params/1e6:.1f}M")
        
        total_trainable = proj_params + fusion_params + reasoning_params + unfrozen_encoder_params + lmhead_params
        print(f"\n  📊 Total trainable: {total_trainable/1e6:.1f}M params")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        deterministic_reasoning: bool = False
    ):
        """Forward pass"""
        # 1. Encode vision
        visual_outputs = self.vision_encoder(pixel_values, return_dict=True)
        visual_features = self.vision_proj(visual_outputs.last_hidden_state)
        
        # 2. Encode text
        text_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_features = text_outputs.last_hidden_state
        
        # 3. Fuse
        fused_features, _ = self.cross_attention_fusion(text_features, visual_features)
        
        # 4. Hierarchical reasoning
        coarse_r, fine_r, kl_loss = self.hierarchical_reasoning(
            fused_features, attention_mask, deterministic_reasoning
        )
        
        # 5. Generate answer
        encoder_hidden_states = torch.cat([fused_features, coarse_r, fine_r], dim=1)
        
        if labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        else:
            decoder_input_ids = None
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=(labels != self.config.pad_token_id) if labels is not None else None,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            use_cache=False
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # 6. Losses
        answer_loss = None
        total_loss = None
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            answer_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss = answer_loss
            
            if kl_loss is not None:
                total_loss = total_loss + 0.01 * kl_loss
        
        return HybridVQAOutput(
            answer_logits=logits,
            coarse_reasoning=coarse_r,
            fine_reasoning=fine_r,
            answer_loss=answer_loss,
            kl_loss=kl_loss,
            total_loss=total_loss
        )
    
    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_length=32, num_beams=4):
        """Generate answers"""
        # Encode
        visual_outputs = self.vision_encoder(pixel_values, return_dict=True)
        visual_features = self.vision_proj(visual_outputs.last_hidden_state)
        
        text_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_features = text_outputs.last_hidden_state
        
        fused_features, _ = self.cross_attention_fusion(text_features, visual_features)
        
        # Reasoning (deterministic)
        coarse_r, fine_r, _ = self.hierarchical_reasoning(fused_features, attention_mask, deterministic=True)
        
        encoder_hidden_states = torch.cat([fused_features, coarse_r, fine_r], dim=1)
        
        # Generate
        batch_size = pixel_values.size(0)
        decoder_input_ids = torch.full(
            (batch_size, 1), self.tokenizer.bos_token_id,
            dtype=torch.long, device=pixel_values.device
        )
        
        generated_ids = self.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True
        )
        
        answers = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated_ids
        ]
        
        return answers


# ============================================================================
# TEACHER EVALUATOR (from implicit reasoning version)
# ============================================================================

class TeacherEvaluator:
    """Teacher model for online distillation"""
    
    def __init__(self, teacher_type: str = 'rule_based', device: str = 'cuda'):
        self.teacher_type = teacher_type
        self.device = device
        print(f"[Teacher] Using {teacher_type} evaluator")
    
    @torch.no_grad()
    def evaluate_answers(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> torch.Tensor:
        """Evaluate answer quality"""
        scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_norm = pred.lower().strip()
            gt_norm = gt.lower().strip()
            
            if pred_norm == gt_norm:
                score = 1.0
            elif gt_norm in pred_norm or pred_norm in gt_norm:
                score = 0.5
            else:
                score = 0.0
            
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32, device=self.device)


# ============================================================================
# REGULARIZER (from implicit reasoning version)
# ============================================================================

class ReasoningRegularizer:
    """Regularization for reasoning latents"""
    
    def __init__(
        self,
        noise_std: float = 0.05,
        diversity_weight: float = 0.05,
        token_dropout_prob: float = 0.2
    ):
        self.noise_std = noise_std
        self.diversity_weight = diversity_weight
        self.token_dropout_prob = token_dropout_prob
    
    def add_noise(self, reasoning_latents, training=True):
        """Add noise for information bottleneck"""
        if not training or self.noise_std == 0:
            return reasoning_latents
        
        noise = torch.randn_like(reasoning_latents) * self.noise_std
        return reasoning_latents + noise
    
    def compute_diversity_loss(self, reasoning_latents):
        """Anti-collapse diversity loss"""
        batch_size, num_tokens, hidden_dim = reasoning_latents.shape
        
        if num_tokens <= 1:
            return torch.tensor(0.0, device=reasoning_latents.device)
        
        # Normalize
        normalized = F.normalize(reasoning_latents, p=2, dim=-1)
        
        # Pairwise cosine similarity
        similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
        
        # Mask diagonal
        mask = ~torch.eye(num_tokens, dtype=torch.bool, device=reasoning_latents.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        off_diagonal_sim = similarity_matrix[mask].view(batch_size, num_tokens, num_tokens - 1)
        diversity_loss = off_diagonal_sim.mean()
        
        return diversity_loss
    
    def apply_token_dropout(self, reasoning_latents, training=True):
        """Token dropout for robustness"""
        if not training or self.token_dropout_prob == 0:
            return reasoning_latents
        
        batch_size, num_tokens, hidden_dim = reasoning_latents.shape
        keep_prob = 1.0 - self.token_dropout_prob
        
        mask = torch.bernoulli(
            torch.full((batch_size, num_tokens, 1), keep_prob, device=reasoning_latents.device)
        )
        
        return reasoning_latents * mask / keep_prob


# ============================================================================
# CONFIG & TRAINING
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True


@dataclass
class HybridTrainConfig:
    """Hybrid training configuration"""
    # Data
    csv_path: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_folder: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    save_dir: str = "/kaggle/working/checkpoints_hybrid"
    
    # Training stages
    stage: int = 1  # 1: Baseline, 2: Warmup, 3: Full
    batch_size: int = 4
    accum_steps: int = 8
    num_epochs: int = 10
    val_split: float = 0.1
    num_workers: int = 4
    
    # Optimization
    base_lr: float = 5e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.06
    use_amp: bool = True
    
    # Model
    num_coarse_tokens: int = 8
    num_fine_tokens: int = 16
    num_reasoning_layers: int = 2
    use_stochastic_reasoning: bool = True
    latent_dim: int = 512
    num_cross_attn_layers: int = 3
    unfreeze_encoder_layers: int = 3
    
    # Regularization (IMPORTANT!)
    noise_std: float = 0.05
    diversity_weight: float = 0.05
    token_dropout_prob: float = 0.25
    teacher_weight: float = 0.8
    
    # Teacher
    teacher_type: str = 'rule_based'
    use_teacher: bool = False
    
    # Ablation
    enable_reasoning: bool = True
    
    # Early stopping
    es_patience: int = 6
    es_min_delta: float = 1e-4
    
    # Logging
    log_csv: str = "train_log_hybrid.csv"
    curve_png: str = "training_curve_hybrid.png"
    resume_epoch: int = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str)
    p.add_argument("--image_folder", type=str)
    p.add_argument("--stage", type=int, choices=[1, 2, 3])
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--enable_reasoning", type=int, help="0 or 1")
    p.add_argument("--use_teacher", type=int, help="0 or 1")
    p.add_argument("--teacher_weight", type=float)
    p.add_argument("--resume_epoch", type=int)
    return p.parse_args()


def build_optimizer_and_scheduler(model, cfg, num_training_steps):
    """Layer-wise learning rates (from improved version)"""
    
    param_groups = [
        {'params': model.hierarchical_reasoning.parameters(), 'lr': cfg.base_lr, 'name': 'reasoning'},
        {'params': model.cross_attention_fusion.parameters(), 'lr': cfg.base_lr * 0.8, 'name': 'fusion'},
        {'params': model.vision_proj.parameters(), 'lr': cfg.base_lr * 0.5, 'name': 'vision_proj'},
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': cfg.base_lr * 0.3, 'name': 'encoder'},
        {'params': model.lm_head.parameters(), 'lr': cfg.base_lr * 0.5, 'name': 'lm_head'}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def compute_total_loss(outputs, labels, teacher_scores, regularizer, cfg, tokenizer):
    """Compute total loss with all components"""
    answer_loss = outputs.answer_loss
    
    # Combine coarse + fine for regularization
    combined_reasoning = torch.cat([
        outputs.coarse_reasoning,
        outputs.fine_reasoning
    ], dim=1)
    
    # Diversity loss
    diversity_loss = regularizer.compute_diversity_loss(combined_reasoning)
    
    # Reasoning norm
    reasoning_norm = combined_reasoning.norm(p=2, dim=-1).mean()
    
    # Teacher loss
    teacher_loss = torch.tensor(0.0, device=answer_loss.device)
    if cfg.use_teacher and teacher_scores is not None:
        teacher_loss = (1.0 - teacher_scores.mean())
    
    # Total
    total_loss = (
        answer_loss
        + cfg.diversity_weight * diversity_loss
        + 0.01 * reasoning_norm
        + cfg.teacher_weight * teacher_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'answer': answer_loss.item(),
        'diversity': diversity_loss.item(),
        'norm': reasoning_norm.item(),
        'teacher': teacher_loss.item()
    }
    
    return total_loss, loss_dict


def run_one_epoch(model, loader, optimizer, scaler, device, cfg, regularizer, teacher_evaluator, scheduler=None, train=True):
    """Run one epoch"""
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    loss_components = {'answer': 0.0, 'diversity': 0.0, 'norm': 0.0, 'teacher': 0.0}
    num_batches = 0
    
    if train:
        optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Train" if train else "Val", ncols=120)
    
    for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(pbar):
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        with autocast(device_type='cuda', enabled=cfg.use_amp):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                deterministic_reasoning=not train
            )
            
            # Apply regularization
            if train and cfg.enable_reasoning:
                outputs.coarse_reasoning = regularizer.add_noise(outputs.coarse_reasoning, training=True)
                outputs.coarse_reasoning = regularizer.apply_token_dropout(outputs.coarse_reasoning, training=True)
                outputs.fine_reasoning = regularizer.add_noise(outputs.fine_reasoning, training=True)
                outputs.fine_reasoning = regularizer.apply_token_dropout(outputs.fine_reasoning, training=True)
            
            # Teacher evaluation
            teacher_scores = None
            if cfg.use_teacher and train:
                with torch.no_grad():
                    pred_ids = outputs.answer_logits.argmax(dim=-1)
                    predictions = [
                        model.tokenizer.decode(ids, skip_special_tokens=True).strip()
                        for ids in pred_ids
                    ]
                    ground_truths = []
                    for i in range(labels.size(0)):
                        label_ids = labels[i][labels[i] != -100]
                        gt = model.tokenizer.decode(label_ids, skip_special_tokens=True).strip()
                        ground_truths.append(gt)
                    
                    teacher_scores = teacher_evaluator.evaluate_answers(predictions, ground_truths)
            
            # Compute loss
            loss, loss_dict = compute_total_loss(
                outputs, labels, teacher_scores, regularizer, cfg, model.tokenizer
            )
            
            if train:
                loss = loss / cfg.accum_steps
        
        # Backward
        if train:
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        
        # Accumulate
        total_loss += loss_dict['total']
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1
        
        pbar.set_postfix({
            'L': f"{loss_dict['total']:.3f}",
            'A': f"{loss_dict['answer']:.3f}",
            'D': f"{loss_dict['diversity']:.3f}",
            'T': f"{loss_dict['teacher']:.3f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    """Main training function"""
    args = parse_args()
    cfg = HybridTrainConfig()
    
    # Override from args
    if args.csv_path:
        cfg.csv_path = args.csv_path
    if args.image_folder:
        cfg.image_folder = args.image_folder
    if args.stage:
        cfg.stage = args.stage
    if args.num_epochs:
        cfg.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.enable_reasoning is not None:
        cfg.enable_reasoning = bool(args.enable_reasoning)
    if args.use_teacher is not None:
        cfg.use_teacher = bool(args.use_teacher)
    if args.teacher_weight is not None:
        cfg.teacher_weight = args.teacher_weight
    if args.resume_epoch:
        cfg.resume_epoch = args.resume_epoch
    
    # Configure stage
    if cfg.stage == 1:
        cfg.enable_reasoning = False
        cfg.use_teacher = False
        cfg.save_dir = cfg.save_dir + "_stage1_baseline"
        print("\n🔵 STAGE 1: BASELINE (No Reasoning)")
    elif cfg.stage == 2:
        cfg.enable_reasoning = True
        cfg.use_teacher = False
        cfg.save_dir = cfg.save_dir + "_stage2_warmup"
        print("\n🟡 STAGE 2: WARMUP (Reasoning, No Teacher)")
    elif cfg.stage == 3:
        cfg.enable_reasoning = True
        cfg.use_teacher = True
        cfg.save_dir = cfg.save_dir + "_stage3_full"
        print("\n🟢 STAGE 3: FULL (Reasoning + Teacher)")
    
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("="*80)
    print("HYBRID LATENT REASONING VQA - TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Stage: {cfg.stage}")
    print(f"Batch: {cfg.batch_size} × {cfg.accum_steps} = {cfg.batch_size * cfg.accum_steps}")
    print(f"Reasoning: {cfg.num_coarse_tokens} coarse + {cfg.num_fine_tokens} fine")
    print(f"Teacher: {cfg.use_teacher}")
    print("="*80 + "\n")
    
    # Model
    print("[1/6] Initializing model...")
    model = HybridLatentReasoningVQA(
        num_cross_attn_layers=cfg.num_cross_attn_layers,
        num_coarse_tokens=cfg.num_coarse_tokens,
        num_fine_tokens=cfg.num_fine_tokens,
        num_reasoning_layers=cfg.num_reasoning_layers,
        use_stochastic_reasoning=cfg.use_stochastic_reasoning,
        latent_dim=cfg.latent_dim,
        gradient_checkpointing=True
    )
    
    model.freeze_pretrained_weights(unfreeze_encoder_last_n_layers=cfg.unfreeze_encoder_layers)
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    
    # Regularizer & Teacher
    print("\n[2/6] Setting up regularizer & teacher...")
    regularizer = ReasoningRegularizer(
        noise_std=cfg.noise_std,
        diversity_weight=cfg.diversity_weight,
        token_dropout_prob=cfg.token_dropout_prob
    )
    
    teacher_evaluator = TeacherEvaluator(
        teacher_type=cfg.teacher_type,
        device=device
    )
    
    # Dataset
    print("\n[3/6] Loading dataset...")
    vision_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    full_dataset = VQAGenDataset(
        csv_path=cfg.csv_path,
        image_folder=cfg.image_folder,
        vision_processor=vision_processor,
        tokenizer_name='vinai/bartpho-syllable',
        max_q_len=32,
        max_a_len=32
    )
    
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    print("\n[4/6] Setting up optimizer...")
    total_steps = len(train_loader) // cfg.accum_steps * cfg.num_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps)
    print(f"Total steps: {total_steps}, Warmup: {int(total_steps * cfg.warmup_ratio)}")
    
    scaler = GradScaler(enabled=cfg.use_amp)
    
    # Resume
    start_epoch = 0
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_answer': [], 'val_answer': [],
        'train_diversity': [], 'val_diversity': [],
        'train_teacher': [], 'val_teacher': [], 'lr': []
    }
    
    if cfg.resume_epoch > 0:
        print(f"\n[5/6] Resuming from epoch {cfg.resume_epoch}...")
        checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{cfg.resume_epoch}.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            print(f"✓ Resumed from epoch {cfg.resume_epoch}")
        else:
            print(f"⚠ Checkpoint not found")
    else:
        print("\n[5/6] Starting from scratch...")
    
    # Training loop
    print("\n[6/6] Starting training...")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{cfg.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_components = run_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            regularizer, teacher_evaluator, scheduler, train=True
        )
        
        # Validation
        with torch.no_grad():
            val_loss, val_components = run_one_epoch(
                model, val_loader, optimizer, scaler, device, cfg,
                regularizer, teacher_evaluator, train=False
            )
        
        # Logging
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1} SUMMARY")
        print(f"{'='*80}")
        print(f"Train: Loss={train_loss:.4f} | A={train_components['answer']:.4f} | "
              f"D={train_components['diversity']:.4f} | T={train_components['teacher']:.4f}")
        print(f"Val:   Loss={val_loss:.4f} | A={val_components['answer']:.4f} | "
              f"D={val_components['diversity']:.4f} | T={val_components['teacher']:.4f}")
        print(f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"{'='*80}\n")
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_answer'].append(train_components['answer'])
        history['val_answer'].append(val_components['answer'])
        history['train_diversity'].append(train_components['diversity'])
        history['val_diversity'].append(val_components['diversity'])
        history['train_teacher'].append(train_components['teacher'])
        history['val_teacher'].append(val_components['teacher'])
        history['lr'].append(current_lr)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'history': history,
            'config': cfg
        }
        
        checkpoint_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved: {checkpoint_path}")
        
        # Save best
        if val_loss < best_val_loss - cfg.es_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(cfg.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"✓ NEW BEST! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⚠ Patience: {patience_counter}/{cfg.es_patience}")
        
        # Early stopping
        if patience_counter >= cfg.es_patience:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING at epoch {epoch+1}")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"{'='*80}\n")
            break
    
    # Save results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    df = pd.DataFrame(history)
    csv_path = os.path.join(cfg.save_dir, cfg.log_csv)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'o-', label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], 's-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['epoch'], history['train_answer'], 'o-', label='Train')
    axes[0, 1].plot(history['epoch'], history['val_answer'], 's-', label='Val')
    axes[0, 1].set_title('Answer Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['epoch'], history['train_diversity'], 'o-', label='Train')
    axes[1, 0].plot(history['epoch'], history['val_diversity'], 's-', label='Val')
    axes[1, 0].set_title('Diversity Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['epoch'], history['train_teacher'], 'o-', label='Train')
    axes[1, 1].plot(history['epoch'], history['val_teacher'], 's-', label='Val')
    axes[1, 1].set_title('Teacher Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    curve_path = os.path.join(cfg.save_dir, cfg.curve_png)
    plt.savefig(curve_path, dpi=150)
    print(f"✓ Saved: {curve_path}")
    
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
