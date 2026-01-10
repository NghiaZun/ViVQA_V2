"""
IMPROVED LATENT REASONING DISTILLATION VQA MODEL
=================================================
Cải tiến dựa trên proposal gốc với các improvements:
1. Auxiliary tasks (visual grounding, reasoning type)
2. Contrastive learning cho reasoning latents
3. Hierarchical reasoning (coarse + fine)
4. Better regularization
5. Interpretability tools

Architecture:
1. Vision Encoder (DINOv2) - Extract visual features
2. Text Encoder (BARTpho encoder) - Encode question
3. Multimodal Fusion (Gated Cross-Attention) - Align vision + text
4. Hierarchical Latent Reasoning Module - Generate R_coarse và R_fine
5. Auxiliary Heads - Visual grounding, reasoning type classification
6. Answer Decoder (BARTpho decoder) - Generate answer conditioned on R

Total: ~490M params (student) + improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoImageProcessor,
    BartphoTokenizer,
    MBartForConditionalGeneration
)
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math


# ============================================================================
# HELPER FUNCTIONS (từ model_dinov2_bartpho_2.py)
# ============================================================================

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift tokens right for teacher forcing"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class GatedCrossAttentionLayer(nn.Module):
    """Gated cross-attention với residual connection"""
    def __init__(self, hidden_dim=1024, num_heads=16, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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
        # Cross-attention: text queries, visual keys/values
        attn_out, attn_weights = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Gating mechanism
        gate_input = torch.cat([text_features, attn_out], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        gated_output = gate_values * attn_out + (1 - gate_values) * text_features
        
        # Residual + norm
        x = self.norm1(text_features + gated_output)
        
        # FFN
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


# ============================================================================
# AUXILIARY HEADS
# ============================================================================

class VisualGroundingHead(nn.Module):
    """
    Auxiliary task: Predict which image regions are relevant
    Helps reasoning latents learn to attend to important visual areas
    """
    def __init__(self, hidden_dim=1024, num_patches=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, reasoning_latents, visual_features):
        """
        Args:
            reasoning_latents: [batch, num_tokens, hidden_dim]
            visual_features: [batch, num_patches, hidden_dim]
        Returns:
            attention_logits: [batch, num_tokens, num_patches]
        """
        # Compute attention scores between reasoning tokens and visual patches
        # [batch, num_tokens, hidden] @ [batch, hidden, num_patches]
        attention_logits = torch.matmul(
            reasoning_latents,
            visual_features.transpose(-1, -2)
        ) / math.sqrt(reasoning_latents.size(-1))
        
        return attention_logits


class ReasoningTypeHead(nn.Module):
    """
    Auxiliary task: Classify reasoning type
    Types: counting, comparison, color, spatial, etc.
    """
    def __init__(self, hidden_dim=1024, num_types=8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_types)
        )
    
    def forward(self, reasoning_latents):
        """
        Args:
            reasoning_latents: [batch, num_tokens, hidden_dim]
        Returns:
            type_logits: [batch, num_types]
        """
        # Pool reasoning tokens (mean pooling)
        pooled = reasoning_latents.mean(dim=1)  # [batch, hidden_dim]
        type_logits = self.classifier(pooled)
        return type_logits


# ============================================================================
# IMPROVED LATENT REASONING MODULE
# ============================================================================

class LatentReasoningModule(nn.Module):
    """
    Improved Latent Reasoning Module với:
    - VAE-style stochastic sampling
    - Learnable reasoning queries
    - Cross-attention over multimodal features
    - Optional dropout regularization
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_reasoning_tokens: int = 16,  # Tăng từ 12 lên 16
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_stochastic: bool = True,
        latent_dim: int = 512,
        reasoning_dropout: float = 0.1  # Dropout on reasoning latents
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_reasoning_tokens = num_reasoning_tokens
        self.use_stochastic = use_stochastic
        self.reasoning_dropout = reasoning_dropout
        
        # Learnable reasoning queries
        self.reasoning_queries = nn.Parameter(
            torch.randn(num_reasoning_tokens, hidden_dim) * 0.02
        )
        
        # Cross-attention layers
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
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
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        multimodal_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate latent reasoning representation
        
        Returns:
            reasoning_latents: [batch, num_reasoning_tokens, hidden_dim]
            kl_loss: KL divergence loss
        """
        batch_size = multimodal_features.size(0)
        
        # Expand reasoning queries
        reasoning_queries = self.reasoning_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Cross-attend over multimodal features
        for layer in self.reasoning_layers:
            reasoning_queries = layer(
                tgt=reasoning_queries,
                memory=multimodal_features,
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
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_size
            
            reasoning_queries = self.from_latent(z)
        
        # Output normalization
        reasoning_latents = self.output_norm(reasoning_queries)
        
        # Dropout regularization during training
        if self.training and self.reasoning_dropout > 0:
            reasoning_latents = F.dropout(
                reasoning_latents, 
                p=self.reasoning_dropout, 
                training=True
            )
        
        return reasoning_latents, kl_loss


class HierarchicalReasoningModule(nn.Module):
    """
    Hierarchical reasoning: Coarse (global) + Fine (detailed)
    Coarse reasoning captures high-level patterns
    Fine reasoning captures specific details
    """
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
        
        # Coarse reasoning (fewer tokens, high-level)
        self.coarse_reasoning = LatentReasoningModule(
            hidden_dim=hidden_dim,
            num_reasoning_tokens=num_coarse_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_stochastic=use_stochastic,
            latent_dim=latent_dim
        )
        
        # Fine reasoning (more tokens, detailed)
        # Conditioned on coarse reasoning
        self.fine_reasoning = LatentReasoningModule(
            hidden_dim=hidden_dim,
            num_reasoning_tokens=num_fine_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_stochastic=use_stochastic,
            latent_dim=latent_dim
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(
        self,
        multimodal_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            coarse_reasoning: [batch, num_coarse_tokens, hidden]
            fine_reasoning: [batch, num_fine_tokens, hidden]
            total_kl_loss: Combined KL loss
        """
        # Coarse reasoning
        coarse_r, kl_coarse = self.coarse_reasoning(
            multimodal_features, attention_mask, deterministic
        )
        
        # Concatenate coarse reasoning with original features
        enhanced_features = torch.cat([
            multimodal_features,
            coarse_r
        ], dim=1)
        
        # Extended attention mask
        if attention_mask is not None:
            coarse_mask = torch.ones(
                coarse_r.size(0), coarse_r.size(1),
                device=attention_mask.device
            )
            enhanced_mask = torch.cat([attention_mask, coarse_mask], dim=1)
        else:
            enhanced_mask = None
        
        # Fine reasoning conditioned on coarse
        fine_r, kl_fine = self.fine_reasoning(
            enhanced_features, enhanced_mask, deterministic
        )
        
        # Total KL loss
        total_kl = None
        if kl_coarse is not None and kl_fine is not None:
            total_kl = kl_coarse + kl_fine
        
        return coarse_r, fine_r, total_kl


# ============================================================================
# OUTPUT DATACLASS
# ============================================================================

@dataclass
class ImprovedLatentReasoningVQAOutput:
    """Output structure với auxiliary losses"""
    answer_logits: torch.Tensor
    coarse_reasoning: torch.Tensor
    fine_reasoning: torch.Tensor
    
    # Losses
    answer_loss: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None
    visual_grounding_loss: Optional[torch.Tensor] = None
    reasoning_type_loss: Optional[torch.Tensor] = None
    contrastive_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    
    # For visualization
    visual_attention: Optional[torch.Tensor] = None
    reasoning_type_logits: Optional[torch.Tensor] = None


# ============================================================================
# MAIN MODEL
# ============================================================================

class ImprovedLatentReasoningVQA(nn.Module):
    """
    Improved VQA Model với:
    1. Hierarchical reasoning (coarse + fine)
    2. Auxiliary tasks (visual grounding, reasoning type)
    3. Contrastive learning ready
    4. Better regularization
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
        num_reasoning_types: int = 8,  # Number of reasoning type classes
        use_auxiliary_tasks: bool = True,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        print("[INFO] Initializing Improved Latent Reasoning VQA Model")
        print(f"  Vision: {dinov2_model_name}")
        print(f"  Language: {bartpho_model_name}")
        print(f"  Coarse tokens: {num_coarse_tokens}, Fine tokens: {num_fine_tokens}")
        print(f"  Auxiliary tasks: {use_auxiliary_tasks}")
        
        self.use_auxiliary_tasks = use_auxiliary_tasks
        
        # === VISION ENCODER ===
        self.vision_encoder = AutoModel.from_pretrained(dinov2_model_name)
        self.vision_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        vision_hidden_dim = self.vision_encoder.config.hidden_size
        
        # === LANGUAGE MODEL ===
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
        
        # === DIMENSION ALIGNMENT ===
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_dim, bart_hidden_dim),
            nn.LayerNorm(bart_hidden_dim),
            nn.Dropout(dropout)
        )
        
        # === MULTIMODAL FUSION ===
        self.cross_attention_fusion = MultiLayerCrossAttention(
            hidden_dim=bart_hidden_dim,
            num_layers=num_cross_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # === HIERARCHICAL REASONING MODULE ===
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
        
        # === AUXILIARY HEADS ===
        if use_auxiliary_tasks:
            # Get num_patches from vision encoder
            # DINOv2-base: 224/14 = 16, so 16x16 = 256 patches
            num_patches = (self.vision_encoder.config.image_size // 
                          self.vision_encoder.config.patch_size) ** 2
            
            self.visual_grounding_head = VisualGroundingHead(
                hidden_dim=bart_hidden_dim,
                num_patches=num_patches
            )
            
            self.reasoning_type_head = ReasoningTypeHead(
                hidden_dim=bart_hidden_dim,
                num_types=num_reasoning_types
            )
        
        # === GRADIENT CHECKPOINTING ===
        if gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()
            self.encoder.gradient_checkpointing_enable()
            self.decoder.gradient_checkpointing_enable()
            print("[INFO] ✓ Gradient checkpointing enabled")
        
        print(f"[INFO] ✓ Improved Latent Reasoning VQA initialized")
    
    def freeze_pretrained_weights(self, unfreeze_encoder_last_n_layers: int = 3):
        """Freeze pretrained, keep reasoning + auxiliary trainable"""
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
        
        aux_params = 0
        if self.use_auxiliary_tasks:
            aux_params += sum(p.numel() for p in self.visual_grounding_head.parameters())
            aux_params += sum(p.numel() for p in self.reasoning_type_head.parameters())
        
        lmhead_params = sum(p.numel() for p in self.lm_head.parameters())
        
        print(f"  ✅ Vision Projection: {proj_params/1e6:.1f}M")
        print(f"  ✅ Cross-Attention Fusion: {fusion_params/1e6:.1f}M")
        print(f"  ✅ Hierarchical Reasoning: {reasoning_params/1e6:.1f}M")
        if self.use_auxiliary_tasks:
            print(f"  ✅ Auxiliary Heads: {aux_params/1e6:.1f}M")
        print(f"  ✅ Encoder last {unfreeze_encoder_last_n_layers} layers: {unfrozen_encoder_params/1e6:.1f}M")
        print(f"  ✅ LM Head: {lmhead_params/1e6:.1f}M")
        
        total_trainable = (proj_params + fusion_params + reasoning_params + 
                          aux_params + unfrozen_encoder_params + lmhead_params)
        print(f"\n  📊 Total trainable: {total_trainable/1e6:.1f}M params")
    
    def encode_image(self, pixel_values):
        """Encode image with DINOv2"""
        outputs = self.vision_encoder(pixel_values, return_dict=True)
        visual_embeds = outputs.last_hidden_state
        visual_features = self.vision_proj(visual_embeds)
        return visual_features
    
    def encode_text(self, input_ids, attention_mask):
        """Encode question with BARTpho encoder"""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return encoder_outputs.last_hidden_state
    
    def fuse_multimodal(self, text_features, visual_features):
        """Fuse with gated cross-attention"""
        fused_features, attention_weights = self.cross_attention_fusion(
            text_features=text_features,
            visual_features=visual_features
        )
        return fused_features, attention_weights
    
    def compute_visual_grounding_loss(
        self, 
        reasoning_latents: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute visual grounding loss
        Encourage reasoning to attend to relevant visual regions
        Simple heuristic: Maximize attention diversity (avoid collapse)
        """
        # Compute attention
        attention_logits = self.visual_grounding_head(
            reasoning_latents, visual_features
        )  # [batch, num_tokens, num_patches]
        
        attention_probs = F.softmax(attention_logits, dim=-1)
        
        # Diversity loss: Encourage different tokens to attend to different patches
        # Compute entropy of attention distribution
        entropy = -(attention_probs * torch.log(attention_probs + 1e-10)).sum(dim=-1)
        entropy_loss = -entropy.mean()  # Maximize entropy
        
        # Uniformity loss: Encourage all patches to be attended to
        patch_attention = attention_probs.mean(dim=1)  # [batch, num_patches]
        uniformity_loss = F.mse_loss(
            patch_attention, 
            torch.ones_like(patch_attention) / patch_attention.size(-1)
        )
        
        total_loss = entropy_loss + 0.1 * uniformity_loss
        return total_loss
    
    def compute_reasoning_type_loss(
        self,
        reasoning_latents: torch.Tensor,
        reasoning_type_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reasoning type classification loss
        If labels not available, use pseudo-labels from question features
        """
        type_logits = self.reasoning_type_head(reasoning_latents)
        
        if reasoning_type_labels is not None:
            # Supervised loss
            loss = F.cross_entropy(type_logits, reasoning_type_labels)
        else:
            # Unsupervised: Maximize prediction confidence (encourage specialization)
            probs = F.softmax(type_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            loss = entropy.mean()  # Minimize entropy = maximize confidence
        
        return loss
    
    def compute_contrastive_loss(
        self,
        reasoning_latents: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Contrastive loss: Similar answers should have similar reasoning
        InfoNCE-style contrastive learning
        """
        batch_size = reasoning_latents.size(0)
        
        # Pool reasoning latents
        pooled_r = reasoning_latents.mean(dim=1)  # [batch, hidden]
        
        # Normalize
        pooled_r = F.normalize(pooled_r, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(pooled_r, pooled_r.T) / temperature
        
        # Create positive pairs: same answer = positive
        # Simple heuristic: compare first token of answer
        answer_first_token = labels[:, 0]  # [batch]
        pos_mask = (answer_first_token.unsqueeze(0) == answer_first_token.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # Exclude self
        
        # Contrastive loss
        exp_sim = torch.exp(sim_matrix)
        
        # Sum over positive pairs
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        
        # Sum over all pairs (exclude self)
        all_sim = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix))
        
        # Loss
        loss = -torch.log(pos_sim / (all_sim + 1e-10) + 1e-10)
        loss = loss[pos_mask.sum(dim=1) > 0].mean()  # Only where positive pairs exist
        
        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=loss.device)
    
    def generate_answer_with_reasoning(
        self,
        fused_features: torch.Tensor,
        coarse_reasoning: torch.Tensor,
        fine_reasoning: torch.Tensor,
        answer_input_ids: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None
    ):
        """Generate answer conditioned on hierarchical reasoning"""
        # Concatenate all reasoning
        encoder_hidden_states = torch.cat([
            fused_features,
            coarse_reasoning,
            fine_reasoning
        ], dim=1)
        
        # Shift decoder inputs
        if answer_input_ids is not None:
            decoder_input_ids = shift_tokens_right(
                answer_input_ids,
                self.config.pad_token_id,
                self.config.decoder_start_token_id
            )
        else:
            decoder_input_ids = answer_input_ids
        
        # Decoder forward
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=answer_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            use_cache=False
        )
        
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        return logits, hidden_states
    
    def compute_answer_loss(self, logits, labels):
        """Cross-entropy loss for answer generation"""
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reasoning_type_labels: Optional[torch.Tensor] = None,
        deterministic_reasoning: bool = False,
        # Loss weights
        kl_weight: float = 0.01,
        visual_grounding_weight: float = 0.05,
        reasoning_type_weight: float = 0.02,
        contrastive_weight: float = 0.05
    ):
        """
        Forward pass with auxiliary tasks
        
        Returns:
            ImprovedLatentReasoningVQAOutput with all losses
        """
        # 1. Encode vision
        visual_features = self.encode_image(pixel_values)
        
        # 2. Encode text
        text_features = self.encode_text(input_ids, attention_mask)
        
        # 3. Fuse multimodal
        fused_features, _ = self.fuse_multimodal(text_features, visual_features)
        
        # 4. Generate hierarchical reasoning
        coarse_r, fine_r, kl_loss = self.hierarchical_reasoning(
            multimodal_features=fused_features,
            attention_mask=attention_mask,
            deterministic=deterministic_reasoning
        )
        
        # 5. Generate answer
        logits, _ = self.generate_answer_with_reasoning(
            fused_features=fused_features,
            coarse_reasoning=coarse_r,
            fine_reasoning=fine_r,
            answer_input_ids=labels,
            answer_attention_mask=(labels != self.config.pad_token_id) if labels is not None else None
        )
        
        # 6. Compute losses
        answer_loss = None
        visual_grounding_loss = None
        reasoning_type_loss = None
        contrastive_loss = None
        total_loss = None
        
        visual_attention = None
        reasoning_type_logits = None
        
        if labels is not None:
            # Main answer loss
            answer_loss = self.compute_answer_loss(logits, labels)
            total_loss = answer_loss
            
            # KL loss
            if kl_loss is not None:
                total_loss = total_loss + kl_weight * kl_loss
            
            # Auxiliary tasks
            if self.use_auxiliary_tasks and self.training:
                # Visual grounding
                visual_grounding_loss = self.compute_visual_grounding_loss(
                    fine_r, visual_features
                )
                total_loss = total_loss + visual_grounding_weight * visual_grounding_loss
                
                # Reasoning type
                reasoning_type_loss = self.compute_reasoning_type_loss(
                    fine_r, reasoning_type_labels
                )
                total_loss = total_loss + reasoning_type_weight * reasoning_type_loss
                
                # Contrastive loss
                contrastive_loss = self.compute_contrastive_loss(
                    fine_r, labels
                )
                total_loss = total_loss + contrastive_weight * contrastive_loss
                
                # Get attention for visualization
                visual_attention = self.visual_grounding_head(fine_r, visual_features)
                reasoning_type_logits = self.reasoning_type_head(fine_r)
        
        return ImprovedLatentReasoningVQAOutput(
            answer_logits=logits,
            coarse_reasoning=coarse_r,
            fine_reasoning=fine_r,
            answer_loss=answer_loss,
            kl_loss=kl_loss,
            visual_grounding_loss=visual_grounding_loss,
            reasoning_type_loss=reasoning_type_loss,
            contrastive_loss=contrastive_loss,
            total_loss=total_loss,
            visual_attention=visual_attention,
            reasoning_type_logits=reasoning_type_logits
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 32,
        num_beams: int = 4,
        num_reasoning_samples: int = 1,
        **generation_kwargs
    ) -> List[str]:
        """Inference generation"""
        # Encode
        visual_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        fused_features, _ = self.fuse_multimodal(text_features, visual_features)
        
        batch_size = fused_features.size(0)
        all_answers = []
        
        for _ in range(num_reasoning_samples):
            # Generate reasoning
            coarse_r, fine_r, _ = self.hierarchical_reasoning(
                multimodal_features=fused_features,
                attention_mask=attention_mask,
                deterministic=(num_reasoning_samples == 1)
            )
            
            # Concatenate reasoning
            encoder_hidden_states = torch.cat([
                fused_features,
                coarse_r,
                fine_r
            ], dim=1)
            
            # BOS tokens
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=fused_features.device
            )
            
            # Generate
            generated_ids = self.decoder.generate(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                max_length=max_length,
                num_beams=num_beams,
                pad_token_id=self.config.pad_token_id,
                eos_token_id=self.config.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                use_cache=True,
                **generation_kwargs
            )
            
            # Decode
            answers = [
                self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                for ids in generated_ids
            ]
            all_answers.append(answers)
        
        return all_answers[0]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing Improved Latent Reasoning VQA Model...")
    
    model = ImprovedLatentReasoningVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        num_coarse_tokens=8,
        num_fine_tokens=16,
        num_reasoning_layers=2,
        use_stochastic_reasoning=True,
        latent_dim=512,
        use_auxiliary_tasks=True,
        gradient_checkpointing=True
    )
    
    model.freeze_pretrained_weights(unfreeze_encoder_last_n_layers=3)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
    
    print("\n[INFO] Testing forward pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
    attention_mask = torch.ones(batch_size, 32).to(device)
    labels = torch.randint(0, 1000, (batch_size, 16)).to(device)
    
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"[INFO] Answer logits: {outputs.answer_logits.shape}")
    print(f"[INFO] Coarse reasoning: {outputs.coarse_reasoning.shape}")
    print(f"[INFO] Fine reasoning: {outputs.fine_reasoning.shape}")
    print(f"[INFO] Total loss: {outputs.total_loss.item() if outputs.total_loss is not None else 'N/A'}")
    if outputs.visual_grounding_loss:
        print(f"[INFO] Visual grounding loss: {outputs.visual_grounding_loss.item()}")
    if outputs.reasoning_type_loss:
        print(f"[INFO] Reasoning type loss: {outputs.reasoning_type_loss.item()}")
    if outputs.contrastive_loss:
        print(f"[INFO] Contrastive loss: {outputs.contrastive_loss.item()}")
    
    print("\n[SUCCESS] Model test passed! ✓")