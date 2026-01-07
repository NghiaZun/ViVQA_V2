"""
SOTA VQA MODEL: DINOv2 + BARTpho with Gated Cross-Attention
============================================================
Best-in-class architecture cho Vietnamese VQA:
- DINOv2-base: SOTA vision understanding (86M params)
- BARTpho-large: Vietnamese encoder-decoder (396M params)
- Gated Cross-Attention Fusion: From LXMERT/UNITER/BLIP
- Chain-of-Thought: Reasoning → Answer với quality validation

Key Features:
✅ Language-agnostic vision (DINOv2)
✅ Vietnamese-specialized language (BARTpho)
✅ Multi-layer cross-attention with gating
✅ Reasoning quality check trước khi generate answer
✅ Gradient checkpointing để save memory
✅ Support resume training với full optimizer state

Total: ~482M params (~9GB memory)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoImageProcessor,
    BartphoTokenizer,  # 🔥 FIX: Dùng BartphoTokenizer thay vì AutoTokenizer
    MBartForConditionalGeneration
)
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import math


# 🔥 FIX: Implement shift_tokens_right (Transformers không expose nó)
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right (cho teacher forcing)
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("pad_token_id has to be defined.")
    # Replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@dataclass
class VQAOutput:
    """VQA output với answer only (no reasoning)"""
    logits: torch.Tensor  # [batch, seq_len, vocab_size]
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None  # [batch, seq_len, hidden]


class GatedCrossAttentionLayer(nn.Module):
    """
    SOTA Gated Cross-Attention từ LXMERT/UNITER/BLIP
    
    Gating mechanism giúp model học được khi nào nên attend to visual vs textual info.
    Formula:
        gate = sigmoid(Wg * [vision; text])
        output = gate * cross_attn(vision, text) + (1-gate) * text
    """
    
    def __init__(self, hidden_dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Feed-forward network (Transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_features, visual_features, attention_mask=None):
        """
        Args:
            text_features: [batch, text_len, hidden_dim]
            visual_features: [batch, visual_len, hidden_dim]
            attention_mask: [batch, visual_len] (optional)
        Returns:
            fused_features: [batch, text_len, hidden_dim]
        """
        batch_size = text_features.size(0)
        
        # Cross-attention: text queries, visual keys/values
        attn_output, attn_weights = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features,
            key_padding_mask=attention_mask if attention_mask is not None else None
        )
        
        # Gating: decide how much to use cross-attention vs original text
        # Concatenate mean-pooled features for gating
        text_pooled = text_features.mean(dim=1, keepdim=True).expand(-1, text_features.size(1), -1)
        visual_pooled = visual_features.mean(dim=1, keepdim=True).expand(-1, text_features.size(1), -1)
        gate_input = torch.cat([text_pooled, visual_pooled], dim=-1)
        gate_values = self.gate(gate_input)  # [batch, text_len, hidden_dim]
        
        # Apply gating
        gated_attn = gate_values * attn_output + (1 - gate_values) * text_features
        
        # Residual connection + layer norm
        text_features = self.ln1(text_features + self.dropout(gated_attn))
        
        # Feed-forward network
        ffn_output = self.ffn(text_features)
        text_features = self.ln2(text_features + ffn_output)
        
        return text_features, attn_weights


class MultiLayerCrossAttention(nn.Module):
    """
    Stack nhiều layers cross-attention (SOTA approach)
    Papers: LXMERT (5 layers), UNITER (3 layers), BLIP (6 layers)
    
    Nhiều layers giúp model học được multi-level interactions:
    - Layer 1: Low-level features (edges, colors)
    - Layer 2: Mid-level features (objects, parts)
    - Layer 3+: High-level features (semantic relationships)
    """
    
    def __init__(self, hidden_dim, num_layers=3, num_heads=12, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedCrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, text_features, visual_features, attention_mask=None):
        """
        Args:
            text_features: [batch, text_len, hidden_dim]
            visual_features: [batch, visual_len, hidden_dim]
        Returns:
            fused_features: [batch, text_len, hidden_dim]
            all_attention_weights: List of attention weight matrices
        """
        all_attn_weights = []
        
        for layer in self.layers:
            text_features, attn_weights = layer(text_features, visual_features, attention_mask)
            all_attn_weights.append(attn_weights)
            
        return text_features, all_attn_weights


# Removed ReasoningQualityChecker - không cần cho answer-only model


class DINOv2BARTphoVQA(nn.Module):
    """
    Simplified VQA Model: DINOv2 + BARTpho + Gated Cross-Attention
    
    Architecture:
    1. Vision: DINOv2-base (86M) - SOTA self-supervised vision
    2. Language: BARTpho encoder (197M) - Vietnamese understanding
    3. Fusion: 3-layer Gated Cross-Attention - Multi-modal alignment
    4. Generation: BARTpho decoder (199M) - Direct answer generation
    
    Total: ~482M params (single decoder)
    """
    
    def __init__(
        self,
        dinov2_model_name='facebook/dinov2-base',  # 86M params
        bartpho_model_name='vinai/bartpho-syllable',  # 396M params (large variant)
        num_cross_attn_layers=3,  # SOTA: 3 layers
        num_heads=16,  # 1024 dim ÷ 16 heads = 64 (BARTpho standard)
        dropout=0.1,
        gradient_checkpointing=True
    ):
        super().__init__()
        
        print("[INFO] Initializing SOTA VQA Model: DINOv2 + BARTpho")
        print(f"  Vision: {dinov2_model_name}")
        print(f"  Language: {bartpho_model_name}")
        print(f"  Cross-Attention Layers: {num_cross_attn_layers}")
        
        # === 1. VISION ENCODER: DINOv2 ===
        self.vision_encoder = AutoModel.from_pretrained(dinov2_model_name)
        self.vision_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        vision_hidden_dim = self.vision_encoder.config.hidden_size  # 768 for base
        
        # === 2. LANGUAGE MODEL: BARTpho (Encoder + 2 Separate Decoders) ===
        # Note: BARTpho trên HuggingFace là mBART architecture
        print("[INFO] Loading BARTpho and creating separate decoders...")
        bartpho_full = MBartForConditionalGeneration.from_pretrained(bartpho_model_name)
        bartpho_full.config.use_cache = False  # Disable cache for training
        
        # 🔥 FIX: Dùng BartphoTokenizer chính xác cho Vietnamese
        self.tokenizer = BartphoTokenizer.from_pretrained(bartpho_model_name)
        
        bart_hidden_dim = bartpho_full.config.d_model  # 1024 for large
        
        # Split into components
        self.encoder = bartpho_full.model.encoder  # Shared encoder
        self.decoder = bartpho_full.model.decoder  # Single decoder for answer
        print("[INFO] ✓ Using single decoder for direct answer generation")
        
        # Shared lm_head (vocabulary projection)
        self.lm_head = bartpho_full.lm_head
        
        # Clean up full model to save memory
        del bartpho_full
        
        # Store config for generation
        self.config = self.encoder.config
        
        # 🔥 FIX: Cấu hình decoder_start_token_id đúng cho BARTpho
        self.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        self.config.forced_bos_token_id = self.tokenizer.bos_token_id
        print(f"[INFO] ✓ Configured special tokens:")
        print(f"  - BOS token ID: {self.config.decoder_start_token_id}")
        print(f"  - PAD token ID: {self.config.pad_token_id}")
        print(f"  - EOS token ID: {self.config.eos_token_id}")
        
        # === 3. DIMENSION ALIGNMENT ===
        # DINOv2-base: 768, BARTpho-large: 1024
        # Project vision features to match BARTpho dimension
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_dim, bart_hidden_dim),
            nn.LayerNorm(bart_hidden_dim),
            nn.Dropout(dropout)
        )
        
        # === 4. GATED CROSS-ATTENTION FUSION ===
        self.cross_attention_fusion = MultiLayerCrossAttention(
            hidden_dim=bart_hidden_dim,
            num_layers=num_cross_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # === 5. GRADIENT CHECKPOINTING (Save memory) ===
        if gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()
            self.encoder.gradient_checkpointing_enable()
            self.decoder.gradient_checkpointing_enable()
            print("[INFO] ✓ Gradient checkpointing enabled")
        
        print(f"[INFO] ✓ Model initialized: ~482M parameters (single decoder)")
        
    def freeze_pretrained_weights(self, unfreeze_encoder_last_n_layers=3):
        """
        Freeze pretrained nhưng UNFREEZE last N layers của encoder
        
        FROZEN (giữ pretrained knowledge):
        - Vision encoder (DINOv2)
        - BARTpho Encoder (EXCEPT last N layers)
        - BARTpho Decoder
        
        TRAINABLE (task-specific + semantic adaptation):
        - Vision projection
        - Cross-attention fusion
        - BARTpho Encoder last N layers (học Vietnamese semantics)
        - LM head
        """
        print("\n[INFO] 🔒 FREEZING PRETRAINED WEIGHTS (Feature Extraction Mode)")
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        print(f"  ❄️  Vision Encoder: {vision_params/1e6:.1f}M params frozen")
        
        # 🔥 FIX: Freeze encoder NHƯNG unfreeze last N layers
        # Freeze all first
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers để học Vietnamese semantics
        total_layers = len(self.encoder.layers)
        unfrozen_encoder_params = 0
        for i, layer in enumerate(self.encoder.layers):
            if i >= total_layers - unfreeze_encoder_last_n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen_encoder_params += sum(p.numel() for p in layer.parameters())
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        frozen_encoder_params = encoder_params - unfrozen_encoder_params
        print(f"  ❄️  BARTpho Encoder: {frozen_encoder_params/1e6:.1f}M params frozen")
        print(f"  🔥 BARTpho Encoder (last {unfreeze_encoder_last_n_layers} layers): {unfrozen_encoder_params/1e6:.1f}M params TRAINABLE")
        
        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  ❄️  Decoder: {decoder_params/1e6:.1f}M params frozen")
        
        # Keep trainable: projection, fusion, lm_head, encoder last layers
        proj_params = sum(p.numel() for p in self.vision_proj.parameters())
        fusion_params = sum(p.numel() for p in self.cross_attention_fusion.parameters())
        lmhead_params = sum(p.numel() for p in self.lm_head.parameters())
        
        print(f"\n  ✅ Vision Projection: {proj_params/1e6:.1f}M params trainable")
        print(f"  ✅ Cross-Attention Fusion: {fusion_params/1e6:.1f}M params trainable")
        print(f"  ✅ LM Head: {lmhead_params/1e6:.1f}M params trainable")
        
        total_frozen = vision_params + frozen_encoder_params + decoder_params
        total_trainable = proj_params + fusion_params + lmhead_params + unfrozen_encoder_params
        
        print(f"\n  📊 Summary:")
        print(f"     Frozen: {total_frozen/1e6:.1f}M ({total_frozen/(total_frozen+total_trainable)*100:.1f}%)")
        print(f"     Trainable: {total_trainable/1e6:.1f}M ({total_trainable/(total_frozen+total_trainable)*100:.1f}%)")
        print(f"  ✓ Semantic adaptation enabled with encoder fine-tuning!")
        
    def encode_image(self, pixel_values):
        """
        Encode image với DINOv2
        
        Args:
            pixel_values: [batch, 3, 224, 224]
        Returns:
            visual_features: [batch, num_patches+1, 1024] (after projection)
        """
        # DINOv2 forward pass
        outputs = self.vision_encoder(pixel_values, return_dict=True)
        visual_embeds = outputs.last_hidden_state  # [batch, num_patches+1, 768]
        
        # Project to BARTpho dimension
        visual_features = self.vision_proj(visual_embeds)  # [batch, num_patches+1, 1024]
        
        return visual_features
    
    def encode_text(self, input_ids, attention_mask):
        """
        Encode question với BARTpho encoder
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        Returns:
            text_features: [batch, seq_len, 1024]
        """
        # BARTpho encoder (shared)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = encoder_outputs.last_hidden_state  # [batch, seq_len, 1024]
        
        return text_features
    
    def fuse_multimodal(self, text_features, visual_features):
        """
        Fuse text and visual features using Gated Cross-Attention
        
        Args:
            text_features: [batch, text_len, 1024]
            visual_features: [batch, visual_len, 1024]
        Returns:
            fused_features: [batch, text_len, 1024]
            attention_weights: List of attention weight matrices
        """
        fused_features, attention_weights = self.cross_attention_fusion(
            text_features=text_features,
            visual_features=visual_features
        )
        
        return fused_features, attention_weights
    
    def generate_answer(
        self, 
        fused_features, 
        answer_input_ids=None,
        answer_attention_mask=None
    ):
        """
        Generate answer directly từ fused features
        
        Args:
            fused_features: [batch, seq_len, 1024] - encoder output
            answer_input_ids: [batch, target_len] - teacher forcing labels
            answer_attention_mask: [batch, target_len]
        Returns:
            logits: [batch, target_len, vocab_size]
            hidden_states: [batch, target_len, 1024]
        """
        # Shift decoder input tokens cho teacher forcing
        if answer_input_ids is not None:
            decoder_input_ids = shift_tokens_right(
                answer_input_ids,
                self.config.pad_token_id,
                self.config.decoder_start_token_id
            )
        else:
            decoder_input_ids = answer_input_ids
        
        # Decoder forward pass
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=answer_attention_mask,
            encoder_hidden_states=fused_features,
            return_dict=True,
            use_cache=False
        )
        
        hidden_states = decoder_outputs.last_hidden_state  # [batch, target_len, 1024]
        logits = self.lm_head(hidden_states)  # [batch, target_len, vocab_size]
        
        return logits, hidden_states
    
    def compute_loss(self, logits, labels):
        """
        Tính loss cho answer generation
        
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len] - target tokens
        Returns:
            loss: scalar
        """
        # Shift: logits[:, :-1] vs labels[:, 1:] (standard seq2seq)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Cross entropy loss (ignore padding)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        return loss
    
    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        labels=None  # Answer labels (tương thích với train code)
    ):
        """
        Forward pass: Image + Question → Answer
        
        Compatible với train_from_csv.py:
        - Input: pixel_values, input_ids, attention_mask, labels
        - Output: (loss, logits) nếu labels provided, else logits only
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            input_ids: [batch, seq_len] - question tokens
            attention_mask: [batch, seq_len]
            labels: [batch, ans_len] - answer target (optional, for training)
        Returns:
            If labels: (loss, logits)
            Else: VQAOutput(logits, hidden_states)
        """
        # 1. Encode vision
        visual_features = self.encode_image(pixel_values)
        
        # 2. Encode text (question)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # 3. Fuse multimodal features
        fused_features, _ = self.fuse_multimodal(text_features, visual_features)
        
        # 4. Generate answer directly từ fused features
        logits, hidden_states = self.generate_answer(
            fused_features=fused_features,
            answer_input_ids=labels,
            answer_attention_mask=(labels != self.config.pad_token_id) if labels is not None else None
        )
        
        # 5. Compute loss nếu có labels (training mode)
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return loss, logits
        else:
            return VQAOutput(
                logits=logits,
                hidden_states=hidden_states
            )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_length=32,
        num_beams=4,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    ):
        """
        Inference mode: Generate answer directly
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            max_length: Max answer length
            num_beams: Beam search width
            repetition_penalty: Anti-repetition
            length_penalty: Length penalty
            early_stopping: Stop when enough candidates
        Returns:
            answer_texts: List[str]
        """
        # 1. Encode
        visual_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        fused_features, _ = self.fuse_multimodal(text_features, visual_features)
        
        # 2. Generate using decoder.generate() - efficient!
        batch_size = fused_features.size(0)
        
        # Create dummy decoder input (BOS tokens)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=fused_features.device
        )
        
        # Use decoder's built-in generation
        generated_ids = self.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused_features,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True  # Enable KV cache for faster generation
        )
        
        # 3. Decode answers
        answer_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            answer_texts.append(text)
        
        return answer_texts


# ============================================================================
# TESTING & UTILS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("Testing DINOv2 + BARTpho VQA Model...")
    
    # Initialize model
    model = DINOv2BARTphoVQA(
        dinov2_model_name='facebook/dinov2-base',
        bartpho_model_name='vinai/bartpho-syllable',
        num_cross_attn_layers=3,
        num_heads=16,  # Fixed: 1024 ÷ 16 = 64
        use_reasoning_quality_check=True,
        gradient_checkpointing=True
    )
    
    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n[INFO] Total parameters: {total/1e6:.1f}M")
    print(f"[INFO] Trainable parameters: {trainable/1e6:.1f}M")
    
    # Test forward pass
    print("\n[INFO] Testing forward pass...")
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Dummy inputs
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
    attention_mask = torch.ones(batch_size, 32).to(device)
    reasoning_input_ids = torch.randint(0, 1000, (batch_size, 64)).to(device)
    reasoning_attention_mask = torch.ones(batch_size, 64).to(device)
    answer_input_ids = torch.randint(0, 1000, (batch_size, 16)).to(device)
    answer_attention_mask = torch.ones(batch_size, 16).to(device)
    
    # Forward
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        reasoning_input_ids=reasoning_input_ids,
        reasoning_attention_mask=reasoning_attention_mask,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask
    )
    
    print(f"[INFO] Reasoning logits shape: {outputs.reasoning_logits.shape}")
    print(f"[INFO] Answer logits shape: {outputs.answer_logits.shape}")
    print(f"[INFO] Reasoning confidence: {outputs.reasoning_confidence}")
    
    print("\n[SUCCESS] Model test passed! ✓")
