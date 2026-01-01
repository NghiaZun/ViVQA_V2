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
    AutoTokenizer,
    MBartForConditionalGeneration
)
from transformers.modeling_outputs import BaseModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class CoTOutput:
    """Chain-of-Thought output với reasoning + answer"""
    reasoning_logits: torch.Tensor  # [batch, seq_len, vocab_size]
    answer_logits: torch.Tensor     # [batch, seq_len, vocab_size]
    reasoning_hidden: Optional[torch.Tensor] = None  # [batch, seq_len, hidden]
    answer_hidden: Optional[torch.Tensor] = None     # [batch, seq_len, hidden]
    reasoning_confidence: Optional[torch.Tensor] = None  # [batch] scalar confidence
    loss: Optional[torch.Tensor] = None


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


class ReasoningQualityChecker(nn.Module):
    """
    Đánh giá quality của reasoning trước khi generate answer
    
    Ý tưởng: Reasoning tốt phải:
    1. Có information từ cả image và question
    2. Coherent (không contradictory)
    3. Relevant to question
    
    Output: confidence score [0, 1]
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output confidence [0, 1]
        )
        
    def forward(self, reasoning_hidden):
        """
        Args:
            reasoning_hidden: [batch, seq_len, hidden_dim]
        Returns:
            confidence: [batch] scalar confidence scores
        """
        # Use [CLS] token or mean pooling
        reasoning_repr = reasoning_hidden.mean(dim=1)  # [batch, hidden_dim]
        confidence = self.scorer(reasoning_repr).squeeze(-1)  # [batch]
        return confidence


class DINOv2BARTphoVQA(nn.Module):
    """
    SOTA VQA Model: DINOv2 + BARTpho + Gated Cross-Attention
    
    Architecture:
    1. Vision: DINOv2-base (86M) - SOTA self-supervised vision
    2. Language: BARTpho-large encoder (197M) - Vietnamese understanding
    3. Fusion: 3-layer Gated Cross-Attention - Multi-modal alignment
    4. Generation: BARTpho-large decoder (199M) - Vietnamese generation
    5. Chain-of-Thought: Reasoning → Quality Check → Answer
    
    Total: ~482M params
    """
    
    def __init__(
        self,
        dinov2_model_name='facebook/dinov2-base',  # 86M params
        bartpho_model_name='vinai/bartpho-syllable',  # 396M params (large variant)
        num_cross_attn_layers=3,  # SOTA: 3 layers
        num_heads=16,  # 1024 dim ÷ 16 heads = 64 (BARTpho standard)
        dropout=0.1,
        use_reasoning_quality_check=True,
        gradient_checkpointing=True,
        reasoning_bottleneck_tokens=None  # None = no bottleneck, 4-8 = compress reasoning
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
        
        # === 2. LANGUAGE MODEL: BARTpho (Encoder + Decoder) ===
        # Note: BARTpho trên HuggingFace là mBART architecture
        self.bartpho = MBartForConditionalGeneration.from_pretrained(bartpho_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bartpho_model_name)
        bart_hidden_dim = self.bartpho.config.d_model  # 1024 for large
        
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
        
        # === 5. REASONING QUALITY CHECKER ===
        self.use_quality_check = use_reasoning_quality_check
        if use_reasoning_quality_check:
            self.reasoning_quality_checker = ReasoningQualityChecker(
                hidden_dim=bart_hidden_dim,
                dropout=dropout
            )
        
        # === 6. REASONING BOTTLENECK (optional) ===
        # Compress reasoning hidden từ [B, T, 1024] → [B, k, 1024]
        # Giúp reasoning trừu tượng hơn, không chỉ copy fused_features
        self.reasoning_bottleneck_tokens = reasoning_bottleneck_tokens
        if reasoning_bottleneck_tokens is not None:
            print(f"[INFO] Using reasoning bottleneck: {reasoning_bottleneck_tokens} tokens")
            # Learned query tokens (similar to Perceiver/BLIP Q-Former)
            self.reasoning_queries = nn.Parameter(
                torch.randn(1, reasoning_bottleneck_tokens, bart_hidden_dim)
            )
            self.reasoning_bottleneck_attn = nn.MultiheadAttention(
                embed_dim=bart_hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.reasoning_bottleneck_norm = nn.LayerNorm(bart_hidden_dim)
        
        # === 7. USE BARTPHO's LM_HEAD ===
        # Use pretrained lm_head for both reasoning and answer
        # Different encoder contexts → different outputs despite same head
        # Reasoning: condition on fused_features
        # Answer: condition on fused_features + reasoning_hidden
        
        # === 7. GRADIENT CHECKPOINTING (Save memory) ===
        if gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable()
            self.bartpho.gradient_checkpointing_enable()
            print("[INFO] ✓ Gradient checkpointing enabled")
        
        print(f"[INFO] ✓ Model initialized: ~482M parameters")
        
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
        # BARTpho encoder
        encoder_outputs = self.bartpho.model.encoder(
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
    
    def generate_reasoning(
        self, 
        fused_features, 
        reasoning_input_ids=None,
        reasoning_attention_mask=None
    ):
        """
        Generate reasoning từ fused features
        
        Args:
            fused_features: [batch, seq_len, 1024] - encoder output
            reasoning_input_ids: [batch, target_len] - teacher forcing labels
            reasoning_attention_mask: [batch, target_len]
        Returns:
            reasoning_logits: [batch, target_len, vocab_size]
            reasoning_hidden: [batch, target_len, 1024]
            reasoning_confidence: [batch] - quality score
        """
        # Decoder forward pass for reasoning
        decoder_outputs = self.bartpho.model.decoder(
            input_ids=reasoning_input_ids,
            attention_mask=reasoning_attention_mask,
            encoder_hidden_states=fused_features,
            return_dict=True,
            use_cache=False
        )
        
        reasoning_hidden = decoder_outputs.last_hidden_state  # [batch, target_len, 1024]
        reasoning_logits = self.bartpho.lm_head(reasoning_hidden)  # [batch, target_len, vocab_size]
        
        # BOTTLENECK: Compress reasoning hidden if enabled
        if self.reasoning_bottleneck_tokens is not None:
            batch_size = reasoning_hidden.size(0)
            # Expand queries: [1, k, 1024] → [batch, k, 1024]
            queries = self.reasoning_queries.expand(batch_size, -1, -1)
            
            # Cross-attention: queries attend to reasoning_hidden
            compressed, _ = self.reasoning_bottleneck_attn(
                query=queries,
                key=reasoning_hidden,
                value=reasoning_hidden
            )
            reasoning_hidden = self.reasoning_bottleneck_norm(compressed)
            # Now reasoning_hidden: [batch, k, 1024] where k << T
        
        # Quality check
        reasoning_confidence = None
        if self.use_quality_check:
            reasoning_confidence = self.reasoning_quality_checker(reasoning_hidden)
        
        return reasoning_logits, reasoning_hidden, reasoning_confidence
    
    def generate_answer(
        self,
        fused_features,
        reasoning_hidden,
        answer_input_ids=None,
        answer_attention_mask=None,
        use_reasoning_only=True
    ):
        """
        Generate answer conditioned on reasoning
        
        Args:
            fused_features: [batch, seq_len, 1024] - encoder output (image+question)
            reasoning_hidden: [batch, reason_len, 1024] - reasoning context
            answer_input_ids: [batch, target_len] - teacher forcing labels
            answer_attention_mask: [batch, target_len]
            use_reasoning_only: If True, answer ONLY uses reasoning_hidden (no shortcut)
        Returns:
            answer_logits: [batch, target_len, vocab_size]
            answer_hidden: [batch, target_len, 1024]
        """
        # FIX: Answer must DEPEND on reasoning_hidden, not bypass it!
        if use_reasoning_only:
            # Answer ONLY uses reasoning hidden (forces dependency)
            encoder_hidden_states = reasoning_hidden
        else:
            # Old behavior: concatenate both (allows shortcut → reasoning ignored!)
            encoder_hidden_states = torch.cat([fused_features, reasoning_hidden], dim=1)
        
        # Decoder forward pass for answer
        decoder_outputs = self.bartpho.model.decoder(
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
            use_cache=False
        )
        
        answer_hidden = decoder_outputs.last_hidden_state  # [batch, target_len, 1024]
        answer_logits = self.bartpho.lm_head(answer_hidden)  # [batch, target_len, vocab_size]
        
        return answer_logits, answer_hidden
    
    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        reasoning_input_ids=None,
        reasoning_attention_mask=None,
        answer_input_ids=None,
        answer_attention_mask=None
    ):
        """
        Full forward pass: Image + Question → Reasoning → Answer
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            input_ids: [batch, seq_len] - question tokens
            attention_mask: [batch, seq_len]
            reasoning_input_ids: [batch, reason_len] - reasoning target (for training)
            reasoning_attention_mask: [batch, reason_len]
            answer_input_ids: [batch, ans_len] - answer target (for training)
            answer_attention_mask: [batch, ans_len]
        Returns:
            CoTOutput with reasoning_logits, answer_logits, hidden states, confidence
        """
        # 1. Encode vision
        visual_features = self.encode_image(pixel_values)
        
        # 2. Encode text (question)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # 3. Fuse multimodal features
        fused_features, _ = self.fuse_multimodal(text_features, visual_features)
        
        # 4. Generate reasoning
        reasoning_logits, reasoning_hidden, reasoning_confidence = self.generate_reasoning(
            fused_features=fused_features,
            reasoning_input_ids=reasoning_input_ids,
            reasoning_attention_mask=reasoning_attention_mask
        )
        
        # 5. Generate answer (conditioned on reasoning)
        answer_logits, answer_hidden = self.generate_answer(
            fused_features=fused_features,
            reasoning_hidden=reasoning_hidden,
            answer_input_ids=answer_input_ids,
            answer_attention_mask=answer_attention_mask
        )
        
        return CoTOutput(
            reasoning_logits=reasoning_logits,
            answer_logits=answer_logits,
            reasoning_hidden=reasoning_hidden,
            answer_hidden=answer_hidden,
            reasoning_confidence=reasoning_confidence
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_reasoning_len=128,
        max_answer_len=32,
        num_beams=4,
        repetition_penalty=1.2,
        length_penalty=1.0,
        no_repeat_ngram_size=0,     # 🔥 Thêm: ngăn lặp n-gram
        early_stopping=True,         # 🔥 Thêm: dừng sớm
        temperature=1.0,             # 🔥 Thêm: control randomness
        top_k=50,                    # 🔥 Thêm: top-k sampling
        top_p=1.0                    # 🔥 Thêm: nucleus sampling
    ):
        """
        Inference mode: Generate reasoning → answer using lm_head
        
        Strategy:
        1. Generate reasoning from fused_features (lm_head)
        2. Generate answer from fused_features + reasoning (lm_head)
        
        Same head, different encoder contexts → different outputs
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            max_reasoning_len: Max length for reasoning
            max_answer_len: Max length for answer
            num_beams: Beam search width
            repetition_penalty: Penalty for repetition (default 1.2)
            length_penalty: Length penalty (default 1.0)
            no_repeat_ngram_size: Ngăn lặp n-gram (0 = tắt, 2-3 recommended)
            early_stopping: Dừng sớm khi đủ beam candidates
            temperature: Temperature for sampling (1.0 = no change)
            top_k: Top-k sampling (50 = reasonable)
            top_p: Nucleus sampling (1.0 = tắt)
        Returns:
            reasoning_text: List[str]
            answer_text: List[str]
            reasoning_confidence: Tensor[batch]
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        
        # 1. Encode
        visual_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        fused_features, _ = self.fuse_multimodal(text_features, visual_features)
        
        # 2. Generate reasoning using bartpho.generate (lm_head)
        encoder_outputs_wrapped = BaseModelOutput(last_hidden_state=fused_features)
        reasoning_outputs = self.bartpho.generate(
            encoder_outputs=encoder_outputs_wrapped,
            max_length=max_reasoning_len,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,  # 🔥 Pass through
            early_stopping=early_stopping,              # 🔥 Pass through
            temperature=temperature,                     # 🔥 Pass through
            top_k=top_k,                                 # 🔥 Pass through
            top_p=top_p,                                 # 🔥 Pass through
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        
        reasoning_text = self.tokenizer.batch_decode(reasoning_outputs, skip_special_tokens=True)
        
        # 3. Get reasoning hidden states
        reasoning_hidden = self.bartpho.model.decoder(
            input_ids=reasoning_outputs,
            encoder_hidden_states=fused_features,
            return_dict=True
        ).last_hidden_state
        
        reasoning_confidence = None
        if self.use_quality_check:
            reasoning_confidence = self.reasoning_quality_checker(reasoning_hidden)
        
        # 4. Generate answer conditioned on reasoning
        # Concat fused_features + reasoning_hidden as new encoder
        enhanced_encoder_output = torch.cat([fused_features, reasoning_hidden], dim=1)
        enhanced_encoder_wrapped = BaseModelOutput(last_hidden_state=enhanced_encoder_output)
        
        answer_outputs = self.bartpho.generate(
            encoder_outputs=enhanced_encoder_wrapped,
            max_length=max_answer_len,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,  # 🔥 Pass through
            early_stopping=early_stopping,              # 🔥 Pass through
            temperature=temperature,                     # 🔥 Pass through
            top_k=top_k,                                 # 🔥 Pass through
            top_p=top_p,                                 # 🔥 Pass through
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        
        answer_text = self.tokenizer.batch_decode(answer_outputs, skip_special_tokens=True)
        
        return reasoning_text, answer_text, reasoning_confidence


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
