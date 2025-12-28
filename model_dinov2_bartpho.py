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
        
        # === 6. GENERATION HEADS ===
        # Custom heads for Chain-of-Thought generation
        # These will be used for BOTH training and inference
        self.reasoning_head = nn.Linear(bart_hidden_dim, self.bartpho.config.vocab_size)
        self.answer_head = nn.Linear(bart_hidden_dim, self.bartpho.config.vocab_size)
        
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
        reasoning_logits = self.reasoning_head(reasoning_hidden)  # [batch, target_len, vocab_size]
        
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
        answer_attention_mask=None
    ):
        """
        Generate answer conditioned on reasoning
        
        Args:
            fused_features: [batch, seq_len, 1024] - encoder output
            reasoning_hidden: [batch, reason_len, 1024] - reasoning context
            answer_input_ids: [batch, target_len] - teacher forcing labels
            answer_attention_mask: [batch, target_len]
        Returns:
            answer_logits: [batch, target_len, vocab_size]
            answer_hidden: [batch, target_len, 1024]
        """
        # Concatenate fused features và reasoning as encoder output
        # Ý tưởng: Answer should attend to both image+question AND reasoning
        enhanced_encoder_output = torch.cat([fused_features, reasoning_hidden], dim=1)
        
        # Decoder forward pass for answer
        decoder_outputs = self.bartpho.model.decoder(
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
            encoder_hidden_states=enhanced_encoder_output,
            return_dict=True,
            use_cache=False
        )
        
        answer_hidden = decoder_outputs.last_hidden_state  # [batch, target_len, 1024]
        answer_logits = self.answer_head(answer_hidden)  # [batch, target_len, vocab_size]
        
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
        num_beams=1,
        temperature=1.0,
        top_p=0.9
    ):
        """
        Inference mode: Generate reasoning và answer using CUSTOM HEADS
        
        CRITICAL: Use same heads as training (reasoning_head, answer_head)
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            max_reasoning_len: Max length for reasoning
            max_answer_len: Max length for answer
            num_beams: Beam search width (default 1 for greedy)
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
        
        # 2. Generate reasoning using CUSTOM reasoning_head (autoregressive)
        reasoning_ids = self._generate_with_custom_head(
            encoder_hidden_states=fused_features,
            head=self.reasoning_head,
            max_length=max_reasoning_len,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p
        )
        
        reasoning_text = self.tokenizer.batch_decode(reasoning_ids, skip_special_tokens=True)
        
        # 3. Get reasoning hidden states for quality check
        reasoning_hidden = self.bartpho.model.decoder(
            input_ids=reasoning_ids,
            encoder_hidden_states=fused_features,
            return_dict=True
        ).last_hidden_state
        
        reasoning_confidence = None
        if self.use_quality_check:
            reasoning_confidence = self.reasoning_quality_checker(reasoning_hidden)
        
        # 4. Generate answer using CUSTOM answer_head (conditioned on reasoning)
        enhanced_encoder_output = torch.cat([fused_features, reasoning_hidden], dim=1)
        
        answer_ids = self._generate_with_custom_head(
            encoder_hidden_states=enhanced_encoder_output,
            head=self.answer_head,
            max_length=max_answer_len,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p
        )
        
        answer_text = self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
        
        return reasoning_text, answer_text, reasoning_confidence
    
    def _generate_with_custom_head(
        self,
        encoder_hidden_states,
        head,
        max_length=128,
        num_beams=1,
        temperature=1.0,
        top_p=0.9
    ):
        """
        Autoregressive generation using custom head
        
        Args:
            encoder_hidden_states: [batch, seq_len, hidden_dim]
            head: nn.Linear custom head (reasoning_head or answer_head)
            max_length: Max generation length
            num_beams: Beam search (1 = greedy)
            temperature: Sampling temperature
            top_p: Nucleus sampling
        Returns:
            generated_ids: [batch, max_length]
        """
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # Start with BOS token
        input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Autoregressive decoding
        for step in range(max_length - 1):
            # Decoder forward
            decoder_outputs = self.bartpho.model.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
                use_cache=False
            )
            
            hidden_states = decoder_outputs.last_hidden_state  # [batch, cur_len, hidden]
            
            # Get logits from custom head
            logits = head(hidden_states[:, -1, :])  # [batch, vocab_size] - last token
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check if all sequences have EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        return input_ids


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
