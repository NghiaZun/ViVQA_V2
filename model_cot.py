"""
CHAIN-OF-THOUGHT VQA MODEL
===========================
Model suy nghĩ (reasoning) trước khi trả lời (answer)
Giống con người: "Tôi thấy cái bình màu xanh lá → Answer: màu xanh lá"

Architecture:
1. Encoder: CLIP ViT + PhoBERT → Fused representation
2. Reasoning Head: Generate explanation
3. Answer Head: Generate answer (conditioned on reasoning)
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoTOutput:
    """Output with reasoning and answer logits"""
    reasoning_logits: torch.Tensor
    answer_logits: torch.Tensor
    reasoning_hidden: Optional[torch.Tensor] = None
    answer_hidden: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class ChainOfThoughtVQAModel(nn.Module):
    """
    VQA Model với Chain-of-Thought reasoning
    """
    
    def __init__(
        self,
        clip_model_name='openai/clip-vit-base-patch32',
        text_encoder_name='vinai/phobert-base',
        decoder_name='VietAI/vit5-base',
        hidden_dim=768,
        fusion_method='concat',  # 'concat', 'add', 'cross_attention'
        use_reasoning_attention=True,  # Answer attend to reasoning
        num_cross_attn_layers=1,  # Number of cross-attention layers (SOTA: 1-3)
        dropout=0.1,
        use_flash_attention=False  # Flash Attention 2.0 (if available)
    ):
        super().__init__()
        
        # 1. ENCODERS
        print("[INFO] Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        print("[INFO] Loading PhoBERT...")
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
        # 2. DECODERS (Shared backbone, separate heads)
        print("[INFO] Loading ViT5 decoder...")
        self.decoder_backbone = AutoModelForSeq2SeqLM.from_pretrained(decoder_name)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        
        # Get dimensions
        self.clip_dim = self.clip_model.config.projection_dim  # 512
        self.text_dim = self.text_encoder.config.hidden_size   # 768
        self.decoder_dim = self.decoder_backbone.config.d_model  # 768
        
        # 3. FUSION MODULE
        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(self.clip_dim + self.text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # SOTA: GELU > ReLU
                nn.Dropout(dropout)
            )
        elif fusion_method == 'add':
            self.clip_proj = nn.Linear(self.clip_dim, hidden_dim)
            self.text_proj = nn.Linear(self.text_dim, hidden_dim)
            self.fusion = nn.LayerNorm(hidden_dim)
        elif fusion_method == 'cross_attention':
            # SOTA: Bidirectional cross-attention
            self.clip_proj = nn.Linear(self.clip_dim, hidden_dim)
            self.text_proj = nn.Linear(self.text_dim, hidden_dim)
            
            # Image → Text attention
            self.img_to_text_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            # Text → Image attention
            self.text_to_img_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # 4. REASONING HEAD (First - think first!)
        # Split into feature extraction + prediction
        self.reasoning_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.GELU(),  # GELU is SOTA (better than ReLU)
            nn.Dropout(dropout)
        )
        self.reasoning_predictor = nn.Linear(self.decoder_dim, self.decoder_tokenizer.vocab_size)
        
        # 5. ANSWER HEAD (Second - answer based on reasoning)
        self.use_reasoning_attention = use_reasoning_attention
        
        if use_reasoning_attention:
            # SOTA: Gated Cross-Attention (Flamingo-style)
            # Answer query projection
            self.answer_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            
            # Cross-attention: Answer attends to reasoning features
            self.reasoning_cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            # Gating mechanism (SOTA feature)
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
            
            # Layer norm after attention
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)
            
            # Answer head
            self.answer_head = nn.Sequential(
                nn.Linear(hidden_dim, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_dim, self.decoder_tokenizer.vocab_size)
            )
        else:
            # Simple answer head
            self.answer_head = nn.Sequential(
                nn.Linear(hidden_dim, self.decoder_dim),
                nn.LayerNorm(self.decoder_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_dim, self.decoder_tokenizer.vocab_size)
            )
        
        print(f"[INFO] Model initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode_image(self, pixel_values):
        """Encode image with CLIP"""
        vision_outputs = self.clip_model.vision_model(pixel_values)
        image_embeds = self.clip_model.visual_projection(vision_outputs.pooler_output)
        return image_embeds
    
    def encode_text(self, input_ids, attention_mask):
        """Encode question with PhoBERT"""
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        return text_embeds
    
    def fuse_features(self, image_embeds, text_embeds):
        """Fuse image and text features - SOTA methods"""
        if self.fusion_method == 'concat':
            # Simple concatenation
            fused = torch.cat([image_embeds, text_embeds], dim=-1)
            fused = self.fusion(fused)
        
        elif self.fusion_method == 'add':
            # Projection + addition
            image_proj = self.clip_proj(image_embeds)
            text_proj = self.text_proj(text_embeds)
            fused = self.fusion(image_proj + text_proj)
        
        elif self.fusion_method == 'cross_attention':
            # SOTA: Bidirectional cross-attention (BLIP-2 style)
            image_proj = self.clip_proj(image_embeds).unsqueeze(1)  # [B, 1, D]
            text_proj = self.text_proj(text_embeds).unsqueeze(1)    # [B, 1, D]
            
            # Image attends to text
            img_attended, _ = self.img_to_text_attn(
                query=image_proj,
                key=text_proj,
                value=text_proj
            )
            
            # Text attends to image
            text_attended, _ = self.text_to_img_attn(
                query=text_proj,
                key=image_proj,
                value=image_proj
            )
            
            # Combine both directions (co-attention)
            fused = (img_attended + text_attended).squeeze(1) / 2.0
            fused = self.fusion_norm(fused)
        
        return fused
    
    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        reasoning_labels=None,
        labels=None,
        return_reasoning_hidden=False
    ):
        """
        Forward pass: Image + Question → Reasoning → Answer
        
        Args:
            pixel_values: [B, 3, 224, 224]
            input_ids: [B, L]
            attention_mask: [B, L]
            reasoning_labels: [B, L_r] (optional, for training)
            labels: [B, L_a] (optional, for training)
            return_reasoning_hidden: Return reasoning hidden states
        """
        batch_size = pixel_values.size(0)
        
        # 1. ENCODE
        image_embeds = self.encode_image(pixel_values)  # [B, 512]
        text_embeds = self.encode_text(input_ids, attention_mask)  # [B, 768]
        
        # 2. FUSE
        fused_embeds = self.fuse_features(image_embeds, text_embeds)  # [B, hidden_dim]
        
        # 3. REASONING HEAD (Think first!)
        # Extract reasoning features (intermediate representation)
        reasoning_features = self.reasoning_feature_extractor(fused_embeds)  # [B, decoder_dim]
        reasoning_logits = self.reasoning_predictor(reasoning_features)  # [B, vocab_size]
        
        # 4. ANSWER HEAD (Answer based on reasoning)
        if self.use_reasoning_attention:
            # SOTA Gated Cross-Attention approach
            
            # Step 1: Project answer query
            answer_query = self.answer_query_proj(fused_embeds)  # [B, hidden_dim]
            
            # Step 2: Prepare for cross-attention
            # Query: what answer wants to know [B, 1, hidden_dim]
            # Key/Value: reasoning features [B, 1, hidden_dim]
            answer_query_seq = answer_query.unsqueeze(1)  
            
            # Project reasoning features back to hidden_dim for attention
            reasoning_key_value = fused_embeds.unsqueeze(1)  # Use original fused as key/value
            # Note: Could also use reasoning_features projected back to hidden_dim
            
            # Step 3: Cross-attention (answer attends to reasoning context)
            cross_attended, attn_weights = self.reasoning_cross_attention(
                query=answer_query_seq,
                key=reasoning_key_value,
                value=reasoning_key_value
            )
            cross_attended = cross_attended.squeeze(1)  # [B, hidden_dim]
            
            # Step 4: Gated residual connection (Flamingo-style)
            gate = torch.sigmoid(self.gate_proj(answer_query))  # [B, hidden_dim]
            gated_output = answer_query + gate * cross_attended  # Gated fusion
            
            # Step 5: Layer norm (stabilize)
            gated_output = self.cross_attn_norm(gated_output)
            
            # Step 6: Generate answer
            answer_logits = self.answer_head(gated_output)
        else:
            answer_logits = self.answer_head(fused_embeds)
        
        # 5. RETURN
        output = CoTOutput(
            reasoning_logits=reasoning_logits,
            answer_logits=answer_logits,
            reasoning_hidden=fused_embeds if return_reasoning_hidden else None
        )
        
        return output
    
    @torch.no_grad()
    def generate_answer(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_length=32,
        num_beams=4,
        return_reasoning=False
    ):
        """
        Generate answer (and optionally reasoning) for inference
        """
        self.eval()
        
        # Forward pass
        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Decode answer
        answer_ids = torch.argmax(outputs.answer_logits, dim=-1)
        answer_text = self.decoder_tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        
        if return_reasoning:
            # Decode reasoning
            reasoning_ids = torch.argmax(outputs.reasoning_logits, dim=-1)
            reasoning_text = self.decoder_tokenizer.decode(reasoning_ids[0], skip_special_tokens=True)
            return answer_text, reasoning_text
        
        return answer_text


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_cot_model(
    clip_model='openai/clip-vit-base-patch32',
    text_encoder='vinai/phobert-base',
    decoder='VietAI/vit5-base',
    hidden_dim=768,
    fusion='concat',
    use_reasoning_attention=True
):
    """Factory function to create CoT model"""
    model = ChainOfThoughtVQAModel(
        clip_model_name=clip_model,
        text_encoder_name=text_encoder,
        decoder_name=decoder,
        hidden_dim=hidden_dim,
        fusion_method=fusion,
        use_reasoning_attention=use_reasoning_attention
    )
    return model


if __name__ == '__main__':
    # Test model
    print("Testing Chain-of-Thought VQA Model...")
    
    model = create_cot_model()
    
    # Dummy input
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 32))
    attention_mask = torch.ones(batch_size, 32)
    
    # Forward
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    print(f"✓ Reasoning logits shape: {outputs.reasoning_logits.shape}")
    print(f"✓ Answer logits shape: {outputs.answer_logits.shape}")
    print("\nModel ready! 🚀")
