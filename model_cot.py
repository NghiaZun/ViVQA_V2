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
        
        # 4. ENCODER PROJECTION (project fused features to decoder dim)
        self.encoder_projection = nn.Sequential(
            nn.Linear(hidden_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout)
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
        Forward pass using ViT5 decoder properly
        
        Args:
            pixel_values: [B, 3, 224, 224]
            input_ids: [B, L] - question tokens
            attention_mask: [B, L]
            reasoning_labels: [B, L_r] - reasoning target (optional for training)
            labels: [B, L_a] - answer target (optional for training)
        """
        batch_size = pixel_values.size(0)
        
        # 1. ENCODE: Image + Text
        image_embeds = self.encode_image(pixel_values)  # [B, 512]
        text_embeds = self.encode_text(input_ids, attention_mask)  # [B, 768]
        
        # 2. FUSE: Combine vision and language
        fused_embeds = self.fuse_features(image_embeds, text_embeds)  # [B, hidden_dim]
        
        # 3. PROJECT: to decoder dimension
        encoder_hidden = self.encoder_projection(fused_embeds)  # [B, decoder_dim]
        encoder_hidden = encoder_hidden.unsqueeze(1)  # [B, 1, decoder_dim]
        
        # 4. DECODE with ViT5 decoder (teacher forcing if labels provided)
        # Reasoning generation (first)
        if reasoning_labels is not None:
            # Training: use teacher forcing
            reasoning_outputs = self.decoder_backbone(
                decoder_input_ids=reasoning_labels,
                encoder_outputs=(encoder_hidden,),
                return_dict=True
            )
            reasoning_logits = reasoning_outputs.logits  # [B, L_r, vocab_size]
        else:
            # Inference: no labels, just return None (use generate() instead)
            reasoning_logits = None
        
        # Answer generation (second, based on reasoning)
        if labels is not None:
            # Training: use teacher forcing
            answer_outputs = self.decoder_backbone(
                decoder_input_ids=labels,
                encoder_outputs=(encoder_hidden,),
                return_dict=True
            )
            answer_logits = answer_outputs.logits  # [B, L_a, vocab_size]
        else:
            # Inference: no labels, just return None (use generate() instead)
            answer_logits = None
        
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
        num_beams=1,
        return_reasoning=False
    ):
        """
        Generate answer with proper autoregressive decoding using ViT5 decoder
        """
        self.eval()
        device = pixel_values.device
        batch_size = pixel_values.size(0)
        
        # 1. Encode inputs
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        fused_embeds = self.fuse_features(image_embeds, text_embeds)
        
        # 2. Project to decoder dimension
        encoder_hidden = self.encoder_projection(fused_embeds)
        encoder_hidden = encoder_hidden.unsqueeze(1)  # [B, 1, decoder_dim]
        
        # 3. Generate answer using ViT5 decoder
        output_ids = self.decoder_backbone.generate(
            inputs_embeds=encoder_hidden,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
        )
        
        # 4. Decode
        answer_text = self.decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if return_reasoning:
            # For reasoning, generate separately
            reasoning_ids = self.decoder_backbone.generate(
                inputs_embeds=encoder_hidden,
                max_length=max_length * 2,  # Reasoning is longer
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
            )
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
