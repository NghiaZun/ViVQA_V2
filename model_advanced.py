import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    CLIPVisionModel,
    CLIPProcessor
)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion module for vision-language interaction
    """
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key_value, key_padding_mask=None):
        # Cross attention: query attends to key_value
        attn_output, _ = self.multihead_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        query = self.layer_norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.layer_norm2(query + ffn_output)
        
        return output


class BimodalAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention: Vision attends to Text and Text attends to Vision
    """
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.vision_to_text = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        self.text_to_vision = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, vision_features, text_features, text_mask=None):
        # Vision attends to text
        v2t = self.vision_to_text(vision_features, text_features, text_mask)
        
        # Text attends to vision
        t2v = self.text_to_vision(text_features, vision_features, None)
        
        # Pool and concatenate
        v2t_pooled = v2t.mean(dim=1)  # (B, hidden)
        t2v_pooled = t2v[:, 0, :]      # CLS token (B, hidden)
        
        # Final fusion
        fused = torch.cat([v2t_pooled, t2v_pooled], dim=-1)
        output = self.fusion_proj(fused)
        
        return output


class VQAAdvancedModel(nn.Module):
    """
    Advanced Student VQA Model with state-of-the-art components:
    - Vision: CLIP ViT encoder (better than BLIP for zero-shot)
    - Text: PhoBERT (Vietnamese-specific)
    - Fusion: Bidirectional Cross-Attention
    - Decoder: VietT5 with enhanced encoder representations
    """

    def __init__(
        self,
        vision_model_name="openai/clip-vit-base-patch32",
        phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
        vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer",
        hidden_dim=768,
        num_fusion_layers=2,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        # -------------------------------------
        # 1. CLIP Vision Encoder (Better visual understanding)
        # -------------------------------------
        print("[INFO] Loading CLIP Vision encoder…")
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)
        
        # Project CLIP features to hidden_dim if needed
        clip_hidden = self.vision_encoder.config.hidden_size
        if clip_hidden != hidden_dim:
            self.vision_proj = nn.Linear(clip_hidden, hidden_dim)
        else:
            self.vision_proj = nn.Identity()

        # -------------------------------------
        # 2. PhoBERT (Text Encoder)
        # -------------------------------------
        print("[INFO] Loading PhoBERT…")
        if not any(f.endswith(("bin", "pt", "safetensors")) for f in os.listdir(phobert_dir)):
            print("[WARN] PhoBERT weights not found locally → using HF hub")
            self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        else:
            self.text_encoder = AutoModel.from_pretrained(phobert_dir)

        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(phobert_dir, use_fast=False)
        except:
            print("[WARN] PhoBERT tokenizer fallback → HF hub")
            self.text_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

        # -------------------------------------
        # 3. Multi-layer Bidirectional Cross-Attention Fusion
        # -------------------------------------
        print(f"[INFO] Initializing {num_fusion_layers}-layer cross-attention fusion…")
        self.fusion_layers = nn.ModuleList([
            BimodalAttentionFusion(hidden_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])
        
        # Additional projection for decoder input
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # -------------------------------------
        # 4. VietT5 Decoder
        # -------------------------------------
        print("[INFO] Loading VietT5…")
        if not any(f.endswith(("bin", "pt", "safetensors")) for f in os.listdir(vit5_dir)):
            print("[WARN] VietT5 weights not found locally → using HF hub")
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
        else:
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained(vit5_dir)

        try:
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(vit5_dir, use_fast=False)
        except:
            print("[WARN] VietT5 tokenizer fallback → HF")
            self.decoder_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", use_fast=False)

    # ===================================================================
    # FORWARD (training)
    # ===================================================================
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        Training: return logits + loss with advanced fusion
        """
        # 1. Vision encoding
        v_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = self.vision_proj(v_out)  # (B, seq_len, hidden)

        # 2. Text encoding
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # t_out: (B, seq_len, hidden)

        # 3. Multi-layer cross-attention fusion
        fused = v_feat.mean(dim=1)  # Initialize with vision pooling
        for fusion_layer in self.fusion_layers:
            fused_features = fusion_layer(v_feat, t_out, ~attention_mask.bool() if attention_mask is not None else None)
            fused = fused + fused_features  # Residual connection
        
        # Project for decoder
        fused = self.decoder_input_proj(fused).unsqueeze(1)  # (B, 1, hidden)
        mask = torch.ones(fused.size()[:2], dtype=torch.long, device=fused.device)

        # 4. T5 encoder (encode the fused representation)
        enc_out = self.decoder.get_encoder()(
            inputs_embeds=fused,
            attention_mask=mask,
        )

        # 5. T5 decoder (training)
        return self.decoder(
            encoder_outputs=enc_out,
            labels=labels,
            return_dict=True
        )

    # ===================================================================
    # GENERATE (inference)
    # ===================================================================
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_new_tokens=96,
        num_beams=4,
        early_stopping=True,
        **kwargs
    ):
        """
        Inference-time sequence generation with advanced fusion
        """
        # 1. Encode vision
        v_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = self.vision_proj(v_out)

        # 2. Encode text
        t_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # 3. Multi-layer cross-attention fusion
        fused = v_feat.mean(dim=1)  # Initialize
        for fusion_layer in self.fusion_layers:
            fused_features = fusion_layer(v_feat, t_out, ~attention_mask.bool() if attention_mask is not None else None)
            fused = fused + fused_features

        # Project for decoder
        fused = self.decoder_input_proj(fused).unsqueeze(1)
        mask = torch.ones(fused.size()[:2], dtype=torch.long, device=fused.device)

        # 4. T5 encoder
        encoder_outputs = self.decoder.get_encoder()(
            inputs_embeds=fused,
            attention_mask=mask,
        )

        # 5. T5 generate
        output_ids = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            **kwargs
        )

        return output_ids


class VQALightweightModel(nn.Module):
    """
    Lightweight version with fewer parameters but still advanced architecture:
    - Vision: Smaller CLIP or EfficientNet
    - Text: DistilPhoBERT or smaller PhoBERT variant
    - Fusion: Single-layer cross-attention with efficient projection
    - Decoder: VietT5-small
    """

    def __init__(
        self,
        vision_model_name="openai/clip-vit-base-patch32",
        phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
        vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer",
        hidden_dim=512,  # Reduced from 768
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        print("[INFO] Loading Lightweight VQA Model...")

        # Vision encoder (CLIP with dimension reduction)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        clip_hidden = self.vision_encoder.config.hidden_size
        self.vision_proj = nn.Linear(clip_hidden, hidden_dim)

        # Text encoder (PhoBERT)
        if not any(f.endswith(("bin", "pt", "safetensors")) for f in os.listdir(phobert_dir)):
            self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        else:
            self.text_encoder = AutoModel.from_pretrained(phobert_dir)

        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(phobert_dir, use_fast=False)
        except:
            self.text_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

        # Project PhoBERT to hidden_dim
        self.text_proj = nn.Linear(768, hidden_dim)

        # Single-layer efficient fusion
        self.fusion = BimodalAttentionFusion(hidden_dim, num_heads, dropout)

        # Decoder projection
        self.decoder_input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # VietT5 Decoder
        if not any(f.endswith(("bin", "pt", "safetensors")) for f in os.listdir(vit5_dir)):
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
        else:
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained(vit5_dir)

        try:
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(vit5_dir, use_fast=False)
        except:
            self.decoder_tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base", use_fast=False)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # Vision
        v_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = self.vision_proj(v_out)

        # Text
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        t_feat = self.text_proj(t_out)

        # Fusion
        fused = self.fusion(v_feat, t_feat, ~attention_mask.bool() if attention_mask is not None else None)
        fused = self.decoder_input_proj(fused).unsqueeze(1)
        
        mask = torch.ones(fused.size()[:2], dtype=torch.long, device=fused.device)

        # Encode
        enc_out = self.decoder.get_encoder()(
            inputs_embeds=fused,
            attention_mask=mask,
        )

        # Decode
        return self.decoder(
            encoder_outputs=enc_out,
            labels=labels,
            return_dict=True
        )

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=96, 
                 num_beams=4, early_stopping=True, **kwargs):
        # Vision
        v_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = self.vision_proj(v_out)

        # Text
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        t_feat = self.text_proj(t_out)

        # Fusion
        fused = self.fusion(v_feat, t_feat, ~attention_mask.bool() if attention_mask is not None else None)
        fused = self.decoder_input_proj(fused).unsqueeze(1)
        
        mask = torch.ones(fused.size()[:2], dtype=torch.long, device=fused.device)

        # Encode
        encoder_outputs = self.decoder.get_encoder()(
            inputs_embeds=fused,
            attention_mask=mask,
        )

        # Generate
        output_ids = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            **kwargs
        )

        return output_ids
