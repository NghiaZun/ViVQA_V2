import os
import torch
from torch import nn
from transformers import (
    BlipForQuestionAnswering,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM
)


class VQAGenModel(nn.Module):
    """
    Student VQA Model:
    - Vision: BLIP ViT encoder
    - Text: PhoBERT
    - Fusion: concat vision + text
    - Decoder: VietT5
    """

    def __init__(
        self,
        vision_model_name="Salesforce/blip-vqa-base",
        phobert_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/phobert_tokenizer",
        vit5_dir="/kaggle/input/checkpoints/transformers/default/1/checkpoints/vit5_tokenizer",
        hidden_dim=768
    ):
        super().__init__()

        # -------------------------------------
        # 1. BLIP ViT (Vision Encoder)
        # -------------------------------------
        print("[INFO] Loading BLIP VQA encoder…")
        blip = BlipForQuestionAnswering.from_pretrained(vision_model_name)
        self.vision_encoder = blip.vision_model  # (B, seq, hidden_dim)

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
        # 3. Fusion Module
        # -------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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

        # Load decoder tokenizer
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
        Training: return logits + loss
        """
        # Vision
        v_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = v_out.mean(dim=1)  # (B, hidden)

        # Text
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        t_feat = t_out[:, 0, :]  # CLS embedding

        # Fusion
        fused = torch.cat([v_feat, t_feat], dim=-1)
        fused = self.fusion(fused).unsqueeze(1)
        mask = torch.ones(fused.size()[:2], dtype=torch.long, device=fused.device)

        # Encode fusion
        enc_out = self.decoder.get_encoder()(
            inputs_embeds=fused,
            attention_mask=mask,
        )

        # Decode (training)
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
        Inference-time sequence generation
        Fully compatible with HF generate()
        """

        # 1. Encode vision
        v_out = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        v_feat = v_out.mean(dim=1)

        # 2. Encode text
        t_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        t_feat = t_out[:, 0, :]

        # 3. Fusion
        fused = torch.cat([v_feat, t_feat], dim=-1)
        fused = self.fusion(fused).unsqueeze(1)

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
            eos_token_id=self.decoder_tokenizer.eos_token_id
        )

        return output_ids
