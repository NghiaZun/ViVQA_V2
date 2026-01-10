# KIẾN TRÚC VÀ Ý TƯỞNG TRAINING - EVOLUTION & LESSONS LEARNED

## 📊 **Tổng quan Evolution**

| Version | Approach | Accuracy | Vấn đề chính |
|---------|----------|----------|--------------|
| **V1** | ViT5 + PhoBERT + CLIP | ~54.7% | Simple concat, không deep interaction |
| **V2** | DINOv2 + BARTpho + Gated Cross-Attention | ~65.3% | Tốt nhưng black-box |
| **V3** | Latent Reasoning VQA |  | Interpretable reasoning |

---

## 🏗️ **KIẾN TRÚC CHI TIẾT**

### **V1: Kiến trúc Cũ (Simple Fusion)**

```
┌─────────────────────────────────────────────┐
│  Image (224×224)                             │
│     ↓                                        │
│  CLIP ViT-B/32 (frozen)                     │
│     ↓                                        │
│  [CLS] token → [512 dim]                    │
│                                              │
│  Question ("Màu gì?")                       │
│     ↓                                        │
│  PhoBERT (frozen)                           │
│     ↓                                        │
│  [CLS] token → [768 dim]                    │
│                                              │
│  Fusion:                                     │
│  concat([512, 768]) → [1280 dim]            │
│     ↓                                        │
│  Linear(1280 → 768) + ReLU + Dropout        │
│     ↓                                        │
│  ViT5 Decoder (220M params)                 │
│     ↓                                        │
│  Answer: "màu xanh"                         │
└─────────────────────────────────────────────┘

Total: ~400M params
Trainable: ~220M params (chỉ ViT5)
```

**❌ Vấn đề V1:**

1. **Late Fusion (Simple Concat)**
   - Chỉ concat 2 vectors CLS rồi project
   - Không có cross-modal attention
   - Image-text interaction quá yếu
   
2. **CLIP Language Bias**
   - CLIP pretrain trên text tiếng Anh
   - Encode question tiếng Việt không tốt
   - Mất semantic information

3. **Limited Capacity**
   - ViT5-base chỉ 220M params
   - Không đủ capacity cho reasoning

4. **Frozen Encoders**
   - CLIP & PhoBERT đều frozen
   - Không adapt được với VQA task
   - Learning chỉ qua fusion layer nhỏ

---

### **V2: DINOv2 + BARTpho + Gated Cross-Attention**

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT                                        │
│  ┌──────────────┐              ┌──────────────┐                │
│  │    Image     │              │   Question   │                 │
│  │  224×224×3   │              │  "Màu gì?"   │                 │
│  └──────┬───────┘              └──────┬───────┘                 │
│         │                             │                          │
│         ▼                             ▼                          │
│  ┌──────────────┐              ┌──────────────┐                │
│  │   DINOv2     │              │   BARTpho    │                 │
│  │   Encoder    │              │   Encoder    │                 │
│  │   (86M)      │              │   (197M)     │                 │
│  └──────┬───────┘              └──────┬───────┘                 │
│         │                             │                          │
│         │ [B, 257, 768]               │ [B, L, 1024]            │
│         ▼                             │                          │
│  ┌──────────────┐                     │                         │
│  │ Vision Proj  │                     │                         │
│  │  768→1024    │                     │                         │
│  └──────┬───────┘                     │                         │
│         │ [B, 257, 1024]              │                         │
│         └─────────────┬───────────────┘                         │
│                       │                                          │
│                       ▼                                          │
│           ┌───────────────────────┐                             │
│           │  Gated Cross-Attn     │ ◄─ 3 layers                │
│           │  (LXMERT-style)       │    Multi-head (16)         │
│           │                       │    With gating             │
│           └───────────┬───────────┘                             │
│                       │                                          │
│                       │ Fused [B, L, 1024]                      │
│                       ▼                                          │
│           ┌───────────────────────┐                             │
│           │   BARTpho Decoder     │                             │
│           │   (199M params)       │                             │
│           └───────────┬───────────┘                             │
│                       │                                          │
│                       ▼                                          │
│                  "màu xanh"                                      │
└─────────────────────────────────────────────────────────────────┘

Total: ~500M params
Trainable: ~250M params (decoder + fusion + projection)
```

**Gated Cross-Attention (Chi tiết):**

```python
# Layer i in 3-layer stack
class GatedCrossAttentionLayer:
    def forward(text, vision):
        # 1. Cross-attention (text queries vision)
        attn_output = MultiHeadAttention(
            query=text,      # [B, L_text, 1024]
            key=vision,      # [B, L_vis, 1024]
            value=vision,
            num_heads=16
        )
        
        # 2. Gating mechanism
        text_pool = text.mean(dim=1)      # [B, 1024]
        vision_pool = vision.mean(dim=1)  # [B, 1024]
        gate = sigmoid(Linear([text_pool; vision_pool]))  # [B, 1024]
        
        # 3. Adaptive fusion
        output = gate * attn_output + (1 - gate) * text
        
        return LayerNorm(output)
```

**✅ Cải tiến V2:**

1. **Deep Multimodal Interaction**
   - 3 layers cross-attention thay vì concat
   - Mỗi layer học different semantic levels
   - Gating học khi nào cần vision vs text

2. **Language-Agnostic Vision**
   - DINOv2 không phụ thuộc text
   - Self-supervised trên 142M images
   - Transfer tốt cho tiếng Việt

3. **Vietnamese-Optimized LM**
   - BARTpho pretrain trên 20GB Vietnamese corpus
   - Native encoder-decoder (tốt cho generation)
   - Large capacity (396M)

4. **Trainable Fusion**
   - Cross-attention layers được train
   - Adapt cho VQA task
   - Rich image-text interactions

### **V3: Latent Reasoning VQA (CURRENT - FIXED)**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│  ┌──────────────┐              ┌──────────────┐                           │
│  │    Image     │              │   Question   │                            │
│  └──────┬───────┘              └──────┬───────┘                            │
│         │                             │                                     │
│         ▼                             ▼                                     │
│  ┌──────────────┐              ┌──────────────┐                           │
│  │   DINOv2     │              │   BARTpho    │                            │
│  │   Encoder    │              │   Encoder    │                            │
│  └──────┬───────┘              └──────┬───────┘                            │
│         │ [B, 257, 768]               │ [B, L, 1024]                       │
│         ▼                             ▼                                     │
│  ┌───────────────────────────────────────────┐                            │
│  │    Vision-First Fusion (FIX #3)           │                            │
│  │    - Vision queries text first            │                            │
│  │    - Image dropout (10%)                  │                            │
│  │    - Prevent text shortcut                │                            │
│  └───────────────┬───────────────────────────┘                            │
│                  │ Fused [B, L, 1024]                                      │
│                  ▼                                                          │
│  ┌───────────────────────────────────────────┐                            │
│  │   Compressed Latent Reasoning (FIX #4)    │                            │
│  │                                            │                            │
│  │   Learnable Queries: [6 tokens, 1024]    │                            │
│  │        ↓                                   │                            │
│  │   Cross-Attention (2 layers)              │                            │
│  │        ↓                                   │                            │
│  │   Compress: 1024 → 256 dim (FIX #4)      │                            │
│  │        ↓                                   │                            │
│  │   VAE Sampling: mu, logvar → z            │                            │
│  │        ↓                                   │                            │
│  │   KL Loss (with free bits) (FIX #2)       │                            │
│  │        ↓                                   │                            │
│  │   Expand: 256 → 1024 dim                  │                            │
│  │        ↓                                   │                            │
│  │   Reasoning Latents [B, 6, 1024]          │                            │
│  │        ↓                                   │                            │
│  │   Diversity Regularizer (FIX #5):         │                            │
│  │   - Orthogonality loss                    │                            │
│  │   - Token dropout (30%)                   │                            │
│  └───────────────┬───────────────────────────┘                            │
│                  │                                                          │
│                  ▼                                                          │
│  ┌───────────────────────────────────────────┐                            │
│  │   BARTpho Decoder (FIX #1)                │                            │
│  │                                            │                            │
│  │   Input: ONLY reasoning latents           │                            │
│  │   (NOT fused features!)                   │                            │
│  │                                            │                            │
│  │   = True bottleneck enforcement           │                            │
│  └───────────────┬───────────────────────────┘                            │
│                  │                                                          │
│                  ▼                                                          │
│             "màu xanh"                                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │  TRAINING CURRICULUM (FIX #8):                              │          │
│  │                                                              │          │
│  │  Stage 1 (5 epochs): Baseline (KL=0, decoder frozen)       │          │
│  │  Stage 2 (10 epochs): KL warmup (0→15.0, decoder unfreeze) │          │
│  │  Stage 3 (20 epochs): Full + Teacher distillation          │          │
│  └─────────────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────┘

Total: ~500M params
Trainable Stage 1: ~126M (decoder frozen)
Trainable Stage 2-3: ~165M (decoder last 2 layers unfrozen)

Latent Bottleneck: 6 tokens × 256 dim = 1,536 dims (vs 257×1024 = 263,168!)
Compression ratio: 171x smaller!
```

**🎯 Điểm khác biệt V3: Thêm Reasoning Module**

**Ý tưởng đơn giản:**

V2 (cũ): Ảnh + Câu hỏi → Trộn → Trả lời trực tiếp (không biết model suy nghĩ gì)

V3 (mới): Ảnh + Câu hỏi → Trộn → **Suy nghĩ (6 tokens)** → Trả lời từ suy nghĩ

```python
# V2: Trả lời trực tiếp
features = fusion(ảnh, câu_hỏi)          # 257 tokens, nhiều info
answer = decoder(features)                # → "màu xanh"
# ❌ Không biết model nghĩ gì

# V3: Phải suy nghĩ trước khi trả lời  
features = fusion(ảnh, câu_hỏi)          # 257 tokens
suy_nghĩ = nén_thành_6_tokens(features)  # 6 tokens reasoning
answer = decoder(suy_nghĩ)                # → "màu xanh"
# ✅ Có thể xem 6 tokens suy nghĩ
```

**Reasoning Module hoạt động:**
1. **Nén 257 tokens → 6 tokens:** Bắt buộc model phải tóm tắt info quan trọng
2. **6 tokens × 256 chiều:** Đủ nhỏ để force abstraction, đủ lớn để chứa reasoning
3. **Decoder chỉ nhìn 6 tokens:** Không thể "ăn gian" dùng raw features
4. **Train từ từ (3 giai đoạn):** 
   - Stage 1: Học trả lời cơ bản (không có reasoning)
   - Stage 2: Từ từ thêm reasoning (tránh "sụp đổ")
   - Stage 3: Thêm **VLM Teacher** (model lớn chỉ bảo)

**VLM Teacher là gì:**
- Dùng **Qwen2-VL-7B-Instruct** làm "giám khảo"
- VLM nhìn ảnh + câu hỏi → đánh giá reasoning latents nào tốt
- Model V3 học cách tạo reasoning latents được VLM đánh giá cao
- Giống như học sinh làm bài → thầy chấm điểm → học sinh biết cách làm tốt hơn

**Tại sao dùng Qwen2-VL:**
- ✅ **Hiểu tiếng Việt tốt:** Pretrain trên multilingual data
- ✅ **Vision + Language:** Native multimodal, không cần adapter
- ✅ **Efficient:** 7B params, chạy được trên GPU 16GB
- ⚠️ **LLaVA:** Tốt nhưng bias tiếng Anh, không native Vietnamese

---

## 📚 **References**

1. **DINOv2:** Oquab et al. (2023) - Self-supervised vision
2. **BARTpho:** Nguyen et al. (2021) - Vietnamese seq2seq
3. **LXMERT:** Tan & Bansal (2019) - Cross-modal attention
4. **β-VAE:** Higgins et al. (2017) - Disentangled representations
5. **Curriculum Learning:** Bengio et al. (2009) - Easy to hard
