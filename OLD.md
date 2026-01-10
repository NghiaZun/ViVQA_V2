## Kiến trúc Model Hiện tại

### **DINOv2 + BARTpho với Gated Cross-Attention**

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT                                        │
│  ┌──────────────┐              ┌──────────────┐                │
│  │    Image     │              │   Question   │                 │
│  │  224x224x3   │              │  "Màu gì?"   │                 │
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
│         │                             │                          │
│         ▼                             │                          │
│  ┌──────────────┐                     │                         │
│  │ Vision Proj  │                     │                         │
│  │  768→1024    │                     │                         │
│  └──────┬───────┘                     │                         │
│         │                             │                          │
│         │ [B, 257, 1024]              │                         │
│         └─────────────┬───────────────┘                         │
│                       │                                          │
│                       ▼                                          │
│           ┌───────────────────────┐                             │
│           │  Gated Cross-Attn     │  ◄── Multi-layer (3 tầng) │
│           │  (LXMERT/UNITER/BLIP) │                             │
│           └───────────┬───────────┘                             │
│                       │                                          │
│                       │ Fused Features [B, L, 1024]             │
│                       │                                          │
│           ┌───────────┴───────────┐                             │
│           │                       │                              │
│           ▼                       ▼                              │
│   ┌───────────────┐       ┌───────────────┐                    │
│   │   Reasoning   │       │     Answer    │                     │
│   │    Decoder    │───────│    Decoder    │                     │
│   │   (BARTpho)   │       │   (BARTpho)   │                     │
│   └───────┬───────┘       └───────┬───────┘                     │
│           │                       │                              │
│           ▼                       ▼                              │
│   "Tôi thấy cái bình      "màu xanh lá"                        │
│    màu xanh lá"                                                  │
│   (Reasoning)              (Answer)                              │
└─────────────────────────────────────────────────────────────────┘

Total Parameters: ~500M
```

### **Thành phần Chi tiết**

#### 1. **Vision Encoder: DINOv2-base** (86M params)
- **Model:** `facebook/dinov2-base`
- **Đặc điểm:**
  - Self-supervised learning trên 142M images
  - Language-agnostic (không phụ thuộc tiếng Anh như CLIP)
  - Output: 257 patches (256 patches + 1 CLS token) × 768 dim
  - SOTA cho computer vision tasks

#### 2. **Language Encoder + Decoder: BARTpho-large** (396M params)
- **Model:** `vinai/bartpho-syllable`
- **Đặc điểm:**
  - Pretrained trên 20GB Vietnamese corpus
  - Encoder (197M) + Decoder (199M)
  - Dimension: 1024 (d_model)
  - Tối ưu cho tiếng Việt

#### 3. **Gated Cross-Attention Fusion** (Multi-layer)
**Công thức:**
```
gate = σ(W_g · [text_pooled; visual_pooled])
attn = CrossAttention(text, visual)
output = gate ⊙ attn + (1-gate) ⊙ text
```

**Số lớp:** 3 layers (theo LXMERT/UNITER best practice)
- Layer 1: Low-level features (màu sắc, edges)
- Layer 2: Mid-level features (objects, parts)
- Layer 3: High-level features (semantic relationships)

**Ưu điểm:**
- Gating mechanism học khi nào cần visual vs textual info
- Multi-head attention (16 heads) cho rich interactions
- Residual connections + Layer Normalization

---

### **Kiến trúc Cũ: ViT5 + PhoBERT + CLIP**

```
┌─────────────────────────────────────────────┐
│  Image  →  CLIP ViT  →  [512 dim]          │
│  Question → PhoBERT → [768 dim]             │
│                                              │
│  Fusion: concat([512, 768]) → [1280 dim]   │
│          → Linear(1280 → 768)               │
│                                              │
│  Decoder: ViT5 → Answer                     │
└─────────────────────────────────────────────┘
```

### **Vấn đề của Kiến trúc Cũ**

| Vấn đề | Giải thích | Impact |
|--------|-----------|---------|
| **1. Simple Concatenation** | Chỉ ghép vector [CLIP; PhoBERT] rồi project qua Linear layer | ❌ Không có tương tác sâu giữa image-text |
| **2. CLIP Language Bias** | CLIP pretrain trên text tiếng Anh → không tối ưu cho tiếng Việt | ❌ Hiểu câu hỏi tiếng Việt kém |
| **3. ViT5 Limited Capacity** | ViT5-base chỉ ~220M params, nhỏ hơn nhiều so với SOTA | ❌ Generation quality thấp |

### **Cải tiến của Kiến trúc Mới**

| Cải tiến | Kỹ thuật | Lợi ích |
|----------|----------|---------|
| **1. Language-agnostic Vision** | DINOv2 (self-supervised, không text) | ✅ Không bias tiếng Anh, hiểu ảnh tốt hơn |
| **2. Vietnamese-specialized LM** | BARTpho-large (396M, pretrain tiếng Việt) | ✅ Hiểu + generate tiếng Việt chuẩn |
| **3. Gated Cross-Attention** | Multi-layer attention với gating | ✅ Deep interaction, adaptive fusion |

### **So sánh Performance**

| Metric | Kiến trúc Cũ | Kiến trúc Mới | Cải thiện |
|--------|--------------|---------------|-----------|
| **Accuracy** | ~54.7% | ~65.3% | +10.6% |

---

## Bài báo và Kỹ thuật Liên quan

### **1. Vision Encoder: DINOv2**
**Paper:** "DINOv2: Learning Robust Visual Features without Supervision"
- **Link:** https://arxiv.org/abs/2304.07193
- **Tác giả:** Meta AI Research (2023)
- **Kỹ thuật:**
  - Self-supervised learning (không cần labels)
  - Self-distillation với multi-crop strategy
  - Pretrain trên 142M images (ImageNet-22k + curated data)
- **Ưu điểm cho VQA:**
  - Language-agnostic (không bị ảnh hưởng bởi text tiếng Anh)
  - SOTA cho dense prediction tasks
  - Transfer tốt sang nhiều domains

### **2. Language Model: BARTpho**
**Paper:** "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese"
- **Link:** https://arxiv.org/abs/2109.09701
- **Tác giả:** VinAI Research (2021)
- **Kỹ thuật:**
  - BART architecture (denoising autoencoding)
  - Pretrain trên 20GB Vietnamese corpus
  - Syllable-level tokenization (tốt cho tiếng Việt)
- **Ưu điểm cho VQA:**
  - Encoder-decoder native (tốt cho generation)
  - Understanding + generation tiếng Việt chuẩn
  - Large model (396M params)

### **3. Gated Cross-Attention Fusion**

#### **LXMERT (2019)**
**Paper:** "LXMERT: Learning Cross-Modality Encoder Representations from Transformers"
- **Link:** https://arxiv.org/abs/1908.07490
- **Tác giả:** UNC Chapel Hill
- **Kỹ thuật:**
  - Cross-modality attention giữa vision và language
  - 5 layers cross-attention trong LXMERT encoder
  - Co-attention mechanism

#### **UNITER (2020)**
**Paper:** "UNITER: UNiversal Image-TExt Representation Learning"
- **Link:** https://arxiv.org/abs/1909.11740
- **Tác giả:** Microsoft Research
- **Kỹ thuật:**
  - Joint image-text embedding
  - Masked language/region modeling
  - 3 layers cross-attention

#### **BLIP (2022)**
**Paper:** "BLIP: Bootstrapping Language-Image Pre-training"
- **Link:** https://arxiv.org/abs/2201.12086
- **Tác giả:** Salesforce Research
- **Kỹ thuật:**
  - Multimodal mixture of encoder-decoder
  - Cross-attention với gating trong decoder
  - Caption bootstrapping

### **4. Chain-of-Thought (CoT)**
**Paper:** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- **Link:** https://arxiv.org/abs/2201.11903
- **Tác giả:** Google Research (2022)
- **Ý tưởng:**
  - Model suy luận từng bước trước khi trả lời
  - "Let's think step by step"
  - Cải thiện accuracy trên reasoning tasks

**Áp dụng cho VQA:**
```
Question: "Cái bình trong ảnh màu gì?"
Reasoning: "Tôi thấy một cái bình ở giữa ảnh. Cái bình có màu xanh lá cây."
Answer: "màu xanh lá"
```

**Loss Function:**
```
L_total = α * L_reasoning + β * L_answer + γ * L_quality
```

### **5. Gradient Checkpointing**
**Paper:** "Training Deep Nets with Sublinear Memory Cost"
- **Link:** https://arxiv.org/abs/1604.06174
- **Kỹ thuật:** Trade computation for memory
- **Áp dụng:** Save ~40% memory khi train large models

---

## Hai Phương pháp Training

### **Method 1: Teacher-Forced Training** (`train_dinov2_bartpho.py`)

#### **Ý tưởng:**
- Standard supervised learning
- Model nhìn ground-truth reasoning khi generate
- Faster convergence, dễ train

#### **Training Flow:**
```
┌─────────────────────────────────────────────────────────┐
│ Input: (image, question, GT_reasoning, GT_answer)       │
│                                                          │
│ Step 1: Encode image + question → fused_features       │
│                                                          │
│ Step 2: Generate reasoning (teacher forcing)            │
│   Decoder input: [SOS] + GT_reasoning[:-1]             │
│   Decoder output: reasoning_logits                      │
│   Loss: CrossEntropy(reasoning_logits, GT_reasoning)    │
│                                                          │
│ Step 3: Check reasoning quality                         │
│   confidence = QualityChecker(reasoning_hidden)         │
│   Loss: MSE(confidence, actual_accuracy)                │
│                                                          │
│ Step 4: Generate answer (teacher forcing)               │
│   Encoder: fused_features + reasoning_hidden            │
│   Decoder input: [SOS] + GT_answer[:-1]                │
│   Decoder output: answer_logits                         │
│   Loss: CrossEntropy(answer_logits, GT_answer)          │
│                                                          │
│ Total Loss: α*L_reasoning + β*L_answer + γ*L_quality   │
└─────────────────────────────────────────────────────────┘
```

---

### **Method 2: Autoregressive Training** (`train_autoregressive_cot.py`)

#### **Ý tưởng:**
- Model generate reasoning KHÔNG nhìn ground-truth
- Dùng generated reasoning để generate answer
- Khớp với inference behavior → better generalization

#### **Training Flow:**
```
┌─────────────────────────────────────────────────────────┐
│ Input: (image, question, GT_reasoning, GT_answer)       │
│                                                          │
│ Step 1: Encode image + question → fused_features       │
│                                                          │
│ Step 2: Generate reasoning (NO teacher forcing)         │
│   generated_reasoning = model.generate(                 │
│       fused_features,                                   │
│       max_length=96,                                    │
│       num_beams=1,        # Greedy/sampling            │
│       do_sample=True      # Stochastic                 │
│   )                                                     │
│   Loss: CrossEntropy(reasoning_logits, GT_reasoning)    │
│                                                          │
│ Step 3: Use GENERATED reasoning for answer             │
│   generated_reasoning_hidden = encode(generated_reasoning)│
│                                                          │
│ Step 4: Generate answer conditioned on GENERATED reasoning│
│   answer_from_gen = model.generate(                    │
│       fused_features + generated_reasoning_hidden      │
│   )                                                     │
│   Loss_answer_gen: CrossEntropy(answer_from_gen, GT_answer)│
│                                                          │
│ Step 5: Also train answer từ GT reasoning (stability)  │
│   answer_from_gt = model.generate(                     │
│       fused_features + GT_reasoning_hidden             │
│   )                                                     │
│   Loss_answer_gt: CrossEntropy(answer_from_gt, GT_answer)│
│                                                          │
│ Total Loss: α*L_reasoning + β*L_answer_gen + γ*L_answer_gt│
└─────────────────────────────────────────────────────────┘

```

### **4. Evaluation**
```python
# Metrics cần track:
- Reasoning accuracy: % tokens đúng
- Answer accuracy: % câu trả lời chính xác
- BLEU score: Quality của generation
- Reasoning confidence: Calibration của quality checker
```

---

## References

1. **DINOv2:** Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision" (2023)
2. **BARTpho:** Nguyen et al. "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese" (2021)
3. **LXMERT:** Tan & Bansal. "LXMERT: Learning Cross-Modality Encoder Representations" (2019)
4. **UNITER:** Chen et al. "UNITER: Universal Image-Text Representation Learning" (2020)
5. **BLIP:** Li et al. "BLIP: Bootstrapping Language-Image Pre-training" (2022)
6. **Chain-of-Thought:** Wei et al. "Chain-of-Thought Prompting Elicits Reasoning" (2022)
7. **Gradient Checkpointing:** Chen et al. "Training Deep Nets with Sublinear Memory Cost" (2016)
