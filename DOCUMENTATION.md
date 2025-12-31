# Vietnamese Visual Question Answering (ViVQA V2)

## 📋 Mục lục
1. [Kiến trúc Model Hiện tại](#kiến-trúc-model-hiện-tại)
2. [Lý do Thay đổi Kiến trúc](#lý-do-thay-đổi-kiến-trúc)
3. [Bài báo và Kỹ thuật Liên quan](#bài-báo-và-kỹ-thuật-liên-quan)
4. [Hai Phương pháp Training](#hai-phương-pháp-training)

---

## 🏗️ Kiến trúc Model Hiện tại

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

Total Parameters: ~482M
Memory Footprint: ~9GB (FP32)
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

#### 4. **Chain-of-Thought (CoT) với Quality Check**
**Flow:**
```
Fused Features → Reasoning Decoder → Quality Check
                                          ↓
                           [High confidence] → Answer Decoder
                           [Low confidence]  → Skip/Penalty
```

**Quality Checker:**
- Input: Reasoning hidden states
- Output: Confidence score [0, 1]
- Loss: MSE(confidence, actual_accuracy)

---

## 🔄 Lý do Thay đổi Kiến trúc

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
| **4. No Reasoning Path** | Trả lời trực tiếp không qua suy luận | ❌ Thiếu explainability |
| **5. Fixed Feature Fusion** | Concat cố định, không adaptive theo context | ❌ Không linh hoạt |

### **Cải tiến của Kiến trúc Mới**

| Cải tiến | Kỹ thuật | Lợi ích |
|----------|----------|---------|
| **1. Language-agnostic Vision** | DINOv2 (self-supervised, không text) | ✅ Không bias tiếng Anh, hiểu ảnh tốt hơn |
| **2. Vietnamese-specialized LM** | BARTpho-large (396M, pretrain tiếng Việt) | ✅ Hiểu + generate tiếng Việt chuẩn |
| **3. Gated Cross-Attention** | Multi-layer attention với gating | ✅ Deep interaction, adaptive fusion |
| **4. Chain-of-Thought** | Reasoning → Answer (2-stage) | ✅ Explainable, better accuracy |
| **5. Larger Model** | 482M params (2x bigger) | ✅ More capacity, better performance |

### **So sánh Performance (Dự kiến)**

| Metric | Kiến trúc Cũ | Kiến trúc Mới | Cải thiện |
|--------|--------------|---------------|-----------|
| **Accuracy** | ~50-55% | ~65-70% | +15% |
| **Vietnamese Quality** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Tốt hơn nhiều |
| **Reasoning Quality** | Không có | ⭐⭐⭐⭐ | Có explainability |
| **Parameters** | ~220M | ~482M | 2.2× larger |

---

## 📚 Bài báo và Kỹ thuật Liên quan

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

**Áp dụng trong Model:**
```python
class GatedCrossAttentionLayer(nn.Module):
    def forward(self, text_features, visual_features):
        # 1. Cross-attention: text queries visual
        attn_output = CrossAttention(
            query=text_features,
            key=visual_features,
            value=visual_features
        )
        
        # 2. Gating: học khi nào cần visual info
        gate = sigmoid(Linear([text_pooled; visual_pooled]))
        
        # 3. Adaptive fusion
        output = gate * attn_output + (1-gate) * text_features
        
        return output
```

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

## 🎯 Hai Phương pháp Training

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

#### **Hyperparameters:**
```python
{
    'batch_size': 16,
    'gradient_accumulation': 4,  # Effective batch = 64
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'epochs': 30,
    'alpha_reasoning': 0.6,      # Reasoning weight
    'alpha_answer': 0.4,         # Answer weight
    'alpha_quality': 0.1,        # Quality weight
    'label_smoothing': 0.1,
    'use_amp': True,             # Mixed precision
}
```

#### **Khi nào dùng:**
- ✅ Initial training từ scratch
- ✅ Khi có ít data
- ✅ Muốn convergence nhanh
- ✅ Debugging model architecture

#### **Commands:**
```bash
# Train from scratch
python train_dinov2_bartpho.py

# Resume từ checkpoint
python train_dinov2_bartpho.py --resume checkpoint_epoch_10.pt
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

#### **Scheduled Sampling:**
- **Idea:** Dần dần giảm teacher forcing, tăng generation
- **Formula:**
  ```python
  generation_ratio = linear_schedule(
      start=0.0,    # 100% teacher forcing
      end=1.0,      # 100% generation
      epoch=current_epoch,
      total_epochs=10
  )
  
  if random() < generation_ratio:
      reasoning = model.generate(...)  # Use generation
  else:
      reasoning = ground_truth         # Use teacher forcing
  ```

#### **Hyperparameters:**
```python
{
    'batch_size': 16,
    'gradient_accumulation': 4,
    'learning_rate': 2e-5,
    'alpha_reasoning': 0.5,           # Reasoning loss
    'beta_answer_gen': 0.3,           # Answer từ generated reasoning
    'gamma_answer_gt': 0.2,           # Answer từ GT reasoning (stability)
    'scheduled_sampling_start': 0.0,  # Start: 100% teacher forcing
    'scheduled_sampling_end': 0.0,    # End: 100% generation
    'scheduled_sampling_anneal_epochs': 10,
}
```

#### **Khi nào dùng:**
- ✅ Fine-tuning sau Method 1
- ✅ Khi đã có pretrained checkpoint
- ✅ Muốn better inference alignment
- ✅ Có nhiều compute (chậm hơn Method 1)

#### **Commands:**
```bash
# Train autoregressive from checkpoint
python train_autoregressive_cot.py --resume best_model_main.pt

# Train with scheduled sampling
python train_autoregressive_cot.py \
    --resume best_model_main.pt \
    --scheduled_sampling_start 0.0 \
    --scheduled_sampling_end 1.0 \
    --anneal_epochs 10
```

---

### **So sánh Hai Methods**

| Khía cạnh | Teacher Forcing | Autoregressive |
|-----------|-----------------|----------------|
| **Training Speed** | ⚡⚡⚡ Nhanh | ⚡⚡ Chậm hơn ~30% |
| **Convergence** | ⚡⚡⚡ Ổn định | ⚡⚡ Có thể unstable |
| **Inference Alignment** | ⭐⭐ Exposure bias | ⭐⭐⭐ Khớp với inference |
| **Generalization** | ⭐⭐ Overfit dễ hơn | ⭐⭐⭐ Generalize tốt hơn |
| **Use Case** | Initial training | Fine-tuning |
| **Memory Usage** | 💾💾 ~9GB | 💾💾💾 ~12GB (cache generated) |

### **Recommended Training Pipeline:**

```
┌──────────────────────────────────────────────────────────┐
│                                                           │
│  Stage 1: Teacher Forcing (20 epochs)                    │
│  ├─ Fast convergence                                     │
│  ├─ Learn basic image-text alignment                     │
│  └─ Save checkpoint: best_model_main.pt                  │
│                                                           │
│           ↓                                               │
│                                                           │
│  Stage 2: Autoregressive (10 epochs)                     │
│  ├─ Load: best_model_main.pt                            │
│  ├─ Fine-tune với generation                             │
│  ├─ Better inference alignment                           │
│  └─ Save checkpoint: best_model_autoregressive.pt        │
│                                                           │
│           ↓                                               │
│                                                           │
│  Final Model: best_model_autoregressive.pt               │
│  ├─ Best accuracy                                        │
│  └─ Best generalization                                  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Training Tips

### **1. Memory Optimization**
```python
# Enable gradient checkpointing
model = DINOv2BARTphoVQA(gradient_checkpointing=True)  # Save 40% memory

# Use mixed precision (FP16)
use_amp = True  # ~2x faster, same accuracy

# Gradient accumulation
gradient_accumulation_steps = 4  # Simulate larger batch
```

### **2. Learning Rate Schedule**
```python
# Cosine schedule với warmup (SOTA)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps * 0.1,  # 10% warmup
    num_training_steps=total_steps
)
```

### **3. Loss Weights**
```python
# Method 1: Teacher Forcing
alpha_reasoning = 0.6  # Reasoning quan trọng hơn
alpha_answer = 0.4
alpha_quality = 0.1

# Method 2: Autoregressive
alpha_reasoning = 0.5
beta_answer_gen = 0.3   # Answer từ generated
gamma_answer_gt = 0.2   # Answer từ GT (stability)
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

## 🚀 Getting Started

### **Setup Environment**
```bash
# Install dependencies
pip install torch transformers pillow tqdm

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### **Quick Start: Method 1**
```bash
# Train với teacher forcing
python train_dinov2_bartpho.py \
    --data_path data/train.jsonl \
    --image_dir data/images/ \
    --output_dir checkpoints/ \
    --batch_size 16 \
    --epochs 20 \
    --learning_rate 2e-5
```

### **Quick Start: Method 2**
```bash
# Fine-tune với autoregressive
python train_autoregressive_cot.py \
    --data_path data/train.jsonl \
    --image_dir data/images/ \
    --output_dir checkpoints/ \
    --resume checkpoints/best_model_main.pt \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 1e-5
```

---

## 📖 References

1. **DINOv2:** Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision" (2023)
2. **BARTpho:** Nguyen et al. "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese" (2021)
3. **LXMERT:** Tan & Bansal. "LXMERT: Learning Cross-Modality Encoder Representations" (2019)
4. **UNITER:** Chen et al. "UNITER: Universal Image-Text Representation Learning" (2020)
5. **BLIP:** Li et al. "BLIP: Bootstrapping Language-Image Pre-training" (2022)
6. **Chain-of-Thought:** Wei et al. "Chain-of-Thought Prompting Elicits Reasoning" (2022)
7. **Gradient Checkpointing:** Chen et al. "Training Deep Nets with Sublinear Memory Cost" (2016)

---

## 📧 Contact

Nếu có câu hỏi về implementation hoặc training, vui lòng tạo issue hoặc liên hệ qua email.

---

**Tài liệu này được tạo:** December 31, 2025
**Phiên bản Model:** DINOv2-BARTpho v2.0
