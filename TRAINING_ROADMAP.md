# 🎯 TRAINING ROADMAP - Chain-of-Thought VQA

## 📋 **OVERVIEW**

**Mục tiêu**: Đạt 70%+ accuracy trên ViVQA với Chain-of-Thought reasoning

**Timeline**: 20 epochs (~8-12 giờ training trên 1 GPU V100/A100)

**Chiến lược**: Progressive learning - từ dễ đến khó, từ reasoning đến answer

---

## 🏗️ **KIẾN TRÚC MODEL**

### **Components**

```
┌─────────────────────────────────────────────────────┐
│                    INPUT                             │
│  Image (224x224) + Question (Vietnamese text)       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              ENCODERS (Frozen/Fine-tuned)            │
│  • CLIP ViT-B/32: Image → 512-dim                  │
│  • PhoBERT: Question → 768-dim                      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         FUSION MODULE (Cross-Attn #1 - Optional)     │
│  Simple: Concat [512 + 768] → 768-dim              │
│  Advanced: Bidirectional Cross-Attention            │
│    • Image ↔ Text attend to each other             │
│    • Better multimodal alignment                    │
│  With: LayerNorm + GELU + Dropout                   │
└─────────────────────────────────────────────────────┘
                        ↓
              [Fused representation]
            "Image + Question combined"
                        ↓
┌─────────────────────────────────────────────────────┐
│       REASONING HEAD (Priority 60% - Think First!)   │
│  Extract reasoning features from fused context       │
│  Generate explanation/reasoning                      │
│  Output: "Tôi thấy cái bình có màu xanh lá cây"    │
│  → reasoning_logits (Vietnamese text)               │
└─────────────────────────────────────────────────────┘
                        ↓
        [Pass reasoning context to answer]
                        ↓
┌─────────────────────────────────────────────────────┐
│  GATED CROSS-ATTENTION (Cross-Attn #2 - CRITICAL!)  │
│  Answer attends to reasoning context                │
│                                                      │
│  Query:     answer_query (what answer wants)        │
│  Key/Value: fused_embeds (with reasoning info)      │
│                                                      │
│  Gate: Learn how much reasoning to use (0-100%)     │
│  Output: answer_query + gate × reasoning_context    │
│                                                      │
│  SOTA Flamingo-style: Learnable gate + Residual     │
└─────────────────────────────────────────────────────┘
                        ↓
         [Answer with reasoning context]
                        ↓
┌─────────────────────────────────────────────────────┐
│      ANSWER HEAD (Priority 40% - Answer Based on    │
│                   Reasoning!)                        │
│  Input: Gated output (original + reasoning)         │
│  Generate final answer: "màu xanh lá"              │
│  → answer_logits (Vietnamese text)                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                 MULTI-TASK LOSS                      │
│  0.6 × Reasoning Loss + 0.4 × Answer Loss          │
│  Weighted by teacher confidence                     │
└─────────────────────────────────────────────────────┘
```

### **Model Size**
- **Total parameters**: ~250M
- **Trainable parameters**: ~50M (with frozen encoders)
- **Memory**: ~8GB GPU (batch=16, fp16)
- **Speed**: ~3-5 samples/sec training

---

## 📊 **DATASET INFO**

### **Statistics**
```python
Train: 11,890 samples
Val:   ~1,500 samples

Question Types:
├── SPATIAL:  30% (3,567 samples) - "ở đâu", "bên cạnh"
├── OBJECT:   25% (2,972 samples) - "cái gì", "đồ vật"
├── COUNT:    15% (1,783 samples) - "bao nhiêu"
├── COLOR:    15% (1,783 samples) - "màu gì"
├── SCENE:    10% (1,189 samples) - "đang làm gì"
└── COMPLEX:   5%   (594 samples) - Multi-hop reasoning
```

### **Teacher Confidence (reasoning_weight)**
```python
Easy (1.0-1.5):   40% - SPATIAL, OBJECT basic
Medium (1.5-2.5): 40% - COUNT, COLOR, SCENE
Hard (2.5-3.0):   20% - COMPLEX multi-hop
```

---

## 🎯 **TRAINING PHASES - 20 EPOCHS**

### **PHASE 1: WARMUP & FOUNDATION (Epochs 1-5)**

**Mục tiêu**: Model học basics, warmup learning rate

**What happens:**
```
Epoch 1-2: Learning rate ramps up (0 → 5e-5)
Epoch 3-5: Start learning basic patterns
```

**Expected metrics:**
| Epoch | LR | Reasoning Loss | Answer Loss | Val Accuracy | Notes |
|-------|-----|----------------|-------------|--------------|-------|
| 1 | 1e-5 | 3.5 | 2.8 | 35-40% | High loss, random guessing |
| 2 | 3e-5 | 2.8 | 2.2 | 45-50% | Starting to learn |
| 3 | 5e-5 | 2.4 | 1.9 | 52-55% | Basic patterns emerge |
| 4 | 5e-5 | 2.1 | 1.7 | 55-58% | Reasoning improves |
| 5 | 5e-5 | 1.9 | 1.5 | 58-60% | **Baseline established** ✓ |

**What to check:**
- [ ] Loss decreasing steadily
- [ ] No NaN/Inf in gradients
- [ ] GPU memory stable (~8GB)
- [ ] Training speed consistent (~3-5 samples/sec)

**Red flags:**
- ❌ Loss not decreasing after Epoch 2 → learning rate too high/low
- ❌ Loss spikes → gradient explosion (check grad clipping)
- ❌ OOM → reduce batch size

**Actions:**
```python
# Monitor gradient norms
print(f"Grad norm: {grad_norm:.3f}")  # Should be 0.5-2.0

# Check sample predictions
for i in range(3):
    print(f"Q: {question[i]}")
    print(f"Reasoning (pred): {pred_reasoning[i]}")
    print(f"Reasoning (true): {true_reasoning[i]}")
    print(f"Answer (pred): {pred_answer[i]}")
    print(f"Answer (true): {true_answer[i]}")
```

---

### **PHASE 2: RAPID LEARNING (Epochs 6-10)**

**Mục tiêu**: Model học nhanh, accuracy tăng mạnh

**What happens:**
```
Epoch 6-8:  Learning rate ở peak (5e-5)
Epoch 9-10: Start cosine decay (5e-5 → 4e-5)
```

**Expected metrics:**
| Epoch | LR | Reasoning Loss | Answer Loss | Val Accuracy | Notes |
|-------|-----|----------------|-------------|--------------|-------|
| 6 | 5e-5 | 1.7 | 1.3 | 60-62% | Reasoning quality improves |
| 7 | 5e-5 | 1.5 | 1.2 | 62-64% | Answer aligns with reasoning |
| 8 | 5e-5 | 1.3 | 1.0 | 64-66% | **Major milestone** ✓ |
| 9 | 4.5e-5 | 1.2 | 0.95 | 66-67% | Fine-tuning starts |
| 10 | 4e-5 | 1.1 | 0.90 | 67-68% | **70% in sight** ✓ |

**Per-type accuracy (Expected at Epoch 10):**
```
SPATIAL:  70-72%  (baseline: 65%)
OBJECT:   72-75%  (baseline: 68%)
COUNT:    60-62%  (baseline: 55%)
COLOR:    65-68%  (baseline: 62%)
SCENE:    62-65%  (baseline: 58%)
COMPLEX:  48-52%  (baseline: 45%)
```

**What to check:**
- [ ] Reasoning quality: Read 10-20 samples manually
- [ ] Answer-reasoning alignment: Do they match?
- [ ] Type-specific performance: Which types lag behind?

**Red flags:**
- ❌ Accuracy plateau at 60% → model not learning reasoning
- ❌ Reasoning good but answer bad → increase alpha_answer
- ❌ Both bad → check data quality

**Actions:**
```python
# Analyze reasoning quality
def analyze_reasoning(model, val_loader):
    good_reasoning = 0
    total = 0
    
    for batch in val_loader:
        pred_reasoning = model.generate_reasoning(batch)
        true_reasoning = batch['reasoning']
        
        # Check semantic similarity (rough)
        if semantic_match(pred_reasoning, true_reasoning):
            good_reasoning += 1
        total += 1
    
    print(f"Reasoning quality: {good_reasoning/total:.1%}")

# Expected: 60-70% reasoning quality at Epoch 10
```

---

### **PHASE 3: FINE-TUNING (Epochs 11-15)**

**Mục tiêu**: Optimize details, polish performance

**What happens:**
```
Epoch 11-15: Cosine decay (4e-5 → 1e-5)
EMA kicks in strongly
Model converging to optimal
```

**Expected metrics:**
| Epoch | LR | Reasoning Loss | Answer Loss | Val Accuracy | Notes |
|-------|-----|----------------|-------------|--------------|-------|
| 11 | 3.5e-5 | 1.0 | 0.85 | 68-69% | Refinement phase |
| 12 | 3e-5 | 0.95 | 0.82 | 69-70% | **TARGET REACHED!** 🎯 |
| 13 | 2.5e-5 | 0.90 | 0.80 | 70-71% | Pushing higher |
| 14 | 2e-5 | 0.88 | 0.78 | 71-72% | **Peak performance** ⭐ |
| 15 | 1.5e-5 | 0.87 | 0.77 | 71-72% | Stable performance |

**Per-type accuracy (Expected at Epoch 14):**
```
SPATIAL:  73-76%  ⬆️ Best improvement
OBJECT:   75-78%  ⬆️ Solid
COUNT:    62-65%  ⬆️ Still improving
COLOR:    68-71%  ⬆️ Good
SCENE:    65-68%  ⬆️ Decent
COMPLEX:  53-58%  ⬆️ Challenging but better
```

**What to check:**
- [ ] EMA model better than regular? (should be +0.5-1%)
- [ ] No overfitting? (val loss < train loss)
- [ ] Attention weights make sense? (visualize)

**Red flags:**
- ❌ Val loss increasing → overfitting (early stop)
- ❌ Train accuracy >> Val accuracy → too much regularization needed

**Actions:**
```python
# Compare EMA vs regular model
print(f"Regular model: {acc_regular:.2%}")
print(f"EMA model: {acc_ema:.2%}")
# Expected: EMA +0.5-1% better

# Check overfitting
train_loss = evaluate(model, train_loader)
val_loss = evaluate(model, val_loader)
print(f"Train loss: {train_loss:.3f}")
print(f"Val loss: {val_loss:.3f}")
# Expected: val_loss ≈ train_loss or slightly higher

# Visualize attention
visualize_cross_attention(model, sample)
# Should see: Answer attends to relevant reasoning parts
```

---

### **PHASE 4: CONVERGENCE (Epochs 16-20)**

**Mục tiêu**: Squeeze last 1-2%, find best checkpoint

**What happens:**
```
Epoch 16-20: Very low LR (1e-5 → 1e-6)
Minimal weight updates
May plateau or slight decline
```

**Expected metrics:**
| Epoch | LR | Reasoning Loss | Answer Loss | Val Accuracy | Notes |
|-------|-----|----------------|-------------|--------------|-------|
| 16 | 1e-5 | 0.86 | 0.76 | 71-72% | Minor tweaks |
| 17 | 8e-6 | 0.85 | 0.75 | 72-73% | **Best checkpoint?** ⭐⭐ |
| 18 | 6e-6 | 0.85 | 0.75 | 72-73% | Plateau |
| 19 | 4e-6 | 0.85 | 0.75 | 71-72% | May decline slightly |
| 20 | 2e-6 | 0.85 | 0.75 | 71-72% | **Training complete** ✓ |

**Final expected accuracy:**
```
Overall:  71-73% (Target: 70%+) ✓✓✓
Best:     72-73% (Epoch 17-18)
Worst:    70-71% (Epoch 19-20)

By type:
SPATIAL:  74-77%
OBJECT:   76-79%
COUNT:    63-66%
COLOR:    69-72%
SCENE:    66-69%
COMPLEX:  55-60%
```

**What to check:**
- [ ] Find best checkpoint (may not be last epoch!)
- [ ] Ensemble top-3 checkpoints? (+1-2% boost)
- [ ] Test-time augmentation? (+0.5-1% boost)

**Red flags:**
- ❌ Accuracy drops >1% → overfitting, use earlier checkpoint
- ❌ No improvement after Epoch 15 → could have stopped earlier

**Actions:**
```python
# Find best checkpoint
checkpoints = {
    'epoch_14': 72.1,
    'epoch_15': 71.8,
    'epoch_16': 72.0,
    'epoch_17': 72.5,  # ← Best!
    'epoch_18': 72.3,
    'epoch_19': 71.9,
    'epoch_20': 71.7
}

best_ckpt = max(checkpoints, key=checkpoints.get)
print(f"Best: {best_ckpt} with {checkpoints[best_ckpt]:.1f}%")

# Ensemble (optional)
ensemble_acc = ensemble([ckpt_14, ckpt_17, ckpt_18])
print(f"Ensemble: {ensemble_acc:.1f}%")
# Expected: +1-2% vs single model
```

---

## 📈 **VISUALIZATION - EXPECTED CURVES**

### **Loss Curves**

```
Loss
3.5 │                 Reasoning Loss
3.0 │  ●               
2.5 │   ●             
2.0 │    ●           
1.5 │     ●●         
1.0 │       ●●●●     
0.5 │           ●●●●●________
0.0 └─────────────────────────────► Epoch
    1  3  5  7  9  11 13 15 17 19

2.5 │                 Answer Loss
2.0 │  ●
1.5 │   ●●
1.0 │     ●●●
0.5 │        ●●●●●●●________
0.0 └─────────────────────────────► Epoch
    1  3  5  7  9  11 13 15 17 19
```

### **Accuracy Curve**

```
Accuracy (%)
75 │                            ⭐⭐
70 │                      ●●●● ⭐ Goal!
65 │              ●●●●●●●
60 │        ●●●●●
55 │    ●●●
50 │  ●●
45 │ ●
40 │●
35 └────────────────────────────────► Epoch
   1  3  5  7  9  11 13 15 17 19

Phases:
├─ Warmup (1-5)
├─ Rapid (6-10)
├─ Fine-tune (11-15)
└─ Converge (16-20)
```

### **Learning Rate Schedule**

```
LR (×10⁻⁵)
5 │      ────────────┐
4 │                   ╲
3 │                    ╲
2 │                     ╲╲
1 │                       ╲╲╲___
0 └─────────────────────────────► Epoch
  1  3  5  7  9  11 13 15 17 19
  
  Warmup → Plateau → Cosine Decay
```

---

## 🔍 **MONITORING CHECKLIST**

### **Every Epoch**
```python
✓ Train loss, val loss
✓ Train accuracy, val accuracy
✓ Learning rate
✓ Gradient norm
✓ Training time
✓ GPU memory
```

### **Every 2-3 Epochs**
```python
✓ Per-type accuracy breakdown
✓ Sample predictions (manual review)
✓ Reasoning quality analysis
✓ Attention weight visualization
```

### **End of training**
```python
✓ Best checkpoint selection
✓ Ensemble evaluation
✓ Test set evaluation
✓ Error analysis
✓ Attention pattern analysis
```

---

## 🚨 **TROUBLESHOOTING GUIDE**

### **Problem 1: Loss not decreasing**

**Symptoms:**
- Loss stays high (>2.5) after Epoch 3
- Accuracy random (~25-30%)

**Solutions:**
```python
# 1. Check learning rate
if loss_not_decreasing:
    CONFIG['learning_rate'] = 1e-4  # Increase
    # or
    CONFIG['warmup_ratio'] = 0.15   # More warmup

# 2. Check gradient flow
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient: {name}")  # Should not happen

# 3. Check data
sample = next(iter(train_loader))
print(sample.keys())
assert 'labels' in sample
assert 'reasoning_labels' in sample
```

---

### **Problem 2: Overfitting early**

**Symptoms:**
- Train acc >> Val acc (gap >10%)
- Val loss increasing after Epoch 10

**Solutions:**
```python
# 1. Increase regularization
CONFIG['dropout'] = 0.15            # From 0.1
CONFIG['weight_decay'] = 0.02       # From 0.01
CONFIG['label_smoothing'] = 0.15    # From 0.1

# 2. More data augmentation
CONFIG['augment_strength'] = 'strong'

# 3. Early stopping
CONFIG['patience'] = 3  # From 5
```

---

### **Problem 3: Reasoning good, answer bad**

**Symptoms:**
- Reasoning loss < 1.0
- Answer loss > 1.0
- Answer accuracy < 65%

**Solutions:**
```python
# Rebalance loss weights
CONFIG['alpha_reasoning'] = 0.4  # Decrease from 0.6
CONFIG['alpha_answer'] = 0.6     # Increase from 0.4

# Check cross-attention
visualize_attention_weights(model)
# Should see strong attention from answer to reasoning
```

---

### **Problem 4: Complex questions fail**

**Symptoms:**
- Simple questions: 75%+
- Complex questions: <50%

**Solutions:**
```python
# 1. Curriculum learning
def curriculum_sampler(epoch):
    if epoch < 5:
        # Easy samples only
        return samples[samples['reasoning_type'] != 'COMPLEX']
    elif epoch < 10:
        # Easy + medium
        return samples[samples['reasoning_weight'] < 2.5]
    else:
        # All samples
        return samples

# 2. Oversample complex
from torch.utils.data import WeightedRandomSampler
weights = [3.0 if s['type'] == 'COMPLEX' else 1.0 for s in dataset]
sampler = WeightedRandomSampler(weights, len(dataset))
```

---

### **Problem 5: GPU OOM**

**Symptoms:**
- "CUDA out of memory" error

**Solutions:**
```python
# Option 1: Reduce batch size
CONFIG['batch_size'] = 8            # From 16
CONFIG['gradient_accumulation_steps'] = 8  # From 4
# Effective batch still 64

# Option 2: Gradient checkpointing
model.gradient_checkpointing_enable()

# Option 3: Freeze more layers
for param in model.clip_model.parameters():
    param.requires_grad = False
```

---

## 📊 **EXAMPLE RUN LOG**

```bash
[EPOCH 1/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 3.245 | Val Loss: 3.189 | Val Acc: 38.2% | LR: 1.0e-05
Reasoning Loss: 2.145 | Answer Loss: 1.100
Time: 28m 15s | GPU: 7.8GB

[EPOCH 2/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 2.687 | Val Loss: 2.623 | Val Acc: 48.5% | LR: 3.0e-05
Reasoning Loss: 1.789 | Answer Loss: 0.898
Time: 27m 42s | GPU: 7.9GB

[EPOCH 5/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 1.834 | Val Loss: 1.912 | Val Acc: 59.3% | LR: 5.0e-05
Reasoning Loss: 1.234 | Answer Loss: 0.678
Time: 27m 38s | GPU: 8.1GB
✓ Baseline established

[EPOCH 10/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 1.012 | Val Loss: 1.089 | Val Acc: 67.8% | LR: 4.0e-05
Reasoning Loss: 0.678 | Answer Loss: 0.434
Per-type: SPATIAL 71% | OBJECT 74% | COUNT 61% | COLOR 67%
Time: 27m 51s | GPU: 8.2GB
✓ 70% in sight!

[EPOCH 14/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.821 | Val Loss: 0.876 | Val Acc: 72.1% | LR: 2.0e-05
Reasoning Loss: 0.534 | Answer Loss: 0.342
Per-type: SPATIAL 75% | OBJECT 77% | COUNT 64% | COLOR 70%
Time: 27m 44s | GPU: 8.3GB
⭐ BEST CHECKPOINT! New best: 72.1%

[EPOCH 17/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.798 | Val Loss: 0.845 | Val Acc: 72.5% | LR: 8.0e-06
Reasoning Loss: 0.521 | Answer Loss: 0.324
Per-type: SPATIAL 76% | OBJECT 78% | COUNT 65% | COLOR 71%
Time: 27m 39s | GPU: 8.3GB
⭐⭐ NEW BEST! 72.5%

[EPOCH 20/20] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.789 | Val Loss: 0.851 | Val Acc: 71.8% | LR: 2.0e-06
Reasoning Loss: 0.515 | Answer Loss: 0.336
Time: 27m 41s | GPU: 8.3GB

════════════════════════════════════════════════════════
TRAINING COMPLETE! 🎉
════════════════════════════════════════════════════════
Best Model: epoch_17.pt
Best Val Accuracy: 72.5%
Total Time: 9h 14m
Target (70%): ✓✓✓ ACHIEVED!

Final Performance:
├─ SPATIAL:  76.2%
├─ OBJECT:   78.1%
├─ COUNT:    65.3%
├─ COLOR:    71.4%
├─ SCENE:    67.8%
└─ COMPLEX:  57.6%
```

---

## 🎯 **SUCCESS CRITERIA**

### **Minimum (Pass)**
- [ ] Overall accuracy ≥ 70%
- [ ] No catastrophic forgetting
- [ ] Training stable (no NaN/Inf)
- [ ] Reasoning quality ≥ 60%

### **Target (Good)**
- [ ] Overall accuracy ≥ 72%
- [ ] SPATIAL/OBJECT ≥ 75%
- [ ] COUNT/COLOR ≥ 65%
- [ ] Reasoning quality ≥ 70%

### **Stretch (Excellent)**
- [ ] Overall accuracy ≥ 74%
- [ ] All types ≥ 60%
- [ ] Reasoning quality ≥ 80%
- [ ] Ensemble ≥ 75%

---

## 🚀 **NEXT STEPS AFTER TRAINING**

### **1. Evaluation**
```python
# Full evaluation on test set
python eval_vqa_with_parser.py \
    --model checkpoints/epoch_17.pt \
    --test_file test.jsonl \
    --output results.json
```

### **2. Error Analysis**
```python
# Analyze failure cases
python analyze_errors.py \
    --predictions results.json \
    --breakdown_by_type
```

### **3. Ensemble (Optional)**
```python
# Combine top-3 checkpoints
python ensemble.py \
    --models epoch_14.pt epoch_17.pt epoch_18.pt \
    --weights 0.3 0.4 0.3
```

### **4. Test-Time Augmentation (Optional)**
```python
# Multiple passes with augmentation
python eval_with_tta.py \
    --model epoch_17.pt \
    --num_augments 5
```

---

## 💡 **PRO TIPS**

### **Training**
1. ⚡ **Save checkpoints often** - Best model may not be last epoch
2. 🔍 **Monitor reasoning quality** - Not just loss/accuracy
3. 📊 **Track per-type metrics** - Find weak spots
4. 🧪 **Manual review samples** - Every 2-3 epochs
5. 💾 **Keep top-3 checkpoints** - For ensembling

### **Optimization**
1. 🎯 **Start simple** - Concat fusion, then try fancier
2. ⚖️ **Balance losses** - Adjust alpha_reasoning/alpha_answer
3. 📈 **Curriculum learning** - Easy → hard samples
4. 🔄 **Data augmentation** - But not too aggressive
5. 🎲 **Ensemble** - Easy +1-2% boost at end

### **Debugging**
1. 🐛 **Visualize attention** - Does it make sense?
2. 📝 **Read predictions** - Manual inspection is key
3. 📉 **Plot learning curves** - Catch problems early
4. 🔬 **Gradient analysis** - Check for vanishing/exploding
5. 💬 **Sample diversity** - Make sure seeing all types

---

## 📚 **REFERENCES**

- Flamingo: https://arxiv.org/abs/2204.14198
- BLIP-2: https://arxiv.org/abs/2301.12597
- LLaVA: https://arxiv.org/abs/2304.08485
- Chain-of-Thought: https://arxiv.org/abs/2201.11903

---

**TL;DR**: 
- **Phases**: Warmup (1-5) → Rapid (6-10) → Fine-tune (11-15) → Converge (16-20)
- **Target**: 70%+ accuracy (expected: 72-73%)
- **Key**: Monitor reasoning quality, not just accuracy
- **Best checkpoint**: Usually Epoch 14-18, not last!

Good luck! 🚀🎯
