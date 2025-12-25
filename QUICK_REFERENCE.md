# 🎯 QUICK REFERENCE - Training Monitor

## 📊 **EPOCH-BY-EPOCH TARGETS**

```
┌─────┬────────┬──────────┬─────────┬──────────┬───────────────┐
│ EP  │   LR   │ Reas Los │ Ans Los │ Val Acc  │ What to Watch │
├─────┼────────┼──────────┼─────────┼──────────┼───────────────┤
│  1  │ 1e-5   │   3.5    │   2.8   │  35-40%  │ Loss drops    │
│  2  │ 3e-5   │   2.8    │   2.2   │  45-50%  │ Learning      │
│  3  │ 5e-5   │   2.4    │   1.9   │  52-55%  │ Patterns      │
│  4  │ 5e-5   │   2.1    │   1.7   │  55-58%  │ Stable        │
│  5  │ 5e-5   │   1.9    │   1.5   │  58-60%  │ ✓ Baseline    │
├─────┼────────┼──────────┼─────────┼──────────┼───────────────┤
│  6  │ 5e-5   │   1.7    │   1.3   │  60-62%  │ Rapid gain    │
│  7  │ 5e-5   │   1.5    │   1.2   │  62-64%  │ Momentum      │
│  8  │ 5e-5   │   1.3    │   1.0   │  64-66%  │ ✓ Milestone   │
│  9  │ 4.5e-5 │   1.2    │   0.95  │  66-67%  │ Fine-tune     │
│ 10  │ 4e-5   │   1.1    │   0.90  │  67-68%  │ ✓ Near goal   │
├─────┼────────┼──────────┼─────────┼──────────┼───────────────┤
│ 11  │ 3.5e-5 │   1.0    │   0.85  │  68-69%  │ Refine        │
│ 12  │ 3e-5   │   0.95   │   0.82  │  69-70%  │ ✓✓ TARGET!    │
│ 13  │ 2.5e-5 │   0.90   │   0.80  │  70-71%  │ Push higher   │
│ 14  │ 2e-5   │   0.88   │   0.78  │  71-72%  │ ⭐ Peak       │
│ 15  │ 1.5e-5 │   0.87   │   0.77  │  71-72%  │ Stable        │
├─────┼────────┼──────────┼─────────┼──────────┼───────────────┤
│ 16  │ 1e-5   │   0.86   │   0.76  │  71-72%  │ Tweaks        │
│ 17  │ 8e-6   │   0.85   │   0.75  │  72-73%  │ ⭐⭐ BEST?    │
│ 18  │ 6e-6   │   0.85   │   0.75  │  72-73%  │ Plateau       │
│ 19  │ 4e-6   │   0.85   │   0.75  │  71-72%  │ May decline   │
│ 20  │ 2e-6   │   0.85   │   0.75  │  71-72%  │ ✓ Complete    │
└─────┴────────┴──────────┴─────────┴──────────┴───────────────┘
```

## 🚨 **RED FLAGS & FIXES**

### Epoch 1-5
```
❌ Loss > 3.0 at Epoch 3
   → Increase LR to 1e-4 or warmup_ratio to 0.15

❌ Accuracy < 40% at Epoch 3
   → Check data loading, verify labels

❌ NaN/Inf in loss
   → Reduce LR, check gradient clipping (max_norm=1.0)

❌ OOM error
   → Reduce batch_size to 8, keep gradient_accum=8
```

### Epoch 6-10
```
❌ Accuracy plateau at 60%
   → Check reasoning quality, may need adjust alpha weights

❌ Reasoning loss stuck at 1.5+
   → Model not learning to reason, check cross-attention

❌ Answer loss stuck at 1.2+
   → Increase alpha_answer from 0.4 to 0.5
```

### Epoch 11-15
```
❌ Val loss > Train loss + 0.3
   → Overfitting! Increase dropout/weight_decay

❌ No improvement after Epoch 12
   → Early stop, use Epoch 12 checkpoint

❌ Accuracy drops
   → LR too low, or overfitting
```

### Epoch 16-20
```
❌ Accuracy drops > 1%
   → Overfitting, use earlier checkpoint

❌ Loss increasing
   → Stop training, use best checkpoint
```

## ✅ **GREEN LIGHTS**

```
✓ Loss decreasing steadily
✓ Accuracy increasing every 2-3 epochs
✓ Val loss ≈ Train loss (gap < 0.2)
✓ Gradient norm: 0.5-2.0
✓ GPU memory stable
✓ No NaN/Inf
✓ Reasoning quality improving (manual check)
```

## 📋 **DAILY CHECKLIST**

### Morning (Start training)
```bash
□ Clear GPU memory: nvidia-smi
□ Check disk space: df -h
□ Backup previous checkpoints
□ Start training: nohup python new_train.py > train.log 2>&1 &
□ Monitor initial epochs: tail -f train.log
```

### Afternoon (Check progress)
```bash
□ Check current epoch: tail -20 train.log
□ Compare with expected metrics
□ GPU stable: nvidia-smi
□ Review sample predictions
```

### Evening (Daily review)
```bash
□ Plot learning curves
□ Compare with roadmap targets
□ Identify any red flags
□ Adjust config if needed (restart tomorrow)
□ Backup checkpoints
```

## 🎯 **MILESTONES**

```
Epoch 5:  60% accuracy → Baseline ✓
Epoch 8:  65% accuracy → Milestone ✓
Epoch 12: 70% accuracy → TARGET ✓✓
Epoch 14: 72% accuracy → Peak ⭐
Epoch 17: 73% accuracy → Best ⭐⭐
```

## 🔧 **QUICK COMMANDS**

### Monitor training
```bash
# Watch live
tail -f train.log

# Check GPU
watch -n 1 nvidia-smi

# Check last epoch
tail -50 train.log | grep "EPOCH"

# Check best accuracy
grep "BEST" train.log
```

### Emergency fixes
```bash
# Kill training
pkill -f new_train.py

# Clear GPU
nvidia-smi | grep python | awk '{print $5}' | xargs kill -9

# Resume from checkpoint
python new_train.py --resume checkpoints/last.pt
```

### Analyze results
```bash
# Plot curves
python plot_training.py --log train.log

# Per-type accuracy
python analyze_types.py --checkpoint checkpoints/best.pt

# Error analysis
python analyze_errors.py --predictions val_predictions.json
```

## 📊 **EXPECTED PER-TYPE ACCURACY**

```
By Epoch 5:
SPATIAL:  62-65%
OBJECT:   65-68%
COUNT:    52-55%
COLOR:    58-60%
COMPLEX:  42-45%

By Epoch 10:
SPATIAL:  70-72%
OBJECT:   72-75%
COUNT:    60-62%
COLOR:    65-68%
COMPLEX:  48-52%

By Epoch 14 (Peak):
SPATIAL:  74-77%
OBJECT:   76-79%
COUNT:    63-66%
COLOR:    69-72%
COMPLEX:  55-60%

By Epoch 17 (Best):
SPATIAL:  75-78%
OBJECT:   77-80%
COUNT:    64-67%
COLOR:    70-73%
COMPLEX:  56-61%
```

## 🎓 **MANUAL REVIEW QUESTIONS**

Every 2-3 epochs, check 10-20 samples:

### Reasoning Quality
```
□ Is reasoning grammatically correct?
□ Does reasoning describe what's in image?
□ Is reasoning relevant to question?
□ Does reasoning lead to answer?
```

### Answer Quality
```
□ Is answer correct?
□ Is answer aligned with reasoning?
□ Is answer concise (not too long)?
□ Is answer in Vietnamese?
```

### Cross-Attention
```
□ Visualize attention weights
□ Does answer attend to relevant reasoning parts?
□ Are attention patterns making sense?
```

## 💾 **CHECKPOINT MANAGEMENT**

### Keep these checkpoints
```
✓ best_model.pt          (highest val acc)
✓ epoch_5.pt             (baseline)
✓ epoch_10.pt            (milestone)
✓ epoch_14.pt            (peak)
✓ epoch_17.pt            (potential best)
✓ last.pt                (for resume)
```

### Delete these (save space)
```
✗ epoch_1.pt through epoch_4.pt  (warmup)
✗ epoch_6.pt through epoch_9.pt  (intermediate)
✗ epoch_11.pt through epoch_13.pt (minor)
✗ epoch_15.pt, 16.pt, 18-20.pt  (converge)
```

## 🎯 **FINAL CHECKLIST**

### Before deployment
```
□ Best checkpoint identified
□ Test set accuracy ≥ 70%
□ Per-type accuracy reviewed
□ Error analysis done
□ Reasoning quality ≥ 70%
□ Attention weights make sense
□ No major failure modes
□ Model size acceptable
□ Inference speed acceptable
```

### Bonus improvements
```
□ Ensemble top-3 checkpoints (+1-2%)
□ Test-time augmentation (+0.5-1%)
□ Post-processing rules (+0.5%)
□ Confidence calibration
```

---

**Print this and stick on your monitor! 📌**
