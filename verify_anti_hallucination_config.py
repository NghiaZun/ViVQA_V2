#!/usr/bin/env python3
"""
Quick verification: Check if anti-hallucination config is correct
Run this before training to avoid wasting time!
"""

import sys
sys.path.insert(0, '/home/nghia-duong/ViVQA_V2')

from anti_hallucination import AntiHallucinationLoss
import torch

print("="*70)
print("ANTI-HALLUCINATION CONFIGURATION CHECK")
print("="*70)

# Create loss with dummy freq dict
dummy_freq = {1: 100, 2: 50, 3: 25}
loss_fn = AntiHallucinationLoss(
    answer_freq_dict=dummy_freq,
    vocab_size=100,
    image_dropout_prob=0.2,
    contrastive_weight=0.0,
    dropout_penalty_weight=0.0,
    freq_smoothing=5.0
)

print("\n✅ CHECKING CONFIGURATION:")
print(f"   Image Dropout Prob: {loss_fn.image_dropout_prob}")
print(f"   Contrastive Weight: {loss_fn.contrastive_weight}")
print(f"   Dropout Penalty Weight: {loss_fn.dropout_penalty_weight}")

print("\n✅ EXPECTED VALUES:")
print(f"   Image Dropout Prob: 0.2 ✓")
print(f"   Contrastive Weight: 0.0 ✓")
print(f"   Dropout Penalty Weight: 0.0 ✓")

# Check weights
if loss_fn.answer_weights is not None:
    weights = loss_fn.answer_weights
    print(f"\n✅ FREQUENCY WEIGHTS:")
    print(f"   Vocab size: {len(weights)}")
    print(f"   Token 1 weight: {weights[1]:.3f}")
    print(f"   Token 2 weight: {weights[2]:.3f}")
    print(f"   Token 3 weight: {weights[3]:.3f}")
    print(f"   Non-answer token (5) weight: {weights[5]:.3f}")
    
    # ALL should be 1.0 since alpha=0
    if torch.allclose(weights, torch.ones_like(weights)):
        print("\n   ✅ ALL WEIGHTS = 1.0 (freq reweighting disabled)")
    else:
        print("\n   ⚠️  WARNING: Weights are NOT uniform!")
        print(f"      This means frequency reweighting is still active!")
        print(f"      Check alpha in anti_hallucination.py (should be 0.0)")

print("\n✅ TEST FORWARD PASS:")
# Dummy data
batch_size = 2
seq_len = 5
vocab_size = 100

logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))
labels[0, -1] = -100  # padding
pixel_values = torch.randn(batch_size, 3, 224, 224)

# Forward
loss, loss_dict = loss_fn(
    logits=logits,
    labels=labels,
    pixel_values=pixel_values,
    apply_dropout=False,
    contrastive_logits=None
)

print(f"   Loss: {loss.item():.4f}")
print(f"   Loss components: {loss_dict}")

# Check if only base_loss exists
if len(loss_dict) == 1 and 'base_loss' in loss_dict:
    print("\n   ✅ ONLY base_loss active (simplified version)")
else:
    print("\n   ⚠️  WARNING: Extra loss components detected!")
    print(f"      Expected: ['base_loss']")
    print(f"      Got: {list(loss_dict.keys())}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE!")
print("="*70)

# Final check
all_good = (
    loss_fn.image_dropout_prob == 0.2 and
    loss_fn.contrastive_weight == 0.0 and
    loss_fn.dropout_penalty_weight == 0.0 and
    len(loss_dict) == 1 and
    'base_loss' in loss_dict
)

if all_good:
    print("\n✅ ALL CHECKS PASSED! Configuration is correct.")
    print("   You can start training now!")
    sys.exit(0)
else:
    print("\n❌ SOME CHECKS FAILED! Please review configuration.")
    print("   Do NOT start training until all checks pass!")
    sys.exit(1)
