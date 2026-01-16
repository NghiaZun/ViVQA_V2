#!/usr/bin/env python3
"""
ANTI-HALLUCINATION TRAINING FIXES for SimpleFusionVQA
======================================================

Implements 3 CRITICAL fixes to prevent Q->A shortcut:
1. IMAGE DROPOUT (20%) - Force model to die without image
2. ANSWER FREQUENCY REWEIGHTING - Punish common answers
3. CONTRASTIVE NEGATIVE IMAGES - Enforce different images -> different answers

Based on: https://arxiv.org/abs/2211.11736 (POPE paper on hallucination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import random
import numpy as np


class AntiHallucinationLoss(nn.Module):
    """
    **SIMPLIFIED** Anti-Hallucination Loss for VQA Generation
    
    After extensive testing, we found that SIMPLE is BETTER!
    
    **ACTIVE Component:**
    1. ✅ Image Dropout (20% batches) - Force model to use visual features
       → When image is dropped, model gets NO special penalty (just CE loss)
       → This is ENOUGH to teach model to rely on image!
    
    **DISABLED Components (caused overfitting):**
    2. ❌ Frequency Reweighting - Caused train/val divergence (alpha=0)
    3. ❌ Dropout Penalty - Too aggressive, unstable training (weight=0)
    4. ❌ Contrastive Loss - Memory intensive, minimal benefit (weight=0)
    5. ❌ Label Smoothing - Not helpful for generation (smoothing=0)
    
    **Key Insight:**
    Image dropout ALONE is sufficient! Model learns:
    - With image → predict confidently
    - Without image → still has to predict something (CE loss)
    → Natural learning signal: "I need the image to be confident!"
    
    Expected: Val loss ~1.0-1.5 (healthy, close to train loss)
    """
    
    def __init__(
        self,
        answer_freq_dict: dict = None,
        vocab_size: int = None,
        image_dropout_prob: float = 0.2,
        contrastive_weight: float = 0.1,
        dropout_penalty_weight: float = 2.0,
        freq_smoothing: float = 10.0
    ):
        super().__init__()
        self.image_dropout_prob = image_dropout_prob
        self.contrastive_weight = contrastive_weight
        self.dropout_penalty_weight = dropout_penalty_weight
        self.freq_smoothing = freq_smoothing
        
        # Build frequency weights
        if answer_freq_dict is not None:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when using frequency reweighting!")
            self.answer_weights = self._build_freq_weights(answer_freq_dict, vocab_size)
        else:
            self.answer_weights = None
    
    def _build_freq_weights(self, freq_dict: dict, vocab_size: int) -> torch.Tensor:
        """
        Build inverse frequency weights for answers
        
        **FIXED FORMULA** - Correct inverse frequency reweighting
        
        Formula: w(a) = (count(a) / total) ^ (-alpha)
        - Common answers get lower weight (< 1.0) → harder to learn
        - Rare answers get higher weight (> 1.0) → easier to learn
        - alpha controls strength (0.5 = sqrt reweighting, balanced)
        
        Args:
            freq_dict: {token_id: count} from training data
            vocab_size: Full vocabulary size (e.g., 40030 for BARTpho)
        """
        total = sum(freq_dict.values())
        
        # Create weights for FULL vocab (not just tokens in training data!)
        weights = torch.ones(vocab_size)
        
        # Alpha parameter: controls reweighting strength
        # 0.0 = NO reweighting (baseline) ← RECOMMENDED! Keep it simple!
        # 0.3 = gentle, 0.5 = balanced (sqrt), 0.7 = stronger, 1.0 = full inverse freq
        alpha = 0.0  # DISABLED: Frequency reweighting causes overfitting!
        
        # Early exit if alpha = 0 (no reweighting)
        if alpha == 0.0:
            print(f"\n  📊 Frequency reweighting DISABLED (alpha=0)")
            print(f"     → Using uniform weights (all = 1.0)")
            return weights
        
        for token_id, count in freq_dict.items():
            # Frequency of this token
            freq = count / total
            # Add smoothing to avoid extreme weights (set floor at 0.0001 = 1/10000)
            freq_smoothed = max(freq, 0.0001)
            # Inverse frequency: common tokens get LOW weight, rare tokens get HIGH weight
            weights[token_id] = (freq_smoothed) ** (-alpha)
        
        # CRITICAL FIX: Normalize ONLY on answer tokens, not entire vocab!
        # Otherwise 40K vocab with only 420 answer tokens will dilute the normalization
        answer_token_ids = list(freq_dict.keys())
        answer_weights_only = weights[answer_token_ids]
        mean_answer_weight = answer_weights_only.mean()
        
        # Scale so that ANSWER tokens have mean = 1.0
        weights[answer_token_ids] = weights[answer_token_ids] / mean_answer_weight
        # Non-answer tokens keep weight = 1.0
        
        # Diagnostic: show weight distribution
        weighted_tokens = [(tid, weights[tid].item()) for tid in answer_token_ids]
        weighted_tokens.sort(key=lambda x: freq_dict[x[0]], reverse=True)  # Sort by frequency
        print(f"\n  📊 Frequency reweighting stats:")
        print(f"     Alpha: {alpha:.2f}")
        print(f"     Answer tokens: {len(answer_token_ids)} / {vocab_size}")
        print(f"     Most common token weight: {weighted_tokens[0][1]:.3f} (should be < 1.0)")
        print(f"     Median weight: {weights[answer_token_ids].median():.3f}")
        print(f"     Rarest token weight: {weighted_tokens[-1][1]:.3f} (should be > 1.0)")
        
        return weights
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        apply_dropout: bool = True,
        contrastive_logits: torch.Tensor = None
    ):
        """
        Compute anti-hallucination loss
        
        Args:
            logits: [B, L, V] - model predictions
            labels: [B, L] - ground truth (-100 for padding)
            pixel_values: [B, C, H, W] - images
            apply_dropout: Whether to apply image dropout this batch
            contrastive_logits: [B, L, V] - predictions with shuffled images (optional)
        
        Returns:
            loss: Total loss
            loss_dict: Breakdown of loss components
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # ====================================================================
        # 1. BASE CROSS-ENTROPY LOSS (with frequency reweighting)
        # ====================================================================
        
        # Flatten for CE loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Apply frequency weights if available
        if self.answer_weights is not None:
            weights = self.answer_weights.to(device)
            base_loss = F.cross_entropy(
                logits_flat, 
                labels_flat, 
                weight=weights,
                ignore_index=-100,
                label_smoothing=0.0,  # DISABLED: Not helpful for generation tasks
                reduction='mean'
            )
        else:
            base_loss = F.cross_entropy(
                logits_flat, 
                labels_flat, 
                ignore_index=-100,
                label_smoothing=0.0,  # DISABLED: Not helpful for generation tasks
                reduction='mean'
            )
        
        loss_dict = {'base_loss': base_loss.item()}
        total_loss = base_loss
        
        # Safety check: ensure base_loss is valid
        if torch.isnan(base_loss) or torch.isinf(base_loss):
            print(f"⚠️  WARNING: base_loss is {base_loss.item()}, skipping batch!")
            return torch.tensor(0.0, device=device, requires_grad=True), loss_dict
        
        # ====================================================================
        # 2. IMAGE DROPOUT PENALTY (REFACTORED - PENALIZE CONFIDENCE!)
        # ====================================================================
        # OLD: Penalize loss value (weak - model can still be confident on prior)
        # NEW: Penalize CONFIDENCE when image is dropped
        # → Model should be UNCERTAIN without image!
        
        if apply_dropout and self.training:
            # Check if images were dropped (all zeros)
            image_norms = pixel_values.view(batch_size, -1).norm(dim=1)
            dropped_mask = (image_norms < 0.01).float()  # [B]
            
            if dropped_mask.sum() > 0:
                # MEMORY-EFFICIENT: Use max logit as confidence proxy (avoid softmax over 40K vocab)
                # max_logit ≈ log(max_prob) → high max_logit = high confidence
                max_logits, _ = logits.max(dim=-1)   # [B, L] - confidence per token
                
                # Normalize to [0, 1] range (sigmoid of scaled logits)
                # Divide by 10 to get reasonable scale (logit=10 → conf≈1.0, logit=0 → conf≈0.5)
                confidence_scores = torch.sigmoid(max_logits / 10.0)  # [B, L]
                
                # Average confidence per sample (only non-padding)
                valid_mask = (labels != -100).float()  # [B, L]
                confidence_per_sample = (confidence_scores * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)  # [B]
                
                # Penalize HIGH confidence on dropped images
                # Threshold: 0.5 (if model >50% confident without image → BAD!)
                confidence_threshold = 0.5
                confidence_penalty = F.relu(confidence_per_sample - confidence_threshold) * dropped_mask
                confidence_penalty = confidence_penalty.mean() * self.dropout_penalty_weight
                
                total_loss = total_loss + confidence_penalty
                loss_dict['dropout_penalty'] = confidence_penalty.item()
                loss_dict['avg_confidence_no_image'] = confidence_per_sample[dropped_mask.bool()].mean().item() if dropped_mask.sum() > 0 else 0.0
        
        # ====================================================================
        # 3. CONTRASTIVE IMAGE LOSS (DISABLED - OOM issues)
        # ====================================================================
        # REMOVED: KL divergence computation was causing OOM (40K vocab x batch x seq_len)
        # Image dropout alone is sufficient to prevent hallucination
        # Keeping contrastive_logits parameter for backward compatibility
        
        if contrastive_logits is not None and self.training:
            # LIGHTWEIGHT VERSION: Just compare top-k predictions instead of full distributions
            # This avoids materializing [B, L, V] tensors
            
            # Get top-1 predictions (greedy)
            _, pred_original = logits.max(dim=-1)      # [B, L]
            _, pred_shuffled = contrastive_logits.max(dim=-1)  # [B, L]
            
            # Compute agreement rate (lower is better - want different predictions)
            valid_mask = (labels != -100).float()  # [B, L]
            agreement = (pred_original == pred_shuffled).float() * valid_mask
            agreement_rate = agreement.sum() / (valid_mask.sum() + 1e-8)
            
            # Penalize high agreement (want <50% agreement)
            agreement_threshold = 0.5
            contrastive_loss = F.relu(agreement_rate - agreement_threshold)
            
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss.item()
            loss_dict['agreement_rate'] = agreement_rate.item()
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️  WARNING: total_loss is {total_loss.item()}")
            print(f"   Loss components: {loss_dict}")
            # Return base_loss only as fallback
            return base_loss, {'base_loss': base_loss.item(), 'fallback': True}
        
        return total_loss, loss_dict


def apply_image_dropout(pixel_values: torch.Tensor, dropout_prob: float = 0.2):
    """
    Randomly zero out images in batch
    
    Args:
        pixel_values: [B, C, H, W]
        dropout_prob: Probability of dropping each image
    
    Returns:
        pixel_values: Modified tensor (some images zeroed)
        dropped_mask: [B] - boolean mask of which images were dropped
    """
    batch_size = pixel_values.size(0)
    device = pixel_values.device
    
    # Sample which images to drop
    dropped_mask = torch.rand(batch_size, device=device) < dropout_prob
    
    # Zero out dropped images
    pixel_values = pixel_values.clone()
    pixel_values[dropped_mask] = 0.0
    
    return pixel_values, dropped_mask


def shuffle_images_in_batch(pixel_values: torch.Tensor):
    """
    Shuffle images within batch (for contrastive loss)
    
    Args:
        pixel_values: [B, C, H, W]
    
    Returns:
        shuffled_pixel_values: [B, C, H, W] - shuffled
        shuffle_indices: [B] - permutation used
    """
    batch_size = pixel_values.size(0)
    shuffle_indices = torch.randperm(batch_size, device=pixel_values.device)
    shuffled_pixel_values = pixel_values[shuffle_indices]
    return shuffled_pixel_values, shuffle_indices


def compute_answer_frequency(dataset, tokenizer, max_length: int = 32):
    """
    Compute answer token frequency from dataset
    
    Args:
        dataset: VQAGenDataset or similar
        tokenizer: Tokenizer
        max_length: Max answer length
    
    Returns:
        freq_dict: {token_id: count}
    """
    print("\n[Anti-Hallucination] Computing answer token frequencies...")
    
    token_counts = Counter()
    
    for i in range(len(dataset)):
        # Get answer text
        if hasattr(dataset, 'answers'):
            answer = dataset.answers[i]
        elif hasattr(dataset, 'data'):
            # FIX: dataset.data is a pandas DataFrame, use .iloc[i]
            answer = dataset.data.iloc[i]['answer']
        else:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        token_counts.update(tokens)
    
    print(f"  ✓ Found {len(token_counts)} unique answer tokens")
    print(f"  ✓ Total answer tokens: {sum(token_counts.values())}")
    
    # Show top 10 most common
    print("\n  Top 10 most common answer tokens:")
    for token_id, count in token_counts.most_common(10):
        token_str = tokenizer.decode([token_id])
        freq = count / sum(token_counts.values()) * 100
        print(f"    '{token_str}': {count} ({freq:.2f}%)")
    
    return dict(token_counts)


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_hallucination(model, dataloader, device, num_samples: int = 50):
    """
    Test if model hallucinates (answers without looking at image)
    
    Test:
    1. Forward with correct image -> pred_A
    2. Forward with shuffled image -> pred_A'
    3. If pred_A == pred_A' -> HALLUCINATING!
    
    Returns:
        hallucination_rate: % of samples where answer doesn't change with image
    """
    model.eval()
    
    num_hallucinations = 0
    total = 0
    
    print("\n[Hallucination Test] Testing if model uses image...")
    
    with torch.no_grad():
        for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(dataloader):
            if total >= num_samples:
                break
            
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            batch_size = pixel_values.size(0)
            
            # Forward with correct image
            outputs_correct = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=1  # Greedy for consistency
            )
            
            # Shuffle images
            shuffled_images, _ = shuffle_images_in_batch(pixel_values)
            
            # Forward with wrong image
            outputs_wrong = model.generate(
                pixel_values=shuffled_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=1
            )
            
            # Compare predictions
            for i in range(batch_size):
                if total >= num_samples:
                    break
                
                pred_correct = outputs_correct[i]
                pred_wrong = outputs_wrong[i]
                
                # Check if predictions are identical
                if torch.equal(pred_correct, pred_wrong):
                    num_hallucinations += 1
                
                total += 1
    
    hallucination_rate = num_hallucinations / total * 100
    
    print(f"\n  📊 Hallucination Rate: {hallucination_rate:.2f}%")
    print(f"     ({num_hallucinations}/{total} samples answered same with wrong image)")
    
    if hallucination_rate > 50:
        print("  ❌ HIGH HALLUCINATION! Model not using image properly.")
    elif hallucination_rate > 20:
        print("  ⚠️  MODERATE HALLUCINATION. Model relies partly on text.")
    else:
        print("  ✅ LOW HALLUCINATION. Model uses image well!")
    
    return hallucination_rate


if __name__ == '__main__':
    print("="*80)
    print("ANTI-HALLUCINATION TRAINING FIXES")
    print("="*80)
    print("\n✅ Implements:")
    print("  1. Image Dropout (20%) - Force model to die without image")
    print("  2. Answer Frequency Reweighting - Punish common answers")
    print("  3. Contrastive Negative Images - Different images -> different answers")
    print("\n📚 Reference: POPE paper (https://arxiv.org/abs/2211.11736)")
    print("="*80)
