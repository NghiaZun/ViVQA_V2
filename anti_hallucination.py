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
    Advanced loss that prevents VQA hallucination
    
    Components:
    1. Base CE loss with frequency reweighting
    2. Image dropout penalty
    3. Contrastive image loss (optional)
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
        
        Formula: w(a) = 1 / log(freq(a) + c)
        - Common answers get lower weight (harder to learn)
        - Rare answers get higher weight (easier to learn)
        
        Args:
            freq_dict: {token_id: count} from training data
            vocab_size: Full vocabulary size (e.g., 40030 for BARTpho)
        """
        total = sum(freq_dict.values())
        
        # Create weights for FULL vocab (not just tokens in training data!)
        weights = torch.ones(vocab_size)
        
        for token_id, count in freq_dict.items():
            freq = count / total
            # Inverse log frequency weighting
            weights[token_id] = 1.0 / (np.log(freq * 1000 + self.freq_smoothing))
        
        # Normalize to mean = 1.0
        weights = weights / weights.mean()
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
                reduction='mean'
            )
        else:
            base_loss = F.cross_entropy(
                logits_flat, 
                labels_flat, 
                ignore_index=-100,
                reduction='mean'
            )
        
        loss_dict = {'base_loss': base_loss.item()}
        total_loss = base_loss
        
        # ====================================================================
        # 2. IMAGE DROPOUT PENALTY
        # ====================================================================
        # If training with image dropout, model should have HIGH loss
        # (because it CANNOT answer without image)
        
        if apply_dropout and self.training:
            # Check if images were dropped (all zeros)
            image_norms = pixel_values.view(batch_size, -1).norm(dim=1)
            dropped_mask = (image_norms < 0.01).float()  # [B]
            
            if dropped_mask.sum() > 0:
                # For dropped images, we want HIGHER loss (model shouldn't be confident)
                # Compute per-sample loss
                per_sample_loss = F.cross_entropy(
                    logits_flat, 
                    labels_flat, 
                    ignore_index=-100,
                    reduction='none'
                ).view(batch_size, -1).mean(dim=1)  # [B]
                
                # Penalty: If model is TOO confident without image -> punish
                # We want loss to be HIGH when image is dropped
                dropout_penalty = -per_sample_loss * dropped_mask  # Negative = we want to INCREASE loss
                dropout_penalty = dropout_penalty.mean() * self.dropout_penalty_weight
                
                total_loss = total_loss + dropout_penalty
                loss_dict['dropout_penalty'] = dropout_penalty.item()
        
        # ====================================================================
        # 3. CONTRASTIVE IMAGE LOSS (optional)
        # ====================================================================
        # If provided shuffled image predictions, ensure they're DIFFERENT
        # P(A|I,Q) >> P(A|I',Q) where I' is wrong image
        
        if contrastive_logits is not None and self.training:
            # Margin loss: correct image should have LOWER loss than wrong image
            correct_probs = F.log_softmax(logits_flat, dim=-1)
            wrong_probs = F.log_softmax(contrastive_logits.view(-1, logits.size(-1)), dim=-1)
            
            # Get probabilities of correct answer
            valid_mask = (labels_flat != -100)
            if valid_mask.sum() > 0:
                correct_answer_prob = correct_probs[valid_mask, labels_flat[valid_mask]]
                wrong_answer_prob = wrong_probs[valid_mask, labels_flat[valid_mask]]
                
                # We want: correct_prob >> wrong_prob
                # Loss = max(0, wrong_prob - correct_prob + margin)
                margin = 0.5
                contrastive_loss = F.relu(wrong_answer_prob - correct_answer_prob + margin).mean()
                
                total_loss = total_loss + self.contrastive_weight * contrastive_loss
                loss_dict['contrastive_loss'] = contrastive_loss.item()
        
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
