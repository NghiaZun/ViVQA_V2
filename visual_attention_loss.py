#!/usr/bin/env python3
"""
Visual Attention Regularization Loss
=====================================

FORCE model to attend to image features (correct anti-hallucination method)

Key idea:
- Extract cross-attention weights from fusion layers
- Penalize LOW attention to visual features
- Encourage model to use image information

This is BETTER than image dropout because:
- Image dropout teaches model "ignore images"
- Attention regularization teaches "MUST look at images"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualAttentionRegularization(nn.Module):
    """
    Force model to attend to visual features
    
    Computes attention entropy and coverage:
    - Low entropy = focused attention (good)
    - High coverage = looking at many image patches (good)
    - Low min attention = ignoring some patches (bad)
    """
    
    def __init__(
        self,
        min_attention_threshold: float = 0.05,
        entropy_weight: float = 0.1,
        coverage_weight: float = 0.2,
        enabled: bool = True
    ):
        super().__init__()
        self.min_attention_threshold = min_attention_threshold
        self.entropy_weight = entropy_weight
        self.coverage_weight = coverage_weight
        self.enabled = enabled
    
    def extract_cross_attention(self, model, return_dict=False):
        """
        Extract cross-attention weights from fusion layers
        
        Note: SimpleFusionVQA uses VisionFirstFusion layers
        which have vision_to_text and text_to_vision attention
        
        Returns:
            attention_weights: [B, num_heads, seq_len, 257] or None
        """
        if not hasattr(model, 'fusion_layers'):
            return None
        
        # Get attention from last fusion layer (most informative)
        last_fusion = model.fusion_layers[-1]
        
        # Access stored attention weights (need to modify VisionFirstFusion to save them)
        # For now, return None (will compute gradient-based proxy)
        return None
    
    def forward(self, model, visual_features, fused_features):
        """
        Compute visual attention regularization loss
        
        Args:
            model: VQA model with fusion layers
            visual_features: [B, 257, D] - DINOv2 features
            fused_features: [B, L, D] - fused text+vision features
        
        Returns:
            loss: scalar tensor
            loss_dict: breakdown of loss components
        """
        if not self.enabled or not self.training:
            return torch.tensor(0.0, device=visual_features.device), {}
        
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # ====================================================================
        # METHOD 1: Gradient-based attention proxy (no architecture change)
        # ====================================================================
        # Idea: Compute gradient of fused features w.r.t. visual features
        # High gradient = high attention, Low gradient = ignoring images
        
        # Compute similarity between fused and visual features
        # This is a proxy for "how much visual info is in fused features"
        visual_norm = F.normalize(visual_features, p=2, dim=-1)  # [B, 257, D]
        fused_norm = F.normalize(fused_features, p=2, dim=-1)    # [B, L, D]
        
        # Compute attention similarity (cosine similarity)
        attention_scores = torch.bmm(fused_norm, visual_norm.transpose(1, 2))  # [B, L, 257]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, L, 257]
        
        # Average over sequence length to get per-sample attention
        avg_attention = attention_weights.mean(dim=1)  # [B, 257]
        
        # ====================================================================
        # LOSS 1: Coverage Loss (encourage looking at many patches)
        # ====================================================================
        # We want attention distributed across image patches
        # Low coverage = model only looks at 1-2 patches (bad!)
        # High coverage = model scans entire image (good!)
        
        # Count patches with significant attention (>threshold)
        significant_patches = (avg_attention > self.min_attention_threshold).float()
        coverage = significant_patches.sum(dim=-1) / 257.0  # [B] in [0, 1]
        
        # Penalize low coverage (want >50% of patches attended)
        target_coverage = 0.5
        coverage_loss = F.relu(target_coverage - coverage).mean()
        
        # ====================================================================
        # LOSS 2: Entropy Regularization (prevent collapse to single patch)
        # ====================================================================
        # High entropy = uniform distribution (too diffuse)
        # Low entropy = peaked distribution (good focus)
        # But TOO low = collapse to 1 patch (bad!)
        
        # Compute entropy: -sum(p * log(p))
        eps = 1e-8
        entropy = -(avg_attention * torch.log(avg_attention + eps)).sum(dim=-1)  # [B]
        
        # Normalize by max entropy (log(257) ≈ 5.55)
        max_entropy = torch.log(torch.tensor(257.0, device=device))
        normalized_entropy = entropy / max_entropy  # [B] in [0, 1]
        
        # Penalize VERY low entropy (want >0.3)
        # This prevents collapse to 1-2 patches
        target_entropy = 0.3
        entropy_loss = F.relu(target_entropy - normalized_entropy).mean()
        
        # ====================================================================
        # TOTAL LOSS
        # ====================================================================
        total_loss = (
            self.coverage_weight * coverage_loss +
            self.entropy_weight * entropy_loss
        )
        
        loss_dict = {
            'visual_attention_loss': total_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'avg_coverage': coverage.mean().item(),
            'avg_entropy': normalized_entropy.mean().item()
        }
        
        return total_loss, loss_dict


def test_visual_attention_loss():
    """Test visual attention regularization"""
    print("Testing Visual Attention Regularization...")
    
    batch_size = 4
    seq_len = 32
    hidden_dim = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy features
    visual_features = torch.randn(batch_size, 257, hidden_dim).to(device)
    fused_features = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    
    # Test loss
    loss_fn = VisualAttentionRegularization(
        min_attention_threshold=0.05,
        entropy_weight=0.1,
        coverage_weight=0.2,
        enabled=True
    )
    
    # Mock model (just needs to exist)
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_layers = [None]  # Dummy
    
    model = MockModel().to(device)
    model.train()
    
    loss, loss_dict = loss_fn(model, visual_features, fused_features)
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  Coverage: {loss_dict['avg_coverage']:.3f} (want >0.5)")
    print(f"  Entropy: {loss_dict['avg_entropy']:.3f} (want >0.3)")
    print(f"  Coverage loss: {loss_dict['coverage_loss']:.4f}")
    print(f"  Entropy loss: {loss_dict['entropy_loss']:.4f}")
    
    # Test backward
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\nVisual Attention Regularization test passed!")


if __name__ == '__main__':
    test_visual_attention_loss()
