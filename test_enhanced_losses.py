"""
Simple test script for enhanced domain adaptation losses using dummy data
"""

import torch
import torch.nn as nn
import numpy as np
from src.losses.enhanced_da_losses import (
    coral_loss, cka_loss, adaptive_pseudo_labeling, 
    compute_prediction_uncertainty, mixup_consistency_loss,
    AdaptiveLossWeighting, ProgressiveDAScheduler,
    ConditionalDomainAdversarialLoss
)
from src.models.domain_discriminator import DomainDiscriminator


def test_enhanced_losses():
    """
    Test the enhanced domain adaptation losses with dummy data
    """
    print("="*60)
    print("Testing Enhanced Domain Adaptation Losses")
    print("="*60)
    
    # Create dummy feature tensors
    batch_size = 16  # Reduced batch size to save memory
    feature_dim = 128  # Reduced feature dimension to save memory
    
    source_features = torch.randn(batch_size, feature_dim)
    target_features = torch.randn(batch_size, feature_dim)
    
    # Test CORAL loss
    coral_l = coral_loss(source_features, target_features)
    print(f"CORAL Loss: {coral_l.item():.4f}")
    
    # Test CKA loss
    cka_l = cka_loss(source_features, target_features)
    print(f"CKA Loss: {cka_l.item():.4f}")
    
    # Test prediction uncertainty
    logits = torch.randn(batch_size, 3)  # 3 classes
    entropy, margin = compute_prediction_uncertainty(logits)
    print(f"Prediction Uncertainty - Entropy: mean={entropy.mean().item():.4f}, std={entropy.std().item():.4f}")
    print(f"Prediction Uncertainty - Margin: mean={margin.mean().item():.4f}, std={margin.std().item():.4f}")
    
    # Test adaptive pseudo-labeling
    pseudo_labels, mask = adaptive_pseudo_labeling(logits, threshold=0.7)
    print(f"Adaptive Pseudo-labeling: {mask.sum().item()}/{batch_size} samples selected for pseudo-labeling")
    
    # Test mixup consistency
    mixed_features, mixed_labels = mixup_consistency_loss(source_features, torch.randn(batch_size, 3))
    print(f"Mixup Consistency: output shapes - features {mixed_features.shape}, labels {mixed_labels.shape}")
    
    print("✓ Enhanced losses tests passed!")
    print()


def test_adaptive_loss_weighting():
    """
    Test adaptive loss weighting mechanism
    """
    print("="*60)
    print("Testing Adaptive Loss Weighting")
    print("="*60)
    
    # Create adaptive loss weighting for 4 losses
    adaptive_weighting = AdaptiveLossWeighting(num_losses=4)
    
    # Simulate some loss values over several iterations
    for i in range(10):
        # Simulate different loss values
        current_losses = [np.random.random() for _ in range(4)]
        weights = adaptive_weighting.update_weights(current_losses)
        print(f"Iteration {i+1}: Losses = {[f'{l:.3f}' for l in current_losses]}, Weights = {[f'{w:.3f}' for w in weights]}")
    
    print("✓ Adaptive loss weighting test passed!")
    print()


def test_progressive_da_scheduler():
    """
    Test progressive domain adaptation scheduler
    """
    print("="*60)
    print("Testing Progressive DA Scheduler")
    print("="*60)
    
    scheduler = ProgressiveDAScheduler(start_ratio=0.1, end_ratio=1.0, total_epochs=10)
    
    for epoch in range(10):
        ratio = scheduler.get_current_ratio(epoch)
        weight = scheduler.get_domain_weight(epoch)
        print(f"Epoch {epoch+1}: Ratio = {ratio:.3f}, DA Weight = {weight:.3f}")
    
    print("✓ Progressive DA scheduler test passed!")
    print()


def test_conditional_domain_adversarial_loss():
    """
    Test conditional domain adversarial loss (CDAN)
    """
    print("="*60)
    print("Testing Conditional Domain Adversarial Loss (CDAN)")
    print("="*60)
    
    # Create dummy features and outputs
    batch_size = 8
    feature_dim = 64
    num_classes = 3
    
    features = torch.randn(batch_size, feature_dim)
    outputs = torch.randn(batch_size, num_classes)
    
    # Create CDAN loss
    cdan_loss = ConditionalDomainAdversarialLoss(
        feature_dim=feature_dim, 
        num_classes=num_classes
    )
    
    loss_value = cdan_loss(features, outputs)
    print(f"CDAN Loss Value: {loss_value.item():.4f}")
    
    print("✓ Conditional domain adversarial loss test passed!")
    print()


def main():
    """
    Main function to run all tests
    """
    print("Running Enhanced Domain Adaptation Loss Tests")
    print("="*80)
    
    # Test enhanced losses
    test_enhanced_losses()
    
    # Test adaptive loss weighting
    test_adaptive_loss_weighting()
    
    # Test progressive DA scheduler
    test_progressive_da_scheduler()
    
    # Test CDAN
    test_conditional_domain_adversarial_loss()
    
    print("="*80)
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()