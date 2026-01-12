"""
Test script for enhanced domain adaptation methods using dummy data
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.bma_mil_model import BMA_MIL_Classifier
from src.feature_extractor import FeatureExtractor
from src.losses.enhanced_da_losses import (
    coral_loss, cka_loss, adaptive_pseudo_labeling, 
    compute_prediction_uncertainty, mixup_consistency_loss
)
from src.utils.enhanced_training import (
    train_model_enhanced_da, 
    train_model_enhanced_uda
)
from src.models.domain_discriminator import DomainDiscriminator
import tempfile
import os


def create_dummy_dataset(num_bags=20, num_patches_per_bag=10, feature_dim=768, num_classes=3):
    """
    Create dummy dataset for domain adaptation testing
    """
    bags = []
    labels = []
    
    for i in range(num_bags):
        # Create random patches for each bag
        bag = torch.randn(num_patches_per_bag, 3, 224, 224)  # [num_patches, channels, height, width]
        label = torch.randint(0, num_classes, (1,)).item()
        
        bags.append(bag)
        labels.append(label)
    
    bags = torch.stack(bags)  # [num_bags, num_patches, channels, height, width]
    labels = torch.tensor(labels)  # [num_bags]
    
    # Create pile IDs (for grouping bags into piles)
    pile_ids = [f"pile_{i//5}" for i in range(num_bags)]  # Group every 5 bags into a pile
    image_paths = [f"image_{i}.jpg" for i in range(num_bags)]
    
    return bags, labels, pile_ids, image_paths


def create_dummy_dataloader(num_bags=20, num_patches_per_bag=10, feature_dim=768, num_classes=3, batch_size=4):
    """
    Create dummy dataloader for domain adaptation testing
    """
    bags, labels, pile_ids, image_paths = create_dummy_dataset(num_bags, num_patches_per_bag, feature_dim, num_classes)
    
    # Create a simple dataset - we'll need to handle the variable length nature of bags
    # Since each bag has different number of patches, we'll create a custom collate function
    def collate_fn(batch):
        bags, labels, pile_ids, image_paths = zip(*batch)
        return torch.stack(bags), torch.stack(labels), list(pile_ids), list(image_paths)
    
    # Create list of samples
    dataset_samples = [(bags[i], labels[i], pile_ids[i], image_paths[i]) for i in range(len(bags))]
    dataset = TensorDataset(torch.arange(len(dataset_samples)))  # Dummy dataset
    
    # Custom dataset class to handle our specific format
    class CustomDataset:
        def __init__(self, bags, labels, pile_ids, image_paths):
            self.bags = bags
            self.labels = labels
            self.pile_ids = pile_ids
            self.image_paths = image_paths
        
        def __len__(self):
            return len(self.bags)
        
        def __getitem__(self, idx):
            return self.bags[idx], self.labels[idx], self.pile_ids[idx], self.image_paths[idx]
    
    dataset = CustomDataset(bags, labels, pile_ids, image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def test_enhanced_losses():
    """
    Test the enhanced domain adaptation losses with dummy data
    """
    print("="*60)
    print("Testing Enhanced Domain Adaptation Losses")
    print("="*60)
    
    # Create dummy feature tensors
    batch_size = 32
    feature_dim = 512
    
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
    print(f"Entropy: mean={entropy.mean().item():.4f}, std={entropy.std().item():.4f}")
    print(f"Margin: mean={margin.mean().item():.4f}, std={margin.std().item():.4f}")
    
    # Test adaptive pseudo-labeling
    pseudo_labels, mask = adaptive_pseudo_labeling(logits, threshold=0.7)
    print(f"Pseudo-labeling: {mask.sum().item()}/{batch_size} samples selected for pseudo-labeling")
    
    # Test mixup consistency
    mixed_features, mixed_labels = mixup_consistency_loss(source_features, torch.randn(batch_size, 3))
    print(f"Mixup consistency: output shapes - features {mixed_features.shape}, labels {mixed_labels.shape}")
    
    print("✓ Enhanced losses tests passed!")
    print()


def test_enhanced_da_training():
    """
    Test the enhanced domain adaptation training with dummy data
    """
    print("="*60)
    print("Testing Enhanced Domain Adaptation Training")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data loaders
    source_loader = create_dummy_dataloader(num_bags=30, batch_size=4)
    target_loader = create_dummy_dataloader(num_bags=20, batch_size=4)
    val_loader = create_dummy_dataloader(num_bags=10, batch_size=2)
    
    # Create a simple feature extractor (using random features for testing)
    class SimpleFeatureExtractor(nn.Module):
        def __init__(self, input_dim=3*224*224, output_dim=512):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(input_dim, output_dim)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            if len(x.shape) == 4:  # Single patch [C, H, W]
                x = self.flatten(x)
                x = self.linear(x)
                x = self.relu(x)
                return x
            elif len(x.shape) == 5:  # Bag of patches [N, C, H, W]
                batch_size = x.shape[0]
                flattened = self.flatten(x)
                features = self.linear(flattened)
                features = self.relu(features)
                return features
    
    # Create model
    feature_extractor = SimpleFeatureExtractor()
    model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor,
        feature_dim=512,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3,
        trainable_layers=-1
    )
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test enhanced DA training
    try:
        train_losses, val_accuracies, val_f1_scores = train_model_enhanced_da(
            model=model,
            train_loader_source=source_loader,
            train_loader_target=target_loader,
            val_loader_target=val_loader,
            num_epochs=3,  # Small number for testing
            learning_rate=1e-4,
            use_coral=True,
            use_cka=True,
            use_cdann=False,
            lambda_coral=0.1,
            lambda_cka=0.1
        )
        print(f"Enhanced DA training completed successfully!")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
        print(f"Final validation F1 score: {val_f1_scores[-1]:.4f}")
    except Exception as e:
        print(f"Enhanced DA training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test enhanced UDA training
    try:
        train_losses_uda, val_accuracies_uda, val_f1_scores_uda = train_model_enhanced_uda(
            model=model,
            train_loader_source=source_loader,
            train_loader_target=target_loader,
            val_loader_target=val_loader,
            num_epochs=3,  # Small number for testing
            learning_rate=1e-4,
            use_adaptive_pseudo=True,
            use_coral=True,
            use_cka=True,
            lambda_coral=0.1,
            lambda_cka=0.1
        )
        print(f"Enhanced UDA training completed successfully!")
        print(f"Final train loss: {train_losses_uda[-1]:.4f}")
        print(f"Final validation accuracy: {val_accuracies_uda[-1]:.4f}")
        print(f"Final validation F1 score: {val_f1_scores_uda[-1]:.4f}")
    except Exception as e:
        print(f"Enhanced UDA training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("✓ Enhanced DA training tests completed!")
    print()


def main():
    """
    Main function to run all tests
    """
    print("Running Enhanced Domain Adaptation Tests")
    print("="*80)
    
    # Test enhanced losses
    test_enhanced_losses()
    
    # Test enhanced training
    test_enhanced_da_training()
    
    print("="*80)
    print("All tests completed!")


if __name__ == "__main__":
    main()