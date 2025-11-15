"""
BMA MIL Classifier - End-to-End Architecture
Data (raw patches) -> FeatureExtractor -> MIL Aggregation -> Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_layers(model, trainable_layers):
    """
    Selectively freeze/unfreeze layers in feature extractor
    
    Args:
        model: Feature extractor model (ViT, ResNet, etc.)
        trainable_layers: 
            - 0: Freeze all layers
            - -1: All layers trainable
            - N (positive int): Make last N blocks/layers trainable, freeze rest
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    if trainable_layers == 0:
        # Fully frozen - already done above
        return
    
    elif trainable_layers == -1:
        # Fully trainable - unfreeze all
        for param in model.parameters():
            param.requires_grad = True
        return
    
    elif trainable_layers > 0:
        # Partially trainable - unfreeze last N layers/blocks
        
        # Try to detect model architecture
        if hasattr(model, 'blocks'):
            # Vision Transformer (timm models)
            total_blocks = len(model.blocks)
            blocks_to_train = min(trainable_layers, total_blocks)
            
            # Unfreeze last N blocks
            for i in range(total_blocks - blocks_to_train, total_blocks):
                for param in model.blocks[i].parameters():
                    param.requires_grad = True
            
            # Also unfreeze norm layer if exists
            if hasattr(model, 'norm'):
                for param in model.norm.parameters():
                    param.requires_grad = True
            
            print(f"  ViT: Unfroze last {blocks_to_train}/{total_blocks} blocks")
        
        elif hasattr(model, 'layer1') and hasattr(model, 'layer4'):
            # ResNet-style architecture
            layers = ['layer1', 'layer2', 'layer3', 'layer4']
            layers_to_train = min(trainable_layers, len(layers))
            
            # Unfreeze last N layers
            for layer_name in layers[-layers_to_train:]:
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Also unfreeze final layers (fc, etc.)
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            
            print(f"  ResNet: Unfroze last {layers_to_train}/{len(layers)} layers")
        
        else:
            # Generic approach: try to unfreeze last N children modules
            children = list(model.children())
            if len(children) > 0:
                modules_to_train = min(trainable_layers, len(children))
                for module in children[-modules_to_train:]:
                    for param in module.parameters():
                        param.requires_grad = True
                print(f"  Generic: Unfroze last {modules_to_train}/{len(children)} modules")
            else:
                # No children, unfreeze all
                for param in model.parameters():
                    param.requires_grad = True
                print(f"  No module structure detected, unfroze all parameters")


class AttentionAggregator(nn.Module):
    """Attention-based MIL aggregator"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, features):
        """
        Args:
            features: [num_instances, feature_dim]
        Returns:
            aggregated: [feature_dim]
            attention_weights: [num_instances]
        """
        # Compute attention weights
        attention_logits = self.attention(features)  # [num_instances, 1]
        attention_weights = F.softmax(attention_logits.squeeze(-1), dim=0)  # [num_instances]
        
        # Weighted aggregation
        weighted_features = torch.sum(features * attention_weights.unsqueeze(-1), dim=0)  # [feature_dim]
        
        # Transform aggregated features
        aggregated = self.feature_encoder(weighted_features)  # [feature_dim]
        
        return aggregated, attention_weights


class BMA_MIL_Classifier(nn.Module):
    """
    End-to-End BMA MIL Classifier
    Architecture: Raw Patches -> Feature Extraction -> MIL Aggregation -> Classification
    """

    def __init__(self, feature_extractor, feature_dim=768, hidden_dim=512, 
                 num_classes=3, dropout=0.3, trainable_layers=-1):
        """
        Args:
            feature_extractor: Pre-trained feature extractor (e.g., ViT, ResNet)
            feature_dim: Dimension of extracted features
            hidden_dim: Hidden dimension for MIL aggregation
            num_classes: Number of output classes
            dropout: Dropout rate
            trainable_layers: Feature extractor training mode
                - 0: Fully frozen
                - -1: Fully trainable
                - N (positive int): Last N blocks/layers trainable
        """
        super().__init__()
        
        # Feature extractor (integrated)
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.trainable_layers = trainable_layers
        
        # Selectively freeze/unfreeze layers
        freeze_layers(self.feature_extractor, trainable_layers)
        
        # MIL aggregation layer
        self.aggregator = AttentionAggregator(feature_dim, hidden_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, bag):
        """
        Forward pass for one bag (image)
        
        Args:
            bag: Tensor [num_patches, 3, H, W] - raw patches from one image
        
        Returns:
            logits: Tensor [num_classes] - classification logits
            attention_weights: Tensor [num_patches] - attention weights
        """
        num_patches = bag.shape[0]
        
        # Step 1: Extract features from each patch
        if self.training and not any(p.requires_grad for p in self.feature_extractor.parameters()):
            # Feature extractor is frozen, use no_grad for efficiency
            with torch.no_grad():
                patch_features = self.feature_extractor(bag)  # [num_patches, feature_dim]
        else:
            # Feature extractor is trainable or in eval mode
            patch_features = self.feature_extractor(bag)  # [num_patches, feature_dim]
        
        # Step 2: MIL aggregation (attention-based)
        bag_feature, attention_weights = self.aggregator(patch_features)  # [feature_dim], [num_patches]
        
        logits = self.classifier(bag_feature)
        return logits, attention_weights

    def forward_with_features(self, bag):
        num_patches = bag.shape[0]
        if self.training and not any(p.requires_grad for p in self.feature_extractor.parameters()):
            with torch.no_grad():
                patch_features = self.feature_extractor(bag)
        else:
            patch_features = self.feature_extractor(bag)
        bag_feature, attention_weights = self.aggregator(patch_features)
        logits = self.classifier(bag_feature)
        return logits, attention_weights, bag_feature
    
    def predict_bag(self, bag):
        """
        Predict class for a single bag
        
        Args:
            bag: Tensor [num_patches, 3, H, W]
        
        Returns:
            pred_class: int - predicted class
            probs: Tensor [num_classes] - class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(bag)
            probs = F.softmax(logits, dim=0)
            pred_class = torch.argmax(probs).item()
        return pred_class, probs
