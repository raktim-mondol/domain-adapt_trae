"""
Feature extraction using ViT-R50 model
"""

import torch
import timm


class FeatureExtractor:
    """Feature extraction using ViT-R50 model"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 trainable_layers=0):
        """
        Initialize FeatureExtractor with option to make last few layers trainable
        
        Args:
            device: Device to run the model on
            trainable_layers: Number of last transformer blocks to make trainable
                            0 = fully frozen (default)
                            1-12 = make last N transformer blocks trainable
        """
        self.device = device
        self.trainable_layers = trainable_layers
        
        self.model = timm.create_model(
            'vit_base_r50_s16_224.orig_in21k',
            pretrained=True,
            num_classes=0  # Remove classifier
        )
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Make last few transformer blocks trainable if requested
        if trainable_layers > 0:
            self._make_last_layers_trainable(trainable_layers)
        
        self.model = self.model.to(device)
        
        # Set appropriate mode based on trainability
        if trainable_layers > 0:
            self.model.train()  # Enable training mode for trainable layers
        else:
            self.model.eval()   # Keep eval mode for fully frozen

        # Get model transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
    
    def _make_last_layers_trainable(self, num_layers):
        """Make the last num_layers transformer blocks trainable"""
        total_blocks = len(self.model.blocks)
        
        if num_layers > total_blocks:
            print(f"Warning: Requested {num_layers} trainable layers, but model only has {total_blocks} blocks.")
            print(f"Making all {total_blocks} transformer blocks trainable.")
            num_layers = total_blocks
        
        # Make last num_layers blocks trainable
        for i in range(total_blocks - num_layers, total_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True
        
        # Also make the final layer norm trainable if any layers are trainable
        if hasattr(self.model, 'norm'):
            for param in self.model.norm.parameters():
                param.requires_grad = True
        
        print(f"Made last {num_layers} transformer blocks + final norm trainable")
        
        # Print trainable parameter summary
        self._print_trainable_summary()
    
    def _print_trainable_summary(self):
        """Print summary of trainable vs frozen parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nFeature Extractor Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        
        if trainable_params > 0:
            print(f"  Status: PARTIALLY TRAINABLE (last {self.trainable_layers} blocks)")
        else:
            print(f"  Status: FULLY FROZEN")

    def extract_features(self, patches):
        """Extract features from list of PIL patches
        
        Features are extracted on GPU but returned on CPU to save GPU memory.
        This allows preprocessing to use CPU memory while training uses GPU.
        """
        features = []

        # Use no_grad only if model is fully frozen
        if self.trainable_layers == 0:
            with torch.no_grad():
                for patch in patches:
                    # Apply transforms
                    tensor_patch = self.transform(patch).unsqueeze(0).to(self.device)

                    # Extract features on GPU
                    feature = self.model(tensor_patch)
                    # Move to CPU to save GPU memory during data loading
                    features.append(feature.cpu())
        else:
            # Allow gradients for trainable layers
            for patch in patches:
                # Apply transforms
                tensor_patch = self.transform(patch).unsqueeze(0).to(self.device)

                # Extract features (gradients enabled for trainable layers)
                feature = self.model(tensor_patch)
                # Move to CPU to save GPU memory during data loading
                features.append(feature.cpu())

        if features:
            result = torch.cat(features, dim=0)  # Shape: [num_patches, feature_dim]
            # Return features on CPU - they will be moved to GPU in training loop
            return result
        else:
            return torch.tensor([], device='cpu')
