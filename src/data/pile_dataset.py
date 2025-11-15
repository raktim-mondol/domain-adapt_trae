"""
PyTorch Dataset for Pile-Level Training
Returns entire piles (all bags/images from a pile) with pile-level labels
"""

import os
import torch
from torch.utils.data import Dataset
from .patch_extractor import PatchExtractor


class BMAPileDataset(Dataset):
    """
    Dataset that returns entire piles (all bags from a pile).
    Each pile contains multiple bags (images), and each bag contains patches.
    
    This is used for pile-level training where:
    - Forward pass processes all bags in a pile
    - Loss is computed at pile level (one label per pile)
    - Gradients flow back through all bags in the pile
    """

    def __init__(self, pile_data_list, image_dir, augmentation=None, is_training=True,
                 include_original_and_augmented=False, num_augmentation_versions=1):
        """
        Args:
            pile_data_list: List of dicts with keys: 
                - 'pile_id': str
                - 'image_paths': list of image paths
                - 'pile_label': int (0-indexed)
            image_dir: Root directory containing images
            augmentation: Augmentation pipeline (optional)
            is_training: Whether this is training set (for augmentation)
            include_original_and_augmented: If True, returns both original and augmented patches
            num_augmentation_versions: Number of augmented versions per patch (default=1)
        """
        self.pile_data_list = pile_data_list
        self.image_dir = image_dir
        self.is_training = is_training
        self.include_original_and_augmented = include_original_and_augmented
        self.num_augmentation_versions = num_augmentation_versions
        
        # Initialize patch extractor with augmentation
        self.patch_extractor = PatchExtractor(
            augmentation=augmentation if is_training else None,
            include_original_and_augmented=include_original_and_augmented and is_training,
            num_augmentation_versions=num_augmentation_versions if is_training else 1
        )

    def __len__(self):
        return len(self.pile_data_list)

    def __getitem__(self, idx):
        """
        Returns:
            pile_bags: List of tensors, each [num_patches, 3, H, W] - all bags from pile
            label: int - pile-level label
            pile_id: str - pile identifier
            num_bags: int - number of bags in this pile
        """
        pile_data = self.pile_data_list[idx]
        pile_id = pile_data['pile_id']
        image_paths = pile_data['image_paths']
        label = pile_data['pile_label']
        
        pile_bags = []
        
        # Process each image (bag) in the pile
        for image_path in image_paths:
            full_path = os.path.join(self.image_dir, image_path)
            
            # Extract patches for this bag
            patches = self.patch_extractor.extract_patches(full_path)
            
            if patches is None or len(patches) == 0:
                # Skip failed images
                continue
            
            # Convert PIL patches to tensors
            patch_tensors = []
            for patch in patches:
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                patch_tensor = transform(patch)
                patch_tensors.append(patch_tensor)
            
            patches_tensor = torch.stack(patch_tensors)  # [num_patches, 3, H, W]
            pile_bags.append(patches_tensor)
        
        # If all bags failed, return a dummy bag
        if len(pile_bags) == 0:
            if self.is_training:
                num_patches = 12 * self.num_augmentation_versions
                if self.include_original_and_augmented:
                    num_patches += 12
            else:
                num_patches = 12
            pile_bags = [torch.zeros(num_patches, 3, 224, 224)]
        
        return pile_bags, label, pile_id, len(pile_bags)


def collate_pile_batch(batch):
    """
    Custom collate function for pile-level batching.
    Since piles have varying numbers of bags, we return lists instead of stacked tensors.
    
    Args:
        batch: List of tuples (pile_bags, label, pile_id, num_bags)
    
    Returns:
        pile_bags_batch: List of lists of tensors (one list per pile)
        labels: Tensor of pile labels
        pile_ids: List of pile IDs
        num_bags: List of bag counts per pile
    """
    pile_bags_batch = []
    labels = []
    pile_ids = []
    num_bags = []
    
    for pile_bags, label, pile_id, n_bags in batch:
        pile_bags_batch.append(pile_bags)
        labels.append(label)
        pile_ids.append(pile_id)
        num_bags.append(n_bags)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return pile_bags_batch, labels, pile_ids, num_bags


def create_pile_dataset_from_piles(df, pile_ids, image_dir, augmentation=None, is_training=True, 
                                  max_images_per_pile=None, include_original_and_augmented=False,
                                  num_augmentation_versions=1):
    """
    Create a pile-level dataset from a list of pile IDs.
    
    Args:
        df: DataFrame with columns ['pile', 'image_path', 'BMA_label']
        pile_ids: List of pile IDs to include
        image_dir: Root directory containing images
        augmentation: Augmentation pipeline
        is_training: Whether this is training set
        max_images_per_pile: Maximum images per pile (optional)
        include_original_and_augmented: If True, includes both original and augmented patches
    
    Returns:
        BMAPileDataset containing piles
    """
    pile_data_list = []
    
    for pile_id in pile_ids:
        pile_images = df[df['pile'] == pile_id]
        
        if len(pile_images) == 0:
            continue
        
        # Get pile label (same for all images in pile)
        pile_label = pile_images['BMA_label'].iloc[0] - 1  # Convert to 0-indexed
        
        # Get all image paths for this pile
        image_paths = pile_images['image_path'].tolist()
        
        # Limit images per pile if specified
        if max_images_per_pile is not None:
            image_paths = image_paths[:max_images_per_pile]
        
        pile_data_list.append({
            'pile_id': pile_id,
            'image_paths': image_paths,
            'pile_label': pile_label
        })
    
    return BMAPileDataset(pile_data_list, image_dir, augmentation, is_training,
                         include_original_and_augmented, num_augmentation_versions)

