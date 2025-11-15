"""
PyTorch Dataset for BMA MIL Classification
Returns individual bags (images) with pile-level labels
"""

import os
import torch
from torch.utils.data import Dataset
from .patch_extractor import PatchExtractor


class BMADataset(Dataset):
    """
    Dataset that returns individual bags (images as bags of patches).
    Each bag belongs to a pile and inherits the pile-level label.
    """

    def __init__(self, image_data_list, image_dir, augmentation=None, is_training=True, 
                 include_original_and_augmented=False, num_augmentation_versions=1):
        """
        Args:
            image_data_list: List of dicts with keys: 'image_path', 'pile_id', 'pile_label'
            image_dir: Root directory containing images
            augmentation: Augmentation pipeline (optional)
            is_training: Whether this is training set (for augmentation)
            include_original_and_augmented: If True, returns both original and augmented patches
            num_augmentation_versions: Number of augmented versions per patch (default=1)
        """
        self.image_data_list = image_data_list
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
        return len(self.image_data_list)

    def __getitem__(self, idx):
        """
        Returns:
            patches: Tensor [num_patches, 3, H, W] - raw patches from one image
                     num_patches depends on num_augmentation_versions and include_original_and_augmented
            label: int - pile-level label (same for all patches, including all augmented versions)
            pile_id: str - pile identifier
            image_path: str - image filename for tracking
        """
        image_data = self.image_data_list[idx]
        image_path = image_data['image_path']
        pile_id = image_data['pile_id']
        label = image_data['pile_label']
        
        full_path = os.path.join(self.image_dir, image_path)
        
        # Extract patches (returns list of PIL Images)
        # Total = 12 * num_augmentation_versions (+ 12 if include_original_and_augmented)
        patches = self.patch_extractor.extract_patches(full_path)
        
        if patches is None or len(patches) == 0:
            # Return empty tensor if extraction failed
            # Calculate expected number of patches
            if self.is_training:
                num_patches = 12 * self.num_augmentation_versions
                if self.include_original_and_augmented:
                    num_patches += 12
            else:
                num_patches = 12
            return torch.zeros(num_patches, 3, 224, 224), label, pile_id, image_path
        
        # Convert PIL patches to tensors
        patch_tensors = []
        for patch in patches:
            # Convert PIL to tensor with normalization
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            patch_tensor = transform(patch)
            patch_tensors.append(patch_tensor)
        
        patches_tensor = torch.stack(patch_tensors, dim=0)  # [num_patches, 3, H, W]
        
        return patches_tensor, label, pile_id, image_path


def create_bag_dataset_from_piles(df, pile_ids, image_dir, augmentation=None, is_training=True, 
                                  max_images_per_pile=None, include_original_and_augmented=False,
                                  num_augmentation_versions=1):
    """
    Helper function to create a dataset from a list of pile IDs.
    
    Args:
        df: DataFrame with columns ['pile', 'image_path', 'BMA_label']
        pile_ids: List of pile IDs to include
        image_dir: Root directory containing images
        augmentation: Augmentation pipeline
        is_training: Whether this is training set
        max_images_per_pile: Maximum images per pile (optional)
        include_original_and_augmented: If True, includes both original and augmented patches
    
    Returns:
        BMADataset containing bags from the specified piles
    """
    # Filter dataframe to only include specified piles
    pile_df = df[df['pile'].isin(pile_ids)].copy()
    
    # Create image data list
    image_data_list = []
    
    for pile_id in pile_ids:
        pile_images = pile_df[pile_df['pile'] == pile_id]
        
        if len(pile_images) == 0:
            continue
        
        # Get pile label (same for all images in pile)
        pile_label = pile_images['BMA_label'].iloc[0] - 1  # Convert to 0-indexed
        
        # Get images (limit if specified)
        image_paths = pile_images['image_path'].tolist()
        if max_images_per_pile is not None:
            image_paths = image_paths[:max_images_per_pile]
        
        # Add each image as a separate bag
        for img_path in image_paths:
            image_data_list.append({
                'image_path': img_path,
                'pile_id': pile_id,
                'pile_label': pile_label
            })
    
    return BMADataset(image_data_list, image_dir, augmentation, is_training, 
                     include_original_and_augmented, num_augmentation_versions)
