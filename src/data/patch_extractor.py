"""
Patch extraction from full-resolution images
"""

from PIL import Image


class PatchExtractor:
    """Extract 12 patches from 4032x3024 images"""

    def __init__(self, patch_size=1008, target_size=224, augmentation=None, 
                 include_original_and_augmented=False, num_augmentation_versions=1):
        self.patch_size = patch_size
        self.target_size = target_size
        self.num_patches_per_image = 12
        self.augmentation = augmentation
        self.include_original_and_augmented = include_original_and_augmented
        self.num_augmentation_versions = num_augmentation_versions

    def extract_patches(self, image_path):
        """
        Extract patches from a single image with optional augmentation.
        
        Supports multiple augmentation versions per patch:
        - num_augmentation_versions=1: Standard (12 patches, or 24 if include_original=True)
        - num_augmentation_versions=3: 3 versions per patch (36 augmented, or 48 with originals)
        
        Returns:
            List of PIL Image patches
            Total patches = 12 * num_augmentation_versions (+ 12 if include_original_and_augmented)
        """
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')

            # Verify image dimensions
            if img.size != (4032, 3024):
                # Resize if needed
                img = img.resize((4032, 3024))

            patches = []

            # Extract 12 patches (3 rows × 4 columns)
            for row in range(3):
                for col in range(4):
                    left = col * self.patch_size
                    upper = row * self.patch_size
                    right = left + self.patch_size
                    lower = upper + self.patch_size

                    # Ensure we don't exceed image boundaries
                    if right <= 4032 and lower <= 3024:
                        patch = img.crop((left, upper, right, lower))
                        # Resize to target size for ViT
                        patch = patch.resize((self.target_size, self.target_size))
                        
                        if self.augmentation is not None:
                            # Add original patch first (if requested)
                            if self.include_original_and_augmented:
                                patches.append(patch.copy())
                            
                            # Add multiple augmented versions
                            for aug_version in range(self.num_augmentation_versions):
                                # Each call to augmentation produces a different random augmentation
                                augmented_patch = self.augmentation(patch.copy())
                                patches.append(augmented_patch)
                        else:
                            # No augmentation - just add original
                            patches.append(patch)

            return patches

        except Exception as e:
            print(f"         [ERROR] Error processing {image_path}: {e}")
            return []
