"""
Test script to visualize and validate data augmentation pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from augmentation import (
    HistogramNormalizer,
    GeometricAugmentation,
    ColorAugmentation,
    NoiseAndBlurAugmentation,
    get_augmentation_pipeline
)


def test_histogram_normalization(image_path):
    """Test different histogram normalization methods"""
    print("Testing Histogram Normalization...")
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    methods = ['none', 'clahe', 'adaptive', 'standard']
    fig, axes = plt.subplots(1, len(methods), figsize=(20, 5))
    
    for idx, method in enumerate(methods):
        normalizer = HistogramNormalizer(method=method)
        normalized_img = normalizer(img.copy())
        axes[idx].imshow(normalized_img)
        axes[idx].set_title(f'Histogram: {method.upper()}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_histogram_normalization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: test_histogram_normalization.png")
    plt.close()


def test_geometric_augmentation(image_path):
    """Test geometric transformations"""
    print("\nTesting Geometric Augmentation...")
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    geo_aug = GeometricAugmentation(
        rotation_range=30,
        zoom_range=(0.8, 1.2),
        shear_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        probability=1.0  # Always apply for testing
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx in range(1, 8):
        aug_img = geo_aug(img.copy())
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Augmented {idx}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_geometric_augmentation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: test_geometric_augmentation.png")
    plt.close()


def test_color_augmentation(image_path):
    """Test color transformations"""
    print("\nTesting Color Augmentation...")
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    color_aug = ColorAugmentation(
        brightness_range=(0.6, 1.4),
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_range=(-0.2, 0.2),
        probability=1.0  # Always apply for testing
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx in range(1, 8):
        aug_img = color_aug(img.copy())
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Augmented {idx}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_color_augmentation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: test_color_augmentation.png")
    plt.close()


def test_noise_and_blur(image_path):
    """Test noise and blur augmentation"""
    print("\nTesting Noise and Blur Augmentation...")
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    noise_aug = NoiseAndBlurAugmentation(
        gaussian_noise_std=0.02,
        gaussian_blur_sigma=(0.5, 3.0),
        probability=1.0  # Always apply for testing
    )
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx in range(1, 8):
        aug_img = noise_aug(img.copy())
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Augmented {idx}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_noise_blur_augmentation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: test_noise_blur_augmentation.png")
    plt.close()


def test_full_pipeline(image_path):
    """Test complete augmentation pipeline"""
    print("\nTesting Full Augmentation Pipeline...")
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Training pipeline (all augmentations)
    train_aug = get_augmentation_pipeline(is_training=True)
    
    # Validation pipeline (histogram only)
    val_aug = get_augmentation_pipeline(is_training=False)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Validation augmentation (histogram only)
    axes[0, 1].imshow(val_aug(img.copy()))
    axes[0, 1].set_title('Val: Histogram Only', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Hide remaining in first row
    for i in range(2, 5):
        axes[0, i].axis('off')
    
    # Training augmentations
    for row in range(1, 3):
        for col in range(5):
            aug_img = train_aug(img.copy())
            axes[row, col].imshow(aug_img)
            axes[row, col].set_title(f'Train Aug {(row-1)*5 + col + 1}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.suptitle('Full Augmentation Pipeline Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_full_pipeline.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: test_full_pipeline.png")
    plt.close()


def test_patch_augmentation(image_path):
    """Test augmentation on actual patches from the image"""
    print("\nTesting Patch-Level Augmentation...")
    
    from bma_mil_classifier import PatchExtractor
    
    # Extract patches without augmentation
    patch_extractor_no_aug = PatchExtractor(augmentation=None)
    patches_original = patch_extractor_no_aug.extract_patches(image_path)
    
    # Extract patches with augmentation
    train_aug = get_augmentation_pipeline(is_training=True)
    patch_extractor_aug = PatchExtractor(augmentation=train_aug)
    patches_augmented = patch_extractor_aug.extract_patches(image_path)
    
    # Visualize first 6 patches
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i in range(min(6, len(patches_original))):
        axes[0, i].imshow(patches_original[i])
        axes[0, i].set_title(f'Original Patch {i}', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(patches_augmented[i])
        axes[1, i].set_title(f'Augmented Patch {i}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Patch Extraction with Augmentation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_patch_augmentation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: test_patch_augmentation.png")
    plt.close()


def print_augmentation_summary():
    """Print summary of augmentation techniques"""
    print("\n" + "="*70)
    print("DATA AUGMENTATION SUMMARY")
    print("="*70)
    
    summary = """
    ✓ HISTOGRAM NORMALIZATION (Applied to Train/Val/Test)
      - CLAHE (Contrast Limited Adaptive Histogram Equalization)
      - Adaptive Histogram Equalization
      - Standard Histogram Equalization
      - Improves contrast and normalizes lighting conditions
    
    ✓ GEOMETRIC TRANSFORMATIONS (Training Only)
      - Rotation: ±15 degrees
      - Zoom: 0.9x to 1.1x
      - Shear: ±10 degrees
      - Horizontal Flip: 50% probability
      - Vertical Flip: 50% probability
      - Increases model robustness to orientation and scale
    
    ✓ COLOR AUGMENTATIONS (Training Only)
      - Brightness: 0.8x to 1.2x
      - Contrast: 0.8x to 1.2x
      - Saturation: 0.8x to 1.2x
      - Hue Shift: ±0.1
      - Handles variations in staining and imaging conditions
    
    ✓ NOISE AND BLUR (Training Only)
      - Gaussian Noise: σ = 0.01
      - Gaussian Blur: σ = 0.1 to 2.0
      - Improves robustness to image quality variations
    
    BENEFITS:
    • Reduces overfitting on training data
    • Improves generalization to new images
    • Handles variations in imaging conditions
    • Increases effective training dataset size
    • Medical imaging specific: normalizes staining variations
    """
    print(summary)
    print("="*70)


def main():
    """Run all augmentation tests"""
    print("\n" + "="*70)
    print("AUGMENTATION PIPELINE TEST SUITE")
    print("="*70)
    
    # Check for sample image
    image_path = 'sample_image.JPG'
    
    if not os.path.exists(image_path):
        print(f"\n⚠ Warning: {image_path} not found!")
        print("Creating a synthetic test image...")
        # Create a synthetic image for testing
        img = Image.new('RGB', (4032, 3024), color=(100, 150, 200))
        img.save(image_path)
        print(f"✓ Created synthetic image: {image_path}")
    
    print(f"\nUsing image: {image_path}")
    print(f"Image size: {Image.open(image_path).size}")
    
    # Run all tests
    try:
        test_histogram_normalization(image_path)
        test_geometric_augmentation(image_path)
        test_color_augmentation(image_path)
        test_noise_and_blur(image_path)
        test_full_pipeline(image_path)
        test_patch_augmentation(image_path)
        
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\nGenerated visualization files:")
        print("  1. test_histogram_normalization.png")
        print("  2. test_geometric_augmentation.png")
        print("  3. test_color_augmentation.png")
        print("  4. test_noise_blur_augmentation.png")
        print("  5. test_full_pipeline.png")
        print("  6. test_patch_augmentation.png")
        
        print_augmentation_summary()
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
