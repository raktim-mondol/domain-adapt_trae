"""
End-to-End Tests for BMA Classification with Augmentation
Tests the complete pipeline from image loading to model inference
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentation import get_augmentation_pipeline
from bma_mil_classifier import (
    PatchExtractor,
    FeatureExtractor,
    ImageLevelAggregator,
    PileLevelAggregator,
    BMA_MIL_Classifier,
    BMADataset
)
from config import Config


def test_patch_extraction_with_augmentation():
    """Test patch extraction with augmentation"""
    print("\n" + "="*70)
    print("TEST 1: Patch Extraction with Augmentation")
    print("="*70)
    
    # Use sample_image.JPG if available, otherwise create test image
    if os.path.exists('sample_image.JPG'):
        test_img_path = 'sample_image.JPG'
        cleanup_image = False
        print("\n   Using sample_image.JPG for testing")
    else:
        test_img_path = 'temp_test_full_image.jpg'
        img = Image.new('RGB', (4032, 3024), color=(100, 150, 200))
        img.save(test_img_path)
        cleanup_image = True
        print("\n   Created temporary test image")
    
    try:
        # Test without augmentation
        print("\n1.1 Testing patch extraction WITHOUT augmentation...")
        patch_extractor_no_aug = PatchExtractor(augmentation=None)
        patches_no_aug = patch_extractor_no_aug.extract_patches(test_img_path)
        
        assert len(patches_no_aug) == 12, f"Expected 12 patches, got {len(patches_no_aug)}"
        assert all(p.size == (224, 224) for p in patches_no_aug), "Patches should be 224x224"
        print(f"   ✓ Extracted {len(patches_no_aug)} patches of size 224x224")
        
        # Test with training augmentation
        print("\n1.2 Testing patch extraction WITH training augmentation...")
        train_aug = get_augmentation_pipeline(is_training=True, config=Config)
        patch_extractor_aug = PatchExtractor(augmentation=train_aug)
        patches_aug = patch_extractor_aug.extract_patches(test_img_path)
        
        assert len(patches_aug) == 12, f"Expected 12 patches, got {len(patches_aug)}"
        # Check sizes and report any mismatches
        sizes = [p.size for p in patches_aug]
        if not all(s == (224, 224) for s in sizes):
            print(f"   ⚠ Warning: Some patches have unexpected sizes: {set(sizes)}")
            print(f"   Note: Augmentation may slightly alter size, checking if close to 224x224...")
            # Allow small variations due to augmentation
            assert all(abs(s[0] - 224) <= 10 and abs(s[1] - 224) <= 10 for s in sizes), \
                f"Patches too far from 224x224: {sizes}"
        print(f"   ✓ Extracted {len(patches_aug)} augmented patches (sizes: {set(sizes)})")
        
        # Test with validation augmentation
        print("\n1.3 Testing patch extraction WITH validation augmentation...")
        val_aug = get_augmentation_pipeline(is_training=False, config=Config)
        patch_extractor_val = PatchExtractor(augmentation=val_aug)
        patches_val = patch_extractor_val.extract_patches(test_img_path)
        
        assert len(patches_val) == 12, f"Expected 12 patches, got {len(patches_val)}"
        print(f"   ✓ Extracted {len(patches_val)} validation patches")
        
        print("\n✓ TEST 1 PASSED: Patch extraction with augmentation works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if cleanup_image and os.path.exists(test_img_path):
            os.remove(test_img_path)


def test_feature_extraction_pipeline():
    """Test feature extraction with augmented patches"""
    print("\n" + "="*70)
    print("TEST 2: Feature Extraction Pipeline")
    print("="*70)
    
    # Use sample_image.JPG if available, otherwise create test image
    if os.path.exists('sample_image.JPG'):
        test_img_path = 'sample_image.JPG'
        cleanup_image = False
        print("\n   Using sample_image.JPG for testing")
    else:
        test_img_path = 'temp_test_feature_image.jpg'
        img = Image.new('RGB', (4032, 3024), color=(100, 150, 200))
        img.save(test_img_path)
        cleanup_image = True
        print("\n   Created temporary test image")
    
    try:
        print("\n2.1 Initializing feature extractor...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        feature_extractor = FeatureExtractor(device=device)
        print("   ✓ Feature extractor initialized")
        
        print("\n2.2 Extracting patches with augmentation...")
        train_aug = get_augmentation_pipeline(is_training=True, config=Config)
        patch_extractor = PatchExtractor(augmentation=train_aug)
        patches = patch_extractor.extract_patches(test_img_path)
        print(f"   ✓ Extracted {len(patches)} patches")
        
        print("\n2.3 Extracting features from patches...")
        features = feature_extractor.extract_features(patches)
        
        assert features.shape[0] == 12, f"Expected 12 feature vectors, got {features.shape[0]}"
        assert features.shape[1] == 768, f"Expected 768-dim features, got {features.shape[1]}"
        print(f"   ✓ Extracted features with shape: {features.shape}")
        print(f"   ✓ Feature dimension: {features.shape[1]} (expected 768)")
        
        print("\n✓ TEST 2 PASSED: Feature extraction pipeline works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if cleanup_image and os.path.exists(test_img_path):
            os.remove(test_img_path)


def test_image_level_aggregation():
    """Test image-level aggregation with augmented features"""
    print("\n" + "="*70)
    print("TEST 3: Image-Level Aggregation")
    print("="*70)
    
    try:
        print("\n3.1 Creating dummy patch features...")
        batch_size = 2
        num_patches = 12
        feature_dim = 768
        
        # Create dummy patch features
        patch_features = torch.randn(batch_size, num_patches, feature_dim)
        print(f"   ✓ Created patch features: {patch_features.shape}")
        
        print("\n3.2 Initializing image-level aggregator...")
        image_aggregator = ImageLevelAggregator(
            input_dim=feature_dim,
            hidden_dim=512
        )
        print("   ✓ Image aggregator initialized")
        
        print("\n3.3 Aggregating patches to image level...")
        image_features = image_aggregator(patch_features)
        
        assert image_features.shape[0] == batch_size, f"Expected batch size {batch_size}"
        assert image_features.shape[1] == 512, f"Expected 512-dim image features"
        print(f"   ✓ Image features shape: {image_features.shape}")
        
        print("\n✓ TEST 3 PASSED: Image-level aggregation works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pile_level_aggregation():
    """Test pile-level aggregation"""
    print("\n" + "="*70)
    print("TEST 4: Pile-Level Aggregation")
    print("="*70)
    
    try:
        print("\n4.1 Creating dummy image features...")
        batch_size = 2
        num_images = 5
        image_dim = 512
        
        # Create dummy image features
        image_features = torch.randn(batch_size, num_images, image_dim)
        print(f"   ✓ Created image features: {image_features.shape}")
        
        print("\n4.2 Initializing pile-level aggregator...")
        pile_aggregator = PileLevelAggregator(
            input_dim=image_dim,
            hidden_dim=256,
            num_classes=4
        )
        print("   ✓ Pile aggregator initialized")
        
        print("\n4.3 Aggregating images to pile level...")
        pile_logits, attention_weights = pile_aggregator(image_features)
        
        assert pile_logits.shape[0] == batch_size, f"Expected batch size {batch_size}"
        assert pile_logits.shape[1] == 4, f"Expected 4 classes"
        assert attention_weights.shape == (batch_size, num_images), "Attention weights shape mismatch"
        print(f"   ✓ Pile logits shape: {pile_logits.shape}")
        print(f"   ✓ Attention weights shape: {attention_weights.shape}")
        
        # Check attention weights sum to 1
        attention_sum = attention_weights.sum(dim=1)
        assert torch.allclose(attention_sum, torch.ones(batch_size), atol=1e-5), "Attention weights should sum to 1"
        print(f"   ✓ Attention weights sum to 1.0")
        
        print("\n✓ TEST 4 PASSED: Pile-level aggregation works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_model_forward():
    """Test complete model forward pass"""
    print("\n" + "="*70)
    print("TEST 5: Complete Model Forward Pass")
    print("="*70)
    
    try:
        print("\n5.1 Initializing complete BMA MIL model...")
        model = BMA_MIL_Classifier(
            feature_dim=768,
            image_hidden_dim=512,
            pile_hidden_dim=256,
            num_classes=4
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"   ✓ Model initialized on device: {device}")
        print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n5.2 Creating dummy input (2 piles with variable images)...")
        # Pile 1: 3 images, each with 12 patches
        pile1_features = torch.randn(3, 12, 768).to(device)
        # Pile 2: 5 images, each with 12 patches
        pile2_features = torch.randn(5, 12, 768).to(device)
        
        patch_features_list = [pile1_features, pile2_features]
        print(f"   ✓ Pile 1: {pile1_features.shape[0]} images")
        print(f"   ✓ Pile 2: {pile2_features.shape[0]} images")
        
        print("\n5.3 Running forward pass...")
        model.eval()
        with torch.no_grad():
            pile_logits, attention_weights = model(patch_features_list)
        
        assert pile_logits.shape == (2, 4), f"Expected (2, 4) logits, got {pile_logits.shape}"
        print(f"   ✓ Output logits shape: {pile_logits.shape}")
        print(f"   ✓ Attention weights shape: {attention_weights.shape}")
        
        # Check predictions
        predictions = torch.argmax(pile_logits, dim=1)
        print(f"   ✓ Predictions: {predictions.cpu().numpy()}")
        
        print("\n✓ TEST 5 PASSED: Complete model forward pass works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_with_augmentation():
    """Test BMADataset with augmentation"""
    print("\n" + "="*70)
    print("TEST 6: BMADataset with Augmentation")
    print("="*70)
    
    # Create dummy CSV data
    csv_path = 'temp_test_data.csv'
    image_dir = '.'
    
    try:
        print("\n6.1 Creating dummy dataset...")
        # Create dummy images
        img_paths = []
        for i in range(4):
            img_path = f'temp_test_img_{i}.jpg'
            img = Image.new('RGB', (4032, 3024), color=(100 + i*20, 150, 200))
            img.save(img_path)
            img_paths.append(img_path)
        
        # Create dummy CSV
        df = pd.DataFrame({
            'pile': ['pile1', 'pile1', 'pile2', 'pile2'],
            'image_path': img_paths,
            'BMA_label': [1, 1, 2, 2]
        })
        df.to_csv(csv_path, index=False)
        print(f"   ✓ Created dummy dataset with {len(df)} images, 2 piles")
        
        print("\n6.2 Initializing feature extractor...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        feature_extractor = FeatureExtractor(device=device)
        print(f"   ✓ Feature extractor initialized on {device}")
        
        print("\n6.3 Creating training dataset with augmentation...")
        train_aug = get_augmentation_pipeline(is_training=True, config=Config)
        train_dataset = BMADataset(
            df, image_dir, feature_extractor,
            augmentation=train_aug,
            is_training=True
        )
        print(f"   ✓ Training dataset created with {len(train_dataset)} piles")
        
        print("\n6.4 Creating validation dataset with augmentation...")
        val_aug = get_augmentation_pipeline(is_training=False, config=Config)
        val_dataset = BMADataset(
            df, image_dir, feature_extractor,
            augmentation=val_aug,
            is_training=False
        )
        print(f"   ✓ Validation dataset created with {len(val_dataset)} piles")
        
        print("\n6.5 Testing dataset __getitem__...")
        patch_features, label, pile_name = train_dataset[0]
        
        print(f"   ✓ Patch features shape: {patch_features.shape}")
        print(f"   ✓ Label: {label}")
        print(f"   ✓ Pile name: {pile_name}")
        
        assert patch_features.ndim == 3, "Expected 3D tensor (num_images, num_patches, feature_dim)"
        assert label in [0, 1, 2, 3], f"Label should be 0-3, got {label}"
        
        print("\n✓ TEST 6 PASSED: BMADataset with augmentation works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
        for img_path in img_paths:
            if os.path.exists(img_path):
                os.remove(img_path)


def test_augmentation_configuration():
    """Test that augmentation respects configuration"""
    print("\n" + "="*70)
    print("TEST 7: Augmentation Configuration")
    print("="*70)
    
    try:
        print("\n7.1 Checking current configuration...")
        print(f"   • Histogram Method: {Config.HISTOGRAM_METHOD}")
        print(f"   • Geometric Augmentation: {Config.ENABLE_GEOMETRIC_AUG}")
        print(f"   • Color Augmentation: {Config.ENABLE_COLOR_AUG}")
        print(f"   • Noise Augmentation: {Config.ENABLE_NOISE_AUG}")
        
        print("\n7.2 Testing training pipeline respects config...")
        train_aug = get_augmentation_pipeline(is_training=True, config=Config)
        
        # Count enabled transforms
        num_transforms = len(train_aug.transforms)
        print(f"   ✓ Number of transforms in training pipeline: {num_transforms}")
        
        # Should have at least histogram normalizer
        assert num_transforms >= 1, "Should have at least histogram normalizer"
        
        print("\n7.3 Testing validation pipeline (histogram only)...")
        val_aug = get_augmentation_pipeline(is_training=False, config=Config)
        
        # Validation should only have histogram normalizer
        assert len(val_aug.transforms) == 1, "Validation should only have histogram normalizer"
        print(f"   ✓ Validation pipeline has {len(val_aug.transforms)} transform (histogram only)")
        
        print("\n✓ TEST 7 PASSED: Augmentation configuration works correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_end_to_end_tests():
    """Run all end-to-end tests"""
    print("\n" + "="*70)
    print("RUNNING END-TO-END TESTS")
    print("="*70)
    print("\nThese tests verify the complete pipeline from image to prediction")
    print("with augmentation integrated at each step.\n")
    
    tests = [
        ("Patch Extraction with Augmentation", test_patch_extraction_with_augmentation),
        ("Feature Extraction Pipeline", test_feature_extraction_pipeline),
        ("Image-Level Aggregation", test_image_level_aggregation),
        ("Pile-Level Aggregation", test_pile_level_aggregation),
        ("Complete Model Forward Pass", test_complete_model_forward),
        ("BMADataset with Augmentation", test_dataset_with_augmentation),
        ("Augmentation Configuration", test_augmentation_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ CRITICAL ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("END-TO-END TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL END-TO-END TESTS PASSED!")
        print("="*70)
        print("\nThe complete pipeline is working correctly:")
        print("  • Patch extraction with augmentation ✓")
        print("  • Feature extraction ✓")
        print("  • Multi-level aggregation ✓")
        print("  • Complete model inference ✓")
        print("  • Dataset integration ✓")
        print("  • Configuration management ✓")
        print("\n✓ Ready for training on real data!")
        return True
    else:
        print("\n" + "="*70)
        print("✗ SOME END-TO-END TESTS FAILED!")
        print("="*70)
        print(f"\nPlease fix the {total - passed} failed test(s) before training.")
        return False


if __name__ == "__main__":
    success = run_end_to_end_tests()
    sys.exit(0 if success else 1)
