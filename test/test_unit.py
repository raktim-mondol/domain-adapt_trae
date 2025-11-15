"""
Unit Tests for Augmentation Module
Tests individual components in isolation
"""

import unittest
import numpy as np
from PIL import Image
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmentation import (
    HistogramNormalizer,
    GeometricAugmentation,
    ColorAugmentation,
    NoiseAndBlurAugmentation,
    ComposedAugmentation,
    get_augmentation_pipeline
)


class TestHistogramNormalizer(unittest.TestCase):
    """Test HistogramNormalizer class"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_none_method(self):
        """Test 'none' method returns unchanged image"""
        normalizer = HistogramNormalizer(method='none')
        result = normalizer(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_clahe_method(self):
        """Test CLAHE histogram normalization"""
        normalizer = HistogramNormalizer(method='clahe')
        result = normalizer(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        self.assertEqual(result.mode, 'RGB')
        
    def test_adaptive_method(self):
        """Test adaptive histogram equalization"""
        normalizer = HistogramNormalizer(method='adaptive')
        result = normalizer(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_standard_method(self):
        """Test standard histogram equalization"""
        normalizer = HistogramNormalizer(method='standard')
        result = normalizer(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_invalid_method(self):
        """Test with invalid method (should default to none)"""
        normalizer = HistogramNormalizer(method='invalid')
        result = normalizer(self.test_img)
        self.assertIsInstance(result, Image.Image)


class TestGeometricAugmentation(unittest.TestCase):
    """Test GeometricAugmentation class"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_rotation(self):
        """Test rotation augmentation"""
        aug = GeometricAugmentation(
            rotation_range=30,
            zoom_range=(1.0, 1.0),
            shear_range=0,
            horizontal_flip=False,
            vertical_flip=False,
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_zoom(self):
        """Test zoom augmentation"""
        aug = GeometricAugmentation(
            rotation_range=0,
            zoom_range=(0.8, 1.2),
            shear_range=0,
            horizontal_flip=False,
            vertical_flip=False,
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_shear(self):
        """Test shear augmentation"""
        aug = GeometricAugmentation(
            rotation_range=0,
            zoom_range=(1.0, 1.0),
            shear_range=15,
            horizontal_flip=False,
            vertical_flip=False,
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_horizontal_flip(self):
        """Test horizontal flip"""
        aug = GeometricAugmentation(
            rotation_range=0,
            zoom_range=(1.0, 1.0),
            shear_range=0,
            horizontal_flip=True,
            vertical_flip=False,
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_vertical_flip(self):
        """Test vertical flip"""
        aug = GeometricAugmentation(
            rotation_range=0,
            zoom_range=(1.0, 1.0),
            shear_range=0,
            horizontal_flip=False,
            vertical_flip=True,
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_zero_probability(self):
        """Test with zero probability (no augmentation)"""
        aug = GeometricAugmentation(probability=0.0)
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)


class TestColorAugmentation(unittest.TestCase):
    """Test ColorAugmentation class"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_brightness(self):
        """Test brightness augmentation"""
        aug = ColorAugmentation(
            brightness_range=(0.8, 1.2),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_range=(0.0, 0.0),
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_contrast(self):
        """Test contrast augmentation"""
        aug = ColorAugmentation(
            brightness_range=(1.0, 1.0),
            contrast_range=(0.8, 1.2),
            saturation_range=(1.0, 1.0),
            hue_range=(0.0, 0.0),
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_saturation(self):
        """Test saturation augmentation"""
        aug = ColorAugmentation(
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(0.8, 1.2),
            hue_range=(0.0, 0.0),
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_hue(self):
        """Test hue augmentation"""
        aug = ColorAugmentation(
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            saturation_range=(1.0, 1.0),
            hue_range=(-0.1, 0.1),
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_zero_probability(self):
        """Test with zero probability (no augmentation)"""
        aug = ColorAugmentation(probability=0.0)
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)


class TestNoiseAndBlurAugmentation(unittest.TestCase):
    """Test NoiseAndBlurAugmentation class"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_gaussian_noise(self):
        """Test Gaussian noise augmentation"""
        aug = NoiseAndBlurAugmentation(
            gaussian_noise_std=0.01,
            gaussian_blur_sigma=(0.1, 0.1),
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_gaussian_blur(self):
        """Test Gaussian blur augmentation"""
        aug = NoiseAndBlurAugmentation(
            gaussian_noise_std=0.0,
            gaussian_blur_sigma=(1.0, 2.0),
            probability=1.0
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_zero_probability(self):
        """Test with zero probability (no augmentation)"""
        aug = NoiseAndBlurAugmentation(probability=0.0)
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)


class TestComposedAugmentation(unittest.TestCase):
    """Test ComposedAugmentation class"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_training_pipeline(self):
        """Test training augmentation pipeline"""
        aug = ComposedAugmentation(
            histogram_method='clahe',
            enable_geometric=True,
            enable_color=True,
            enable_noise=True,
            is_training=True
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_validation_pipeline(self):
        """Test validation augmentation pipeline (histogram only)"""
        aug = ComposedAugmentation(
            histogram_method='clahe',
            is_training=False
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_disabled_augmentations(self):
        """Test with all augmentations disabled"""
        aug = ComposedAugmentation(
            histogram_method='none',
            enable_geometric=False,
            enable_color=False,
            enable_noise=False,
            is_training=True
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_custom_parameters(self):
        """Test with custom parameters"""
        aug = ComposedAugmentation(
            histogram_method='clahe',
            enable_geometric=True,
            enable_color=False,
            enable_noise=False,
            is_training=True,
            rotation_range=30,
            zoom_range=(0.8, 1.3),
            shear_range=15,
            geometric_prob=0.7
        )
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)


class TestFactoryFunction(unittest.TestCase):
    """Test get_augmentation_pipeline factory function"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_training_pipeline_default(self):
        """Test training pipeline with default config"""
        aug = get_augmentation_pipeline(is_training=True)
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_validation_pipeline_default(self):
        """Test validation pipeline with default config"""
        aug = get_augmentation_pipeline(is_training=False)
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)
        
    def test_with_config(self):
        """Test with Config object"""
        from config import Config
        aug = get_augmentation_pipeline(is_training=True, config=Config)
        result = aug(self.test_img)
        self.assertIsInstance(result, Image.Image)


class TestImageProperties(unittest.TestCase):
    """Test that augmentations preserve image properties"""
    
    def setUp(self):
        """Create test image"""
        self.test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    def test_size_preservation(self):
        """Test that image size is preserved"""
        aug = get_augmentation_pipeline(is_training=True)
        result = aug(self.test_img)
        self.assertEqual(result.size, self.test_img.size)
        
    def test_mode_preservation(self):
        """Test that RGB mode is preserved"""
        aug = get_augmentation_pipeline(is_training=True)
        result = aug(self.test_img)
        self.assertEqual(result.mode, 'RGB')
        
    def test_pixel_range(self):
        """Test that pixel values are in valid range"""
        aug = get_augmentation_pipeline(is_training=True)
        result = aug(self.test_img)
        arr = np.array(result)
        self.assertTrue(np.all(arr >= 0))
        self.assertTrue(np.all(arr <= 255))


def run_unit_tests():
    """Run all unit tests"""
    print("="*70)
    print("RUNNING UNIT TESTS")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHistogramNormalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestGeometricAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestColorAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseAndBlurAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestComposedAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestFactoryFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestImageProperties))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("UNIT TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL UNIT TESTS PASSED!")
        return True
    else:
        print("\n✗ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)
