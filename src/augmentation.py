"""
Data Augmentation Module for BMA Classification
Includes histogram normalization, geometric transforms, and color augmentations
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import random
import torchvision.transforms.functional as TF


class HistogramNormalizer:
    """Various histogram normalization techniques for medical images"""
    
    def __init__(self, method='clahe'):
        self.method = method
        
    def __call__(self, image):
        if self.method == 'none':
            return image
            
        img_array = np.array(image)
        
        if self.method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(img_array.shape) == 3:
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                img_array = clahe.apply(img_array)
                
        elif self.method == 'adaptive':
            if len(img_array.shape) == 3:
                for i in range(3):
                    img_array[:, :, i] = cv2.equalizeHist(img_array[:, :, i])
            else:
                img_array = cv2.equalizeHist(img_array)
                
        elif self.method == 'standard':
            if len(img_array.shape) == 3:
                ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                img_array = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                img_array = cv2.equalizeHist(img_array)
        
        return Image.fromarray(img_array)


class GeometricAugmentation:
    """Geometric transformations: rotation, zoom, shear, flip"""
    
    def __init__(self, rotation_range=15, zoom_range=(0.9, 1.1), shear_range=10,
                 horizontal_flip=True, vertical_flip=True, probability=0.5):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.probability = probability
        
    def __call__(self, image):
        if random.random() < self.probability:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        if random.random() < self.probability:
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            w, h = image.size
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            image = TF.resize(image, (new_h, new_w))
            
            if zoom_factor > 1.0:
                image = TF.center_crop(image, (h, w))
            else:
                padding = ((w - new_w) // 2, (h - new_h) // 2)
                image = TF.pad(image, padding, fill=0)
        
        if random.random() < self.probability:
            shear_angle = random.uniform(-self.shear_range, self.shear_range)
            image = TF.affine(image, angle=0, translate=[0, 0], 
                            scale=1.0, shear=[shear_angle, 0],
                            interpolation=TF.InterpolationMode.BILINEAR)
        
        if self.horizontal_flip and random.random() < self.probability:
            image = TF.hflip(image)
        
        if self.vertical_flip and random.random() < self.probability:
            image = TF.vflip(image)
        
        return image


class ColorAugmentation:
    """Color space augmentations"""
    
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1), probability=0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.probability = probability
        
    def __call__(self, image):
        if random.random() < self.probability:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        if random.random() < self.probability:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        if random.random() < self.probability:
            factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        
        if random.random() < self.probability:
            hue_factor = random.uniform(self.hue_range[0], self.hue_range[1])
            img_array = np.array(image)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180
            img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            image = Image.fromarray(img_array)
        
        return image


class NoiseAndBlurAugmentation:
    """Add noise and blur for robustness"""
    
    def __init__(self, gaussian_noise_std=0.01, gaussian_blur_sigma=(0.1, 2.0), probability=0.3):
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.probability = probability
        
    def __call__(self, image):
        if random.random() < self.probability:
            img_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.gaussian_noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        if random.random() < self.probability:
            sigma = random.uniform(self.gaussian_blur_sigma[0], self.gaussian_blur_sigma[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return image


class ComposedAugmentation:
    """Compose all augmentation techniques"""
    
    def __init__(self, histogram_method='clahe', enable_geometric=True, enable_color=True,
                 enable_noise=True, is_training=True, target_size=224, **kwargs):
        self.is_training = is_training
        self.target_size = target_size
        self.transforms = []
        
        self.transforms.append(HistogramNormalizer(method=histogram_method))
        
        if is_training:
            if enable_geometric:
                self.transforms.append(GeometricAugmentation(
                    rotation_range=kwargs.get('rotation_range', 15),
                    zoom_range=kwargs.get('zoom_range', (0.9, 1.1)),
                    shear_range=kwargs.get('shear_range', 10),
                    horizontal_flip=kwargs.get('horizontal_flip', True),
                    vertical_flip=kwargs.get('vertical_flip', True),
                    probability=kwargs.get('geometric_prob', 0.5)
                ))
            
            if enable_color:
                self.transforms.append(ColorAugmentation(
                    brightness_range=kwargs.get('brightness_range', (0.8, 1.2)),
                    contrast_range=kwargs.get('contrast_range', (0.8, 1.2)),
                    saturation_range=kwargs.get('saturation_range', (0.8, 1.2)),
                    hue_range=kwargs.get('hue_range', (-0.1, 0.1)),
                    probability=kwargs.get('color_prob', 0.5)
                ))
            
            if enable_noise:
                self.transforms.append(NoiseAndBlurAugmentation(
                    gaussian_noise_std=kwargs.get('noise_std', 0.01),
                    gaussian_blur_sigma=kwargs.get('blur_sigma', (0.1, 2.0)),
                    probability=kwargs.get('noise_prob', 0.3)
                ))
    
    def __call__(self, image):
        # Store original size
        original_size = image.size
        
        # Apply all transformations
        for transform in self.transforms:
            image = transform(image)
        
        # Ensure consistent size after augmentation (force resize if needed)
        if image.size != original_size:
            image = image.resize(original_size, Image.LANCZOS)
        
        return image


def get_augmentation_pipeline(is_training=True, config=None, target_size=224):
    """Factory function to create augmentation pipeline"""
    if config is None:
        return ComposedAugmentation(
            histogram_method='clahe',
            enable_geometric=True,
            enable_color=True,
            enable_noise=True,
            is_training=is_training,
            target_size=target_size
        )
    else:
        return ComposedAugmentation(
            histogram_method=config.HISTOGRAM_METHOD,
            enable_geometric=config.ENABLE_GEOMETRIC_AUG,
            enable_color=config.ENABLE_COLOR_AUG,
            enable_noise=config.ENABLE_NOISE_AUG,
            is_training=is_training,
            target_size=target_size,
            rotation_range=config.ROTATION_RANGE,
            zoom_range=config.ZOOM_RANGE,
            shear_range=config.SHEAR_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            vertical_flip=config.VERTICAL_FLIP,
            geometric_prob=config.GEOMETRIC_PROB,
            brightness_range=config.BRIGHTNESS_RANGE,
            contrast_range=config.CONTRAST_RANGE,
            saturation_range=config.SATURATION_RANGE,
            hue_range=config.HUE_RANGE,
            color_prob=config.COLOR_PROB,
            noise_std=config.NOISE_STD,
            blur_sigma=config.BLUR_SIGMA,
            noise_prob=config.NOISE_PROB
        )
