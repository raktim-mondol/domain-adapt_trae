"""
BMA MIL Classification Package
Multi-Level Multiple Instance Learning for BMA Classification
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .models import BMA_MIL_Classifier
from .data import BMADataset, PatchExtractor
from .augmentation import get_augmentation_pipeline
from .feature_extractor import FeatureExtractor

__all__ = [
    'BMA_MIL_Classifier',
    'BMADataset',
    'PatchExtractor',
    'FeatureExtractor',
    'get_augmentation_pipeline'
]
