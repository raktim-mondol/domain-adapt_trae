"""
Main training script for BMA MIL Classifier
Implements Option 1: Train on bags, evaluate on piles
Supports both standard split (70/10/20) and k-fold cross-validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

from src.models import BMA_MIL_Classifier
from src.data.dataset import create_bag_dataset_from_piles
from src.data.pile_dataset import create_pile_dataset_from_piles, collate_pile_batch
from src.feature_extractor import FeatureExtractor
from src.augmentation import get_augmentation_pipeline
from src.utils import (
    train_model,
    train_model_da,
    train_model_uda,
    train_model_ssda,
    train_model_pile_level,
    compute_class_weights,
    evaluate_model,
    setup_logging,
    save_results_to_file
)
from configs.config import Config


def split_piles_standard(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """Split piles into train/val/test sets with stratification"""
    # Get unique piles and their labels
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    unique_piles = pile_labels['pile'].values
    pile_bma_labels = pile_labels['BMA_label'].values - 1  # 0-indexed
    
    # Check class distribution
    class_counts = pile_labels['BMA_label'].value_counts().sort_index()
    print(f"\nPile-level class distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} piles")
    
    # Check if stratification is possible
    min_class_count = min(class_counts.values)
    n_splits = 3  # train, val, test
    
    print(f"\nSplitting {len(unique_piles)} piles:")
    print(f"  Train: {train_ratio*100:.1f}% target")
    print(f"  Val:   {val_ratio*100:.1f}% target")
    print(f"  Test:  {test_ratio*100:.1f}% target")
    
    # For small datasets, use manual stratified splitting
    if min_class_count < n_splits * 2:
        print(f"\n[WARNING] Smallest class has only {min_class_count} piles.")
        print("Using manual stratified split to ensure all classes in all splits...")
        
        # Manual stratified split - ensure each class is represented in all splits
        train_piles_list = []
        val_piles_list = []
        test_piles_list = []
        
        np.random.seed(random_state)
        
        for cls in sorted(class_counts.index):
            cls_piles = pile_labels[pile_labels['BMA_label'] == cls]['pile'].values
            cls_piles_shuffled = np.random.permutation(cls_piles)
            
            n_cls = len(cls_piles_shuffled)
            
            # Calculate split sizes for this class
            if n_cls >= 3:
                # Try to maintain ratios, but ensure at least 1 in each split
                n_cls_test = max(1, int(n_cls * test_ratio))
                n_cls_val = max(1, int(n_cls * val_ratio))
                n_cls_train = n_cls - n_cls_test - n_cls_val
                
                # If train becomes 0, adjust
                if n_cls_train < 1:
                    n_cls_train = 1
                    n_cls_val = max(1, n_cls - n_cls_train - n_cls_test)
                    n_cls_test = n_cls - n_cls_train - n_cls_val
            else:
                # Very small class - distribute as best as possible
                if n_cls == 1:
                    n_cls_train, n_cls_val, n_cls_test = 1, 0, 0
                elif n_cls == 2:
                    n_cls_train, n_cls_val, n_cls_test = 1, 1, 0
                else:
                    print(f"  [WARNING] Class {cls} has {n_cls} piles - minimal split")
            
            # Split this class's piles
            train_piles_list.extend(cls_piles_shuffled[:n_cls_train])
            val_piles_list.extend(cls_piles_shuffled[n_cls_train:n_cls_train+n_cls_val])
            test_piles_list.extend(cls_piles_shuffled[n_cls_train+n_cls_val:])
            
            print(f"  Class {cls}: {n_cls_train} train, {n_cls_val} val, {n_cls - n_cls_train - n_cls_val} test")
        
        train_piles = np.array(train_piles_list)
        val_piles = np.array(val_piles_list)
        test_piles = np.array(test_piles_list)
        
        print("\n[SUCCESS] Manual stratified split completed")
        
    else:
        # Standard stratified split for larger datasets
        try:
            # First split: train vs (val+test)
            train_piles, temp_piles, train_labels, temp_labels = train_test_split(
                unique_piles, pile_bma_labels, 
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=pile_bma_labels
            )
            
            # Second split: val vs test
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_piles, test_piles, val_labels, test_labels = train_test_split(
                temp_piles, temp_labels, 
                test_size=(1 - val_ratio_adjusted),
                random_state=random_state,
                stratify=temp_labels
            )

            print("\n[SUCCESS] Stratified split successful")
            
        except ValueError as e:
            print(f"\n[WARNING] Stratified split failed: {e}")
            print("Using random split instead...")
            
            # Fallback to random split
            train_piles, temp_piles = train_test_split(
                unique_piles,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )
            
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_piles, test_piles = train_test_split(
                temp_piles,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_state
            )
    
    # Verify no overlap
    assert len(set(train_piles) & set(val_piles)) == 0, "Train/Val overlap!"
    assert len(set(train_piles) & set(test_piles)) == 0, "Train/Test overlap!"
    assert len(set(val_piles) & set(test_piles)) == 0, "Val/Test overlap!"
    
    print(f"\n[SUCCESS] No pile overlap between splits")
    
    # Verify class distribution in splits
    train_labels_check = pile_labels[pile_labels['pile'].isin(train_piles)]['BMA_label']
    val_labels_check = pile_labels[pile_labels['pile'].isin(val_piles)]['BMA_label']
    test_labels_check = pile_labels[pile_labels['pile'].isin(test_piles)]['BMA_label']
    
    train_classes = set(train_labels_check.unique())
    val_classes = set(val_labels_check.unique())
    test_classes = set(test_labels_check.unique())
    all_classes = set(pile_labels['BMA_label'].unique())
    
    print(f"\nClass distribution verification:")
    print(f"  Train set has classes: {sorted(train_classes)}")
    print(f"  Val set has classes:   {sorted(val_classes)}")
    print(f"  Test set has classes:  {sorted(test_classes)}")
    
    if train_classes != all_classes:
        print(f"  [WARNING] Train set missing classes: {sorted(all_classes - train_classes)}")
    if val_classes != all_classes and len(val_piles) > 0:
        print(f"  [WARNING] Val set missing classes: {sorted(all_classes - val_classes)}")
    if test_classes != all_classes and len(test_piles) > 0:
        print(f"  [WARNING] Test set missing classes: {sorted(all_classes - test_classes)}")
    
    if train_classes == val_classes == test_classes == all_classes:
        print(f"  [SUCCESS] All splits contain all {len(all_classes)} classes!")
    
    return list(train_piles), list(val_piles), list(test_piles)


def split_piles_kfold(df, n_folds=5, random_state=42):
    """Split piles for k-fold cross-validation with handling for small classes"""
    # Get unique piles and their labels
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    unique_piles = pile_labels['pile'].values
    pile_bma_labels = pile_labels['BMA_label'].values - 1  # 0-indexed
    
    # Show overall class distribution
    class_counts = pile_labels['BMA_label'].value_counts().sort_index()
    print(f"\nPile-level class distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} piles")
    
    print(f"\nSetting up {n_folds}-fold cross-validation")
    print(f"Total piles: {len(unique_piles)}")
    
    # Check if any class has fewer samples than n_folds
    min_class_count = min(class_counts.values)
    if min_class_count < n_folds:
        print(f"\n[WARNING] Smallest class has only {min_class_count} piles, less than {n_folds} folds.")
        print("Using custom stratified split with cyclic distribution for small classes...")
        
        # Manual stratified k-fold split
        np.random.seed(random_state)
        fold_splits = [{'train': [], 'val': []} for _ in range(n_folds)]
        
        for cls in sorted(class_counts.index):
            cls_piles = pile_labels[pile_labels['BMA_label'] == cls]['pile'].values
            cls_piles_shuffled = np.random.permutation(cls_piles)
            n_cls = len(cls_piles_shuffled)
            
            if n_cls >= n_folds:
                # Standard stratified split for this class
                fold_size = n_cls // n_folds
                remainder = n_cls % n_folds
                
                start_idx = 0
                for fold in range(n_folds):
                    # Add 1 extra pile to first 'remainder' folds to distribute evenly
                    end_idx = start_idx + fold_size + (1 if fold < remainder else 0)
                    fold_splits[fold]['val'].extend(cls_piles_shuffled[start_idx:end_idx])
                    start_idx = end_idx
            else:
                # For classes with fewer piles than folds, use cyclic distribution
                # Each pile will appear in validation of different folds in a round-robin fashion
                # This ensures all folds get at least 1 pile from this class in validation
                print(f"  [WARNING] Class {cls}: {n_cls} piles < {n_folds} folds")
                print(f"             Using cyclic distribution - some piles will appear in multiple validation sets")
                
                # Distribute piles cyclically across all folds
                for fold in range(n_folds):
                    # Use modulo to cycle through available piles
                    pile_idx = fold % n_cls
                    fold_splits[fold]['val'].append(cls_piles_shuffled[pile_idx])
        
        # Now assign training sets (all piles not in validation for that fold)
        for fold in range(n_folds):
            val_set = set(fold_splits[fold]['val'])
            train_set = [p for p in unique_piles if p not in val_set]
            fold_splits[fold]['train'] = train_set
            
            # Check if train set has all classes
            train_labels_check = pile_labels[pile_labels['pile'].isin(train_set)]
            train_classes = set(train_labels_check['BMA_label'].unique())
            missing_train_classes = set(class_counts.index) - train_classes
            
            if missing_train_classes:
                # This can happen when a small class has all its piles in validation
                # Solution: Keep at least one pile from each small class in training
                print(f"  [WARNING] Fold {fold+1}: Training set missing classes {sorted(missing_train_classes)}")
                print(f"             Adding one pile from each missing class back to training")
                
                for missing_cls in missing_train_classes:
                    # Find a pile from this class that's currently in validation
                    cls_piles_in_val = [p for p in val_set if pile_labels[pile_labels['pile'] == p]['BMA_label'].values[0] == missing_cls]
                    if cls_piles_in_val:
                        # Move one pile from val to train (but keep it in val too for testing)
                        # Actually, we need a different approach - keep all in validation but also add to train
                        # NO - we should not have overlap. Instead, for very small classes:
                        # Put different piles in val for each fold, and rest in training
                        pass
                
                # Re-calculate: for small classes, ensure proper split
                # Let's use a different strategy: put one pile in val, rest in train
                val_set_new = set()
                for cls in sorted(class_counts.index):
                    cls_piles_all = pile_labels[pile_labels['BMA_label'] == cls]['pile'].values
                    n_cls = len(cls_piles_all)
                    
                    if n_cls < n_folds:
                        # For this fold, pick which pile goes to validation
                        pile_idx = fold % n_cls
                        val_set_new.add(cls_piles_all[pile_idx])
                    else:
                        # Use the already assigned validation piles
                        cls_val_piles = [p for p in val_set if pile_labels[pile_labels['pile'] == p]['BMA_label'].values[0] == cls]
                        val_set_new.update(cls_val_piles)
                
                fold_splits[fold]['val'] = list(val_set_new)
                train_set = [p for p in unique_piles if p not in val_set_new]
                fold_splits[fold]['train'] = train_set
                val_set = val_set_new
            
            # Verify no overlap
            assert len(val_set & set(train_set)) == 0, f"Overlap in fold {fold}!"
        
        # Convert to list of tuples format
        result_splits = [(fold['train'], fold['val']) for fold in fold_splits]
        
        # Print fold information with class distribution
        for fold_idx, (train_piles, val_piles) in enumerate(result_splits):
            train_labels = pile_labels[pile_labels['pile'].isin(train_piles)]
            val_labels = pile_labels[pile_labels['pile'].isin(val_piles)]
            
            train_class_counts = train_labels['BMA_label'].value_counts().sort_index()
            val_class_counts = val_labels['BMA_label'].value_counts().sort_index()
            
            print(f"\n  Fold {fold_idx+1}: {len(train_piles)} train, {len(val_piles)} val piles")
            print(f"    Train classes: {dict(train_class_counts)}")
            print(f"    Val classes: {dict(val_class_counts)}")
            
            # Verify all classes present in both sets
            all_classes_set = set(class_counts.index)
            missing_train = all_classes_set - set(train_class_counts.index)
            missing_val = all_classes_set - set(val_class_counts.index)
            
            if missing_train:
                print(f"    [ERROR] Train set missing classes: {sorted(missing_train)}")
            if missing_val:
                print(f"    [NOTE] Val set missing classes: {sorted(missing_val)} (expected for small classes)")
        
        return result_splits
    
    else:
        # Standard stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        fold_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(unique_piles, pile_bma_labels)):
            train_piles = unique_piles[train_idx].tolist()
            val_piles = unique_piles[val_idx].tolist()
            
            # Verify no overlap
            assert len(set(train_piles) & set(val_piles)) == 0, f"Overlap in fold {fold}!"
            
            fold_splits.append((train_piles, val_piles))
            print(f"  Fold {fold+1}: {len(train_piles)} train, {len(val_piles)} val piles")
        
        return fold_splits


def create_dataloaders(df, train_piles, val_piles, test_piles, image_dir, 
                       augmentation, max_images_per_pile, batch_size=1, 
                       include_original_and_augmented=False, num_augmentation_versions=1):
    """Create dataloaders for train/val/test sets"""
    
    # Create datasets from pile splits
    train_dataset = create_bag_dataset_from_piles(
        df, train_piles, image_dir, 
        augmentation=augmentation, 
        is_training=True, 
        max_images_per_pile=max_images_per_pile,
        include_original_and_augmented=include_original_and_augmented,
        num_augmentation_versions=num_augmentation_versions
    )
    
    val_dataset = create_bag_dataset_from_piles(
        df, val_piles, image_dir, 
        augmentation=None,  # No augmentation for validation
        is_training=False, 
        max_images_per_pile=max_images_per_pile,
        include_original_and_augmented=False,  # Always False for validation
        num_augmentation_versions=1
    )
    
    test_dataset = None
    if test_piles is not None:
        test_dataset = create_bag_dataset_from_piles(
            df, test_piles, image_dir, 
            augmentation=None,  # No augmentation for test
            is_training=False, 
            max_images_per_pile=max_images_per_pile,
            include_original_and_augmented=False,  # Always False for test
            num_augmentation_versions=1
        )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} bags from {len(train_piles)} piles")
    print(f"  Val:   {len(val_dataset)} bags from {len(val_piles)} piles")
    if test_dataset:
        print(f"  Test:  {len(test_dataset)} bags from {len(test_piles)} piles")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    return train_loader, val_loader, test_loader


def plot_training_history(train_losses, val_accuracies, val_f1_scores, fold=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    
    # Plot training loss
    axes[0].plot(train_losses)
    axes[0].set_title(f'Training Loss{fold_str}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Plot validation accuracy
    axes[1].plot(val_accuracies)
    axes[1].set_title(f'Validation Accuracy (Pile-level){fold_str}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    # Plot validation F1
    axes[2].plot(val_f1_scores)
    axes[2].set_title(f'Validation F1-Score (Pile-level){fold_str}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if fold is not None:
        plt.savefig(f'results/training_history_fold{fold}.png')
    else:
        plt.savefig(Config.TRAINING_PLOT_PATH)
    plt.close()


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--da_mode', type=str, default=None)
    parser.add_argument('--source_csv', type=str, default=None)
    parser.add_argument('--source_images', type=str, default=None)
    parser.add_argument('--target_csv', type=str, default=None)
    parser.add_argument('--target_images', type=str, default=None)
    parser.add_argument('--target_csv_labeled', type=str, default=None)
    parser.add_argument('--target_csv_unlabeled', type=str, default=None)
    parser.add_argument('--level', type=str, default=None)
    args = parser.parse_args()

    if args.level is not None:
        if args.level not in ['bag', 'pile']:
            raise ValueError('level must be bag or pile')
        Config.TRAINING_LEVEL = args.level
    if args.da_mode is not None:
        Config.DA_MODE = args.da_mode
        Config.DA_ENABLED = (args.da_mode == 'supervised')
        Config.UDA_ENABLED = (args.da_mode == 'unsupervised')
        Config.SSDA_ENABLED = (args.da_mode == 'semi')
        if Config.DA_MODE in ['supervised', 'unsupervised', 'semi'] and Config.TRAINING_LEVEL != 'bag':
            print('[WARNING] DA requires bag-level. Forcing level=bag')
            Config.TRAINING_LEVEL = 'bag'
    if args.source_csv is not None:
        Config.SOURCE_DATA_PATH = args.source_csv
        Config.QLD1_DATA_PATH = args.source_csv
    if args.source_images is not None:
        Config.SOURCE_IMAGE_DIR = args.source_images
        Config.QLD1_IMAGE_DIR = args.source_images
    if args.target_csv is not None:
        Config.TARGET_DATA_PATH = args.target_csv
        Config.QLD2_DATA_PATH = args.target_csv
    if args.target_images is not None:
        Config.TARGET_IMAGE_DIR = args.target_images
        Config.QLD2_IMAGE_DIR = args.target_images
    if args.target_csv_labeled is not None:
        Config.QLD2_LABELED_DATA_PATH = args.target_csv_labeled
    if args.target_csv_unlabeled is not None:
        Config.QLD2_UNLABELED_DATA_PATH = args.target_csv_unlabeled

    # Setup logging
    logger = setup_logging(log_dir=Config.LOG_DIR, mode='standard')
    
    print(f"\n{'='*80}")
    print(f"BMA MIL Classifier - End-to-End Training")
    print(f"{'='*80}")
    print(f"Architecture: Data (raw patches) -> FeatureExtractor + MIL -> Classification")
    print(f"Training: Bag-level (each image)")
    print(f"Evaluation: Pile-level (aggregated from bags)")
    print(f"Device: {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Split Mode: {Config.SPLIT_MODE}")
    print(f"DA Mode: {Config.DA_MODE}")
    print(f"Source CSV: {Config.SOURCE_DATA_PATH}")
    print(f"Source Images: {Config.SOURCE_IMAGE_DIR}")
    print(f"Target CSV: {Config.TARGET_DATA_PATH}")
    print(f"Target Images: {Config.TARGET_IMAGE_DIR}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(Config.DATA_PATH)
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)
    df1 = None
    df2 = None
    if Config.DA_ENABLED:
        df1 = pd.read_csv(Config.QLD1_DATA_PATH)
        df1 = df1[df1['BMA_label'] != 'BMA_label']
        df1['BMA_label'] = df1['BMA_label'].astype(int)
        df2 = pd.read_csv(Config.QLD2_DATA_PATH)
        df2 = df2[df2['BMA_label'] != 'BMA_label']
        df2['BMA_label'] = df2['BMA_label'].astype(int)
    
    # Filter out class 4 if needed
    if 4 in df['BMA_label'].unique():
        df = df[df['BMA_label'] != 4]
        print("Filtered out class 4")
    
    print(f"Total images: {len(df)}")
    print(f"Total piles: {df['pile'].nunique()}")
    print(f"Classes: {sorted(df['BMA_label'].unique())}")
    
    # Setup augmentation
    augmentation = get_augmentation_pipeline(is_training=True, target_size=224)
    
    # Initialize feature extractor
    print(f"\nInitializing feature extractor: {Config.FEATURE_EXTRACTOR_MODEL}")
    feature_extractor = FeatureExtractor(
        device='cpu',
        trainable_layers=0
    )
    
    # SPLIT MODE: Standard or K-Fold
    if Config.SPLIT_MODE == 'standard':
        print(f"\n{'='*80}")
        print("Running Standard Split (Train/Val/Test)")
        print(f"{'='*80}")
        
        if Config.DA_ENABLED and Config.TRAINING_LEVEL != 'pile':
            train_piles_s, val_piles_s, test_piles_s = split_piles_standard(
                df1,
                train_ratio=Config.TRAIN_RATIO,
                val_ratio=Config.VAL_RATIO,
                test_ratio=Config.TEST_RATIO,
                random_state=Config.RANDOM_STATE
            )
            train_piles_t, val_piles_t, test_piles_t = split_piles_standard(
                df2,
                train_ratio=Config.TRAIN_RATIO,
                val_ratio=Config.VAL_RATIO,
                test_ratio=Config.TEST_RATIO,
                random_state=Config.RANDOM_STATE
            )
        else:
            train_piles, val_piles, test_piles = split_piles_standard(
                df, 
                train_ratio=Config.TRAIN_RATIO,
                val_ratio=Config.VAL_RATIO,
                test_ratio=Config.TEST_RATIO,
                random_state=Config.RANDOM_STATE
            )
        
        # Compute class weights
        train_df = df[df['pile'].isin(train_piles)]
        class_weights = None
        if Config.USE_WEIGHTED_LOSS:
            class_weights = compute_class_weights(train_df, Config.NUM_CLASSES, Config.DEVICE)
        
        # Create dataloaders based on training level
        if Config.TRAINING_LEVEL == 'pile':
            print(f"\n{'='*60}")
            print(f"PILE-LEVEL TRAINING MODE")
            print(f"{'='*60}")
            print("Training: Pile-level (entire piles)")
            print("Validation: Pile-level")
            print("Weight updates: Per pile")
            
            # Calculate and display patch count
            num_aug_patches = 12 * Config.NUM_AUGMENTATION_VERSIONS
            total_patches = num_aug_patches + (12 if Config.INCLUDE_ORIGINAL_AND_AUGMENTED else 0)
            if Config.INCLUDE_ORIGINAL_AND_AUGMENTED:
                print(f"Augmentation: {total_patches} patches (12 original + {num_aug_patches} augmented versions)")
                print(f"             ({Config.NUM_AUGMENTATION_VERSIONS} augmentation versions per patch)")
            else:
                print(f"Augmentation: {total_patches} patches (only augmented, no originals)")
                print(f"             ({Config.NUM_AUGMENTATION_VERSIONS} augmentation versions per patch)")
            print(f"{'='*60}\n")
            
            # Create pile-level datasets
            train_dataset = create_pile_dataset_from_piles(
                df, train_piles, Config.IMAGE_DIR,
                augmentation=augmentation,
                is_training=True,
                max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
                include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
            )
            
            val_dataset = create_pile_dataset_from_piles(
                df, val_piles, Config.IMAGE_DIR,
                augmentation=None,
                is_training=False,
                max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
                include_original_and_augmented=False,  # Always False for validation
                num_augmentation_versions=1
            )
            
            test_dataset = create_pile_dataset_from_piles(
                df, test_piles, Config.IMAGE_DIR,
                augmentation=None,
                is_training=False,
                max_images_per_pile=Config.MAX_IMAGES_PER_PILE,
                include_original_and_augmented=False,  # Always False for test
                num_augmentation_versions=1
            )
            
            print(f"Dataset sizes (PILE-LEVEL):")
            print(f"  Train: {len(train_dataset)} piles")
            print(f"  Val:   {len(val_dataset)} piles")
            print(f"  Test:  {len(test_dataset)} piles")
            
            # Create dataloaders with custom collate function
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,  # Process 1 pile at a time
                shuffle=True,
                num_workers=0,
                collate_fn=collate_pile_batch,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_pile_batch,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_pile_batch,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
        else:  # bag-level training
            print(f"\n{'='*60}")
            print(f"BAG-LEVEL TRAINING MODE")
            print(f"{'='*60}")
            print("Training: Bag-level (individual images)")
            print("Validation: Pile-level (aggregated)")
            print("Weight updates: Per bag")
            
            # Calculate and display patch count
            num_aug_patches = 12 * Config.NUM_AUGMENTATION_VERSIONS
            total_patches = num_aug_patches + (12 if Config.INCLUDE_ORIGINAL_AND_AUGMENTED else 0)
            if Config.INCLUDE_ORIGINAL_AND_AUGMENTED:
                print(f"Augmentation: {total_patches} patches (12 original + {num_aug_patches} augmented versions)")
                print(f"             ({Config.NUM_AUGMENTATION_VERSIONS} augmentation versions per patch)")
            else:
                print(f"Augmentation: {total_patches} patches (only augmented, no originals)")
                print(f"             ({Config.NUM_AUGMENTATION_VERSIONS} augmentation versions per patch)")
            print(f"{'='*60}\n")
            
            if Config.UDA_ENABLED:
                train_loader_source, _, _ = create_dataloaders(
                    df1, train_piles_s, val_piles_s, test_piles_s,
                    Config.QLD1_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
                train_loader_target, val_loader, test_loader = create_dataloaders(
                    df2, train_piles_t, val_piles_t, test_piles_t,
                    Config.QLD2_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
            elif Config.SSDA_ENABLED:
                df2_l = pd.read_csv(Config.QLD2_LABELED_DATA_PATH)
                df2_l = df2_l[df2_l['BMA_label'] != 'BMA_label']
                df2_l['BMA_label'] = df2_l['BMA_label'].astype(int)
                df2_u = pd.read_csv(Config.QLD2_UNLABELED_DATA_PATH)
                df2_u = df2_u[df2_u['BMA_label'] != 'BMA_label']
                df2_u['BMA_label'] = df2_u['BMA_label'].astype(int)
                train_loader_source, _, _ = create_dataloaders(
                    df1, train_piles_s, val_piles_s, test_piles_s,
                    Config.QLD1_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
                train_loader_target_labeled, val_loader, test_loader = create_dataloaders(
                    df2_l, train_piles_t, val_piles_t, test_piles_t,
                    Config.QLD2_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
                train_loader_target_unlabeled, _, _ = create_dataloaders(
                    df2_u, train_piles_t, val_piles_t, test_piles_t,
                    Config.QLD2_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
            elif Config.DA_ENABLED:
                train_loader_source, _, _ = create_dataloaders(
                    df1, train_piles_s, val_piles_s, test_piles_s,
                    Config.QLD1_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
                train_loader_target, val_loader, test_loader = create_dataloaders(
                    df2, train_piles_t, val_piles_t, test_piles_t,
                    Config.QLD2_IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
            else:
                train_loader, val_loader, test_loader = create_dataloaders(
                    df, train_piles, val_piles, test_piles,
                    Config.IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                    batch_size=1,
                    include_original_and_augmented=Config.INCLUDE_ORIGINAL_AND_AUGMENTED,
                    num_augmentation_versions=Config.NUM_AUGMENTATION_VERSIONS
                )
        
        # Create model
        print(f"\nCreating end-to-end model...")
        model = BMA_MIL_Classifier(
            feature_extractor=feature_extractor.model,
            feature_dim=Config.FEATURE_DIM,
            hidden_dim=Config.IMAGE_HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
            dropout=Config.DROPOUT_RATE,
            trainable_layers=Config.TRAINABLE_FEATURE_LAYERS
        )
        model = model.to(Config.DEVICE)
        
        # Print training mode
        if Config.TRAINABLE_FEATURE_LAYERS == 0:
            print(f"Feature Extractor: Fully Frozen")
        elif Config.TRAINABLE_FEATURE_LAYERS == -1:
            print(f"Feature Extractor: Fully Trainable")
        else:
            print(f"Feature Extractor: Last {Config.TRAINABLE_FEATURE_LAYERS} layers trainable")
        
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Train model based on training level
        print(f"\nStarting training...")
        if Config.TRAINING_LEVEL == 'pile':
            train_losses, val_accuracies, val_f1_scores = train_model_pile_level(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=Config.NUM_EPOCHS,
                learning_rate=Config.LEARNING_RATE,
                class_weights=class_weights,
                fold=None
            )
        else:
            if Config.UDA_ENABLED:
                train_losses, val_accuracies, val_f1_scores = train_model_uda(
                    model=model,
                    train_loader_source=train_loader_source,
                    train_loader_target=train_loader_target,
                    val_loader_target=val_loader,
                    num_epochs=Config.NUM_EPOCHS,
                    learning_rate=Config.LEARNING_RATE,
                    class_weights=class_weights,
                    fold=None
                )
            elif Config.SSDA_ENABLED:
                train_losses, val_accuracies, val_f1_scores = train_model_ssda(
                    model=model,
                    train_loader_source=train_loader_source,
                    train_loader_target_labeled=train_loader_target_labeled,
                    train_loader_target_unlabeled=train_loader_target_unlabeled,
                    val_loader_target=val_loader,
                    num_epochs=Config.NUM_EPOCHS,
                    learning_rate=Config.LEARNING_RATE,
                    class_weights=class_weights,
                    fold=None
                )
            elif Config.DA_ENABLED:
                train_losses, val_accuracies, val_f1_scores = train_model_da(
                    model=model,
                    train_loader_source=train_loader_source,
                    train_loader_target=train_loader_target,
                    val_loader_target=val_loader,
                    num_epochs=Config.NUM_EPOCHS,
                    learning_rate=Config.LEARNING_RATE,
                    class_weights=class_weights,
                    fold=None
                )
            else:
                train_losses, val_accuracies, val_f1_scores = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=Config.NUM_EPOCHS,
                    learning_rate=Config.LEARNING_RATE,
                    class_weights=class_weights,
                    fold=None
                )
        
        # Plot training history
        plot_training_history(train_losses, val_accuracies, val_f1_scores)
        
        # Evaluate on test set
        if test_loader is not None:
            print(f"\n{'='*80}")
            print("Final Evaluation on Test Set")
            print(f"{'='*80}")
            
            # Load best model
            checkpoint = torch.load(Config.BEST_MODEL_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_results = evaluate_model(model, test_loader, Config.DEVICE)
            
            # Calculate per-class F1 scores
            from sklearn.metrics import f1_score
            pile_f1_per_class = f1_score(test_results['pile_labels'], 
                                         test_results['pile_predictions'], 
                                         average=None, 
                                         zero_division=0)
            
            # Save results to log file
            save_results_to_file(
                results_data={
                    'weighted_f1': test_results['pile_f1'],
                    'bag_accuracy': test_results['bag_accuracy'],
                    'bag_f1': test_results['bag_f1']
                },
                accuracy=test_results['pile_accuracy'],
                f1_scores=pile_f1_per_class,
                confusion_matrix=test_results['pile_confusion_matrix'],
                mode='standard'
            )
            
            if logger and logger.hasHandlers():
                logger.info(f"Test Results - Pile Acc: {test_results['pile_accuracy']:.4f}, Pile F1: {test_results['pile_f1']:.4f}")
            print("\n[SUCCESS] Results saved to log directory")
    
    elif Config.SPLIT_MODE == 'kfold':
        print(f"\n{'='*80}")
        print(f"Running {Config.NUM_FOLDS}-Fold Cross-Validation")
        print(f"{'='*80}")
        
        # Get fold splits
        fold_splits = split_piles_kfold(df, Config.NUM_FOLDS, Config.RANDOM_STATE)
        
        fold_results = []
        
        for fold, (train_piles, val_piles) in enumerate(fold_splits):
            print(f"\n{'='*80}")
            print(f"Fold {fold+1}/{Config.NUM_FOLDS}")
            print(f"{'='*80}")
            
            # Display class distribution for this fold
            pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
            train_labels = pile_labels[pile_labels['pile'].isin(train_piles)]
            val_labels = pile_labels[pile_labels['pile'].isin(val_piles)]
            
            train_class_counts = train_labels['BMA_label'].value_counts().sort_index()
            val_class_counts = val_labels['BMA_label'].value_counts().sort_index()
            
            print(f"\nClass distribution for Fold {fold+1}:")
            print(f"  Training set:")
            for cls in sorted(train_class_counts.index):
                print(f"    Class {cls}: {train_class_counts[cls]} piles")
            print(f"  Validation set (test for this fold):")
            for cls in sorted(val_class_counts.index):
                print(f"    Class {cls}: {val_class_counts[cls]} piles")
            
            # Create dataloaders for this fold
            train_loader, val_loader, _ = create_dataloaders(
                df, train_piles, val_piles, None,
                Config.IMAGE_DIR, augmentation, Config.MAX_IMAGES_PER_PILE,
                batch_size=1
            )
            
            # Compute class weights
            train_df = df[df['pile'].isin(train_piles)]
            class_weights = None
            if Config.USE_WEIGHTED_LOSS:
                class_weights = compute_class_weights(train_df, Config.NUM_CLASSES, Config.DEVICE)
            
            # Create model for this fold
            model = BMA_MIL_Classifier(
                feature_extractor=feature_extractor.model,
                feature_dim=Config.FEATURE_DIM,
                hidden_dim=Config.IMAGE_HIDDEN_DIM,
                num_classes=Config.NUM_CLASSES,
                dropout=Config.DROPOUT_RATE,
                trainable_layers=Config.TRAINABLE_FEATURE_LAYERS
            )
            model = model.to(Config.DEVICE)
            
            if fold == 0:  # Print only for first fold
                if Config.TRAINABLE_FEATURE_LAYERS == 0:
                    print(f"Feature Extractor: Fully Frozen")
                elif Config.TRAINABLE_FEATURE_LAYERS == -1:
                    print(f"Feature Extractor: Fully Trainable")
                else:
                    print(f"Feature Extractor: Last {Config.TRAINABLE_FEATURE_LAYERS} layers trainable")
            
            # Train model
            train_losses, val_accuracies, val_f1_scores = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=Config.NUM_EPOCHS,
                learning_rate=Config.LEARNING_RATE,
                class_weights=class_weights,
                fold=fold+1
            )
            
            # Plot training history for this fold
            plot_training_history(train_losses, val_accuracies, val_f1_scores, fold=fold+1)
            
            # Store fold results
            best_acc = max(val_accuracies)
            best_f1 = max(val_f1_scores)
            fold_results.append({
                'fold': fold+1,
                'best_accuracy': best_acc,
                'best_f1': best_f1
            })
            
            print(f"\nFold {fold+1} Best Results: Acc={best_acc:.4f}, F1={best_f1:.4f}")
        
        # Print cross-validation summary
        print(f"\n{'='*80}")
        print("Cross-Validation Summary")
        print(f"{'='*80}")
        
        avg_acc = np.mean([r['best_accuracy'] for r in fold_results])
        std_acc = np.std([r['best_accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['best_f1'] for r in fold_results])
        std_f1 = np.std([r['best_f1'] for r in fold_results])
        
        print(f"\nAverage Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        print(f"\nPer-Fold Results:")
        for result in fold_results:
            print(f"  Fold {result['fold']}: Acc={result['best_accuracy']:.4f}, F1={result['best_f1']:.4f}")
        
        # For cross-validation, create a simple summary file manually
        if logger and logger.hasHandlers():
            logger.info(f"Cross-Validation Complete - Avg Acc: {avg_acc:.4f} ± {std_acc:.4f}, Avg F1: {avg_f1:.4f} ± {std_f1:.4f}")
        
        print("\n[SUCCESS] Cross-validation results logged")
    
    else:
        raise ValueError(f"Invalid SPLIT_MODE: {Config.SPLIT_MODE}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

