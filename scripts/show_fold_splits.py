"""
Show which piles are used for training and validation in each K-fold split,
along with category distribution for each fold.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

from configs.config import Config


def split_piles_kfold(df, n_folds=5, random_state=42):
    """
    Split piles for k-fold cross-validation with handling for small classes.
    EXACT COPY from train.py (lines 183-331) to ensure identical splits.
    """
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
                # Re-calculate: for small classes, ensure proper split
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


def load_data_splits(config):
    """Load data and create pile-level splits (same as training)"""
    # Load CSV
    df = pd.read_csv(config.DATA_PATH)
    
    # Filter out header rows if any (same as training line 462)
    df = df[df['BMA_label'] != 'BMA_label']
    df['BMA_label'] = df['BMA_label'].astype(int)
    
    # Filter out class 4 if needed (same as training line 466-468)
    if 4 in df['BMA_label'].unique():
        df = df[df['BMA_label'] != 4]
        print("Filtered out class 4 (as in training)")
    
    return df


def analyze_fold_splits(config):
    """Analyze and display K-fold splits with pile names and category distribution"""
    
    # Load data - SAME WAY AS TRAINING
    df = load_data_splits(config)
    
    # Get fold splits - EXACT SAME AS TRAINING (using custom split function)
    fold_splits = split_piles_kfold(df, config.NUM_FOLDS, config.RANDOM_STATE)
    
    print(f"\n{'='*100}")
    print(f"K-FOLD CROSS-VALIDATION SPLIT ANALYSIS")
    print(f"{'='*100}")
    print(f"\nConfiguration:")
    print(f"  Number of Folds: {config.NUM_FOLDS}")
    print(f"  Random State: {config.RANDOM_STATE}")
    print(f"  Total Piles: {df['pile'].nunique()}")
    print(f"  Total Images: {len(df)}")
    
    # Get pile labels for analysis
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    pile_bma_labels = pile_labels['BMA_label'].values - 1  # 0-indexed
    
    print(f"\nOverall Category Distribution (Piles):")
    print(f"  [Using 0-indexed labels for consistency with training]")
    
    # Count piles per category (0-indexed)
    overall_pile_distribution = Counter(pile_bma_labels)
    all_classes_in_data = sorted(overall_pile_distribution.keys())
    
    for class_idx in all_classes_in_data:
        count = overall_pile_distribution[class_idx]
        percentage = (count / len(pile_labels)) * 100
        # Show both 0-indexed (used in code) and original 1-indexed (in CSV)
        print(f"  Class {class_idx} (CSV: {class_idx+1}): {count} piles ({percentage:.1f}%)")
    
    # Create a mapping from pile to label (0-indexed for consistency)
    pile_to_label = dict(zip(pile_labels['pile'], pile_bma_labels))
    
    # Analyze each fold
    all_fold_info = []
    
    for fold, (train_pile_ids, val_pile_ids) in enumerate(fold_splits):
        fold_num = fold + 1
        
        # Get labels for train and validation piles
        train_labels = [pile_to_label[pile] for pile in train_pile_ids]
        val_labels = [pile_to_label[pile] for pile in val_pile_ids]
        
        # Count category distribution
        train_distribution = Counter(train_labels)
        val_distribution = Counter(val_labels)
        
        # Get image counts
        train_df = df[df['pile'].isin(train_pile_ids)]
        val_df = df[df['pile'].isin(val_pile_ids)]
        
        fold_info = {
            'fold_num': fold_num,
            'train_piles': sorted(train_pile_ids),
            'val_piles': sorted(val_pile_ids),
            'train_distribution': train_distribution,
            'val_distribution': val_distribution,
            'train_image_count': len(train_df),
            'val_image_count': len(val_df)
        }
        
        all_fold_info.append(fold_info)
    
    # Print detailed results for each fold
    for fold_info in all_fold_info:
        print(f"\n{'='*100}")
        print(f"FOLD {fold_info['fold_num']}")
        print(f"{'='*100}")
        
        print(f"\nTRAINING SET:")
        print(f"  Total Piles: {len(fold_info['train_piles'])}")
        print(f"  Total Images: {fold_info['train_image_count']}")
        print(f"\n  Category Distribution (Piles):")
        for class_idx in all_classes_in_data:
            count = fold_info['train_distribution'].get(class_idx, 0)
            percentage = (count / len(fold_info['train_piles'])) * 100 if len(fold_info['train_piles']) > 0 else 0
            print(f"    Class {class_idx}: {count} piles ({percentage:.1f}%)")
        
        print(f"\n  Pile Names:")
        # Group training piles by category
        train_piles_by_class = {i: [] for i in all_classes_in_data}
        for pile in fold_info['train_piles']:
            label = pile_to_label[pile]
            train_piles_by_class[label].append(pile)
        
        for class_idx in sorted(train_piles_by_class.keys()):
            piles = train_piles_by_class[class_idx]
            if piles:
                print(f"    Class {class_idx}: {', '.join(map(str, piles))}")
        
        print(f"\nVALIDATION SET:")
        print(f"  Total Piles: {len(fold_info['val_piles'])}")
        print(f"  Total Images: {fold_info['val_image_count']}")
        print(f"\n  Category Distribution (Piles):")
        for class_idx in all_classes_in_data:
            count = fold_info['val_distribution'].get(class_idx, 0)
            percentage = (count / len(fold_info['val_piles'])) * 100 if len(fold_info['val_piles']) > 0 else 0
            print(f"    Class {class_idx}: {count} piles ({percentage:.1f}%)")
        
        print(f"\n  Pile Names:")
        # Group validation piles by category
        val_piles_by_class = {i: [] for i in all_classes_in_data}
        for pile in fold_info['val_piles']:
            label = pile_to_label[pile]
            val_piles_by_class[label].append(pile)
        
        for class_idx in sorted(val_piles_by_class.keys()):
            piles = val_piles_by_class[class_idx]
            if piles:
                print(f"    Class {class_idx}: {', '.join(map(str, piles))}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY TABLE - PILE DISTRIBUTION ACROSS FOLDS")
    print(f"{'='*100}")
    
    # Create dynamic header based on classes present in data
    header = f"\n{'Fold':<8} {'Set':<12} {'Total':<10}"
    for class_idx in all_classes_in_data:
        header += f" {'Class ' + str(class_idx):<10}"
    print(header)
    print(f"{'-'*70}")
    
    for fold_info in all_fold_info:
        # Training row
        train_str = f"{fold_info['fold_num']:<8} {'Training':<12} {len(fold_info['train_piles']):<10}"
        for class_idx in all_classes_in_data:
            train_count = fold_info['train_distribution'].get(class_idx, 0)
            train_str += f" {train_count:<10}"
        print(train_str)
        
        # Validation row
        val_str = f"{'':<8} {'Validation':<12} {len(fold_info['val_piles']):<10}"
        for class_idx in all_classes_in_data:
            val_count = fold_info['val_distribution'].get(class_idx, 0)
            val_str += f" {val_count:<10}"
        print(val_str)
        print(f"{'-'*70}")
    
    # Save results to file
    results_file = 'results/kfold_split_analysis.txt'
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"K-FOLD CROSS-VALIDATION SPLIT ANALYSIS\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Number of Folds: {config.NUM_FOLDS}\n")
        f.write(f"  Random State: {config.RANDOM_STATE}\n")
        f.write(f"  Total Piles: {len(pile_labels)}\n")
        f.write(f"  Total Images: {len(df)}\n")
        f.write(f"\nOverall Category Distribution (Piles):\n")
        f.write(f"  [Using 0-indexed labels for consistency with training]\n")
        
        for class_idx in all_classes_in_data:
            count = overall_pile_distribution[class_idx]
            percentage = (count / len(pile_labels)) * 100
            # Show both 0-indexed (used in code) and original 1-indexed (in CSV)
            f.write(f"  Class {class_idx} (CSV: {class_idx+1}): {count} piles ({percentage:.1f}%)\n")
        
        # Write detailed results for each fold
        for fold_info in all_fold_info:
            f.write(f"\n{'='*100}\n")
            f.write(f"FOLD {fold_info['fold_num']}\n")
            f.write(f"{'='*100}\n")
            
            f.write(f"\nTRAINING SET:\n")
            f.write(f"  Total Piles: {len(fold_info['train_piles'])}\n")
            f.write(f"  Total Images: {fold_info['train_image_count']}\n")
            f.write(f"\n  Category Distribution (Piles):\n")
            for class_idx in all_classes_in_data:
                count = fold_info['train_distribution'].get(class_idx, 0)
                percentage = (count / len(fold_info['train_piles'])) * 100 if len(fold_info['train_piles']) > 0 else 0
                f.write(f"    Class {class_idx}: {count} piles ({percentage:.1f}%)\n")
            
            f.write(f"\n  Pile Names:\n")
            train_piles_by_class = {i: [] for i in all_classes_in_data}
            for pile in fold_info['train_piles']:
                label = pile_to_label[pile]
                train_piles_by_class[label].append(pile)
            
            for class_idx in sorted(train_piles_by_class.keys()):
                piles = train_piles_by_class[class_idx]
                if piles:
                    f.write(f"    Class {class_idx}: {', '.join(map(str, piles))}\n")
            
            f.write(f"\nVALIDATION SET:\n")
            f.write(f"  Total Piles: {len(fold_info['val_piles'])}\n")
            f.write(f"  Total Images: {fold_info['val_image_count']}\n")
            f.write(f"\n  Category Distribution (Piles):\n")
            for class_idx in all_classes_in_data:
                count = fold_info['val_distribution'].get(class_idx, 0)
                percentage = (count / len(fold_info['val_piles'])) * 100 if len(fold_info['val_piles']) > 0 else 0
                f.write(f"    Class {class_idx}: {count} piles ({percentage:.1f}%)\n")
            
            f.write(f"\n  Pile Names:\n")
            val_piles_by_class = {i: [] for i in all_classes_in_data}
            for pile in fold_info['val_piles']:
                label = pile_to_label[pile]
                val_piles_by_class[label].append(pile)
            
            for class_idx in sorted(val_piles_by_class.keys()):
                piles = val_piles_by_class[class_idx]
                if piles:
                    f.write(f"    Class {class_idx}: {', '.join(map(str, piles))}\n")
        
        # Write summary table
        f.write(f"\n{'='*100}\n")
        f.write(f"SUMMARY TABLE - PILE DISTRIBUTION ACROSS FOLDS\n")
        f.write(f"{'='*100}\n\n")
        
        # Create dynamic header based on classes present in data
        header = f"{'Fold':<8} {'Set':<12} {'Total':<10}"
        for class_idx in all_classes_in_data:
            header += f" {'Class ' + str(class_idx):<10}"
        f.write(header + "\n")
        f.write(f"{'-'*70}\n")
        
        for fold_info in all_fold_info:
            # Training row
            train_str = f"{fold_info['fold_num']:<8} {'Training':<12} {len(fold_info['train_piles']):<10}"
            for class_idx in all_classes_in_data:
                train_count = fold_info['train_distribution'].get(class_idx, 0)
                train_str += f" {train_count:<10}"
            f.write(train_str + "\n")
            
            # Validation row
            val_str = f"{'':<8} {'Validation':<12} {len(fold_info['val_piles']):<10}"
            for class_idx in all_classes_in_data:
                val_count = fold_info['val_distribution'].get(class_idx, 0)
                val_str += f" {val_count:<10}"
            f.write(val_str + "\n")
            f.write(f"{'-'*70}\n")
    
    print(f"\n{'='*100}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*100}")


def main():
    config = Config()
    analyze_fold_splits(config)


if __name__ == "__main__":
    main()

