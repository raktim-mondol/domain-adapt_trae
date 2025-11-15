"""
Test script to verify k-fold splitting with small classes
"""

import numpy as np
import pandas as pd

def test_cyclic_distribution():
    """Test the cyclic distribution logic for small classes"""
    
    # Simulate your data
    data = {
        'pile': [],
        'BMA_label': []
    }
    
    # Class 1: 2 piles
    for i in range(2):
        data['pile'].append(f'pile_1_{i}')
        data['BMA_label'].append(1)
    
    # Class 2: 48 piles
    for i in range(48):
        data['pile'].append(f'pile_2_{i}')
        data['BMA_label'].append(2)
    
    # Class 3: 22 piles
    for i in range(22):
        data['pile'].append(f'pile_3_{i}')
        data['BMA_label'].append(3)
    
    df = pd.DataFrame(data)
    pile_labels = df.groupby('pile')['BMA_label'].first().reset_index()
    
    print("="*80)
    print("Testing K-Fold Split with Small Classes")
    print("="*80)
    
    # Parameters
    n_folds = 3
    random_state = 42
    
    # Show class distribution
    class_counts = pile_labels['BMA_label'].value_counts().sort_index()
    print(f"\nPile-level class distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} piles")
    
    print(f"\nSetting up {n_folds}-fold cross-validation")
    print(f"Total piles: {len(pile_labels)}")
    
    # Simulate the cyclic distribution
    np.random.seed(random_state)
    unique_piles = pile_labels['pile'].values
    fold_splits = [{'train': [], 'val': []} for _ in range(n_folds)]
    
    print("\n" + "="*80)
    print("Cyclic Distribution for Small Classes")
    print("="*80)
    
    for cls in sorted(class_counts.index):
        cls_piles = pile_labels[pile_labels['BMA_label'] == cls]['pile'].values
        cls_piles_shuffled = np.random.permutation(cls_piles)
        n_cls = len(cls_piles_shuffled)
        
        print(f"\nClass {cls}: {n_cls} piles")
        
        if n_cls >= n_folds:
            # Standard stratified split
            fold_size = n_cls // n_folds
            remainder = n_cls % n_folds
            
            start_idx = 0
            for fold in range(n_folds):
                end_idx = start_idx + fold_size + (1 if fold < remainder else 0)
                fold_splits[fold]['val'].extend(cls_piles_shuffled[start_idx:end_idx])
                print(f"  Fold {fold+1} val: {list(cls_piles_shuffled[start_idx:end_idx])}")
                start_idx = end_idx
        else:
            # Cyclic distribution
            print(f"  Using cyclic distribution (n_cls={n_cls} < n_folds={n_folds})")
            print(f"  Available piles: {list(cls_piles_shuffled)}")
            
            for fold in range(n_folds):
                pile_idx = fold % n_cls
                selected_pile = cls_piles_shuffled[pile_idx]
                fold_splits[fold]['val'].append(selected_pile)
                print(f"  Fold {fold+1}: fold % {n_cls} = {pile_idx} -> {selected_pile}")
    
    # Assign training sets
    print("\n" + "="*80)
    print("Final Fold Splits")
    print("="*80)
    
    for fold in range(n_folds):
        val_set = set(fold_splits[fold]['val'])
        train_set = [p for p in unique_piles if p not in val_set]
        fold_splits[fold]['train'] = train_set
        
        # Get class counts
        train_labels = pile_labels[pile_labels['pile'].isin(train_set)]
        val_labels = pile_labels[pile_labels['pile'].isin(val_set)]
        
        train_class_counts = train_labels['BMA_label'].value_counts().sort_index()
        val_class_counts = val_labels['BMA_label'].value_counts().sort_index()
        
        print(f"\nFold {fold+1}:")
        print(f"  Train: {len(train_set)} piles")
        print(f"    Class distribution: {dict(train_class_counts)}")
        print(f"  Val: {len(val_set)} piles")
        print(f"    Class distribution: {dict(val_class_counts)}")
        
        # Check for missing classes
        all_classes = set(class_counts.index)
        missing_train = all_classes - set(train_class_counts.index)
        missing_val = all_classes - set(val_class_counts.index)
        
        if missing_train:
            print(f"    [ERROR] MISSING in train: {sorted(missing_train)}")
        else:
            print(f"    [OK] All classes present in train")
            
        if missing_val:
            print(f"    [WARNING] MISSING in val: {sorted(missing_val)}")
        else:
            print(f"    [OK] All classes present in val")
        
        # Verify no overlap
        overlap = val_set & set(train_set)
        if overlap:
            print(f"    [ERROR] OVERLAP: {overlap}")
        else:
            print(f"    [OK] No overlap between train/val")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)

if __name__ == '__main__':
    test_cyclic_distribution()

