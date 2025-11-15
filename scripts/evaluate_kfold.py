"""
Evaluate saved K-fold models and show per-fold and per-class performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from configs.config import Config
from src.models.bma_mil_model import BMA_MIL_Classifier
from src.feature_extractor import FeatureExtractor
from src.data.dataset import BMADataset
from src.utils.pooling import aggregate_all_methods, AttentionPooling
from src.augmentation import ComposedAugmentation
from tqdm import tqdm


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


def evaluate_fold(model, test_loader, device, fold_num):
    """
    Evaluate a single fold and return detailed metrics
    """
    model.eval()

    pile_predictions = {}  # {pile_id: {'preds': [], 'label': int}}

    print(f"\n{'='*60}")
    print(f"Evaluating Fold {fold_num}...")
    print(f"{'='*60}")

    test_pbar = tqdm(test_loader, desc=f'Testing Fold {fold_num}', unit='bag')

    with torch.no_grad():
        for bags, labels, pile_ids, image_paths in test_pbar:
            bags = bags.to(device)

            # Process each bag
            for i in range(bags.shape[0]):
                bag = bags[i]  # [num_patches, 3, H, W]
                pile_id = pile_ids[i]
                label = labels[i].item()

                # Get prediction for this bag
                logits, _ = model(bag)
                pred_probs = torch.softmax(logits, dim=0)  # [num_classes]

                # Store for pile-level aggregation
                if pile_id not in pile_predictions:
                    pile_predictions[pile_id] = {'preds': [], 'label': label}
                pile_predictions[pile_id]['preds'].append(pred_probs.cpu().numpy())

    # Pile-level aggregation with multiple pooling methods
    num_classes = pile_predictions[list(pile_predictions.keys())[0]]['preds'][0].shape[0]
    attention_model = AttentionPooling(num_classes=num_classes).to(device)
    attention_model.eval()

    pooling_methods = ['mean', 'max', 'majority']
    results_by_method = {method: {'preds': [], 'labels': [], 'pile_ids': []}
                        for method in pooling_methods}

    for pile_id, data in pile_predictions.items():
        bag_probs_tensor = torch.tensor(np.array(data['preds']), dtype=torch.float32).to(device)
        true_label = data['label']

        # Get predictions for all methods
        with torch.no_grad():
            aggregated = aggregate_all_methods(bag_probs_tensor, attention_model=attention_model)

        for method, agg_probs in aggregated.items():
            if method not in pooling_methods:
                continue  # Skip methods not in our list (e.g., attention)

            if isinstance(agg_probs, torch.Tensor):
                pred_class = torch.argmax(agg_probs).item()
            else:
                pred_class = np.argmax(agg_probs)

            results_by_method[method]['preds'].append(pred_class)
            results_by_method[method]['labels'].append(true_label)
            results_by_method[method]['pile_ids'].append(pile_id)

    # Calculate metrics for each pooling method
    fold_results = {}

    # Define all possible classes (0, 1, 2)
    all_classes = [0, 1, 2]

    for method in pooling_methods:
        preds = results_by_method[method]['preds']
        labels = results_by_method[method]['labels']

        acc = accuracy_score(labels, preds)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0, labels=all_classes)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0, labels=all_classes)
        cm = confusion_matrix(labels, preds, labels=all_classes)

        fold_results[method] = {
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': preds,
            'labels': labels,
            'num_piles': len(preds)
        }

    return fold_results


def print_fold_results(fold_num, fold_results, pooling_methods):
    """Print results for a single fold"""
    print(f"\n{'='*80}")
    print(f"FOLD {fold_num} RESULTS")
    print(f"{'='*80}")

    for method in pooling_methods:
        result = fold_results[method]
        print(f"\n{method.upper()} POOLING:")
        print(f"  Overall Accuracy: {result['accuracy']:.4f}")
        print(f"  Overall F1-Score (weighted): {result['f1_weighted']:.4f}")
        print(f"  F1-Score per class:")
        for class_idx, f1 in enumerate(result['f1_per_class']):
            print(f"    Class {class_idx}: {f1:.4f}")
        print(f"  Number of test piles: {result['num_piles']}")
        print(f"\n  Confusion Matrix:")
        print(result['confusion_matrix'])

        # Classification report - always use all 3 classes
        all_classes = [0, 1, 2]
        print(f"\n  Classification Report:")
        print(classification_report(result['labels'], result['predictions'],
                                   labels=all_classes,
                                   target_names=[f'Class {i}' for i in all_classes],
                                   zero_division=0))


def print_summary_results(all_fold_results, pooling_methods, num_classes):
    """Print average results across all folds"""
    print(f"\n{'='*80}")
    print(f"SUMMARY - AVERAGE ACROSS ALL FOLDS")
    print(f"{'='*80}")

    for method in pooling_methods:
        print(f"\n{method.upper()} POOLING:")

        # Collect metrics across folds
        accuracies = [fold_results[method]['accuracy'] for fold_results in all_fold_results]
        f1_weighted = [fold_results[method]['f1_weighted'] for fold_results in all_fold_results]
        f1_per_class_all_folds = [fold_results[method]['f1_per_class'] for fold_results in all_fold_results]

        # Calculate averages
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_f1_weighted = np.mean(f1_weighted)
        std_f1_weighted = np.std(f1_weighted)

        # Average F1 per class
        avg_f1_per_class = np.mean(f1_per_class_all_folds, axis=0)
        std_f1_per_class = np.std(f1_per_class_all_folds, axis=0)

        print(f"  Overall Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"  Overall F1-Score (weighted): {avg_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
        print(f"  F1-Score per class:")
        for class_idx in range(num_classes):
            print(f"    Class {class_idx}: {avg_f1_per_class[class_idx]:.4f} ± {std_f1_per_class[class_idx]:.4f}")

        print(f"\n  Per-fold breakdown:")
        print(f"  {'Fold':<8} {'Accuracy':<12} {'F1-Weighted':<15} " +
              " ".join([f"F1-Class{i:<2}" for i in range(num_classes)]))
        print(f"  {'-'*80}")
        for fold_idx, fold_results in enumerate(all_fold_results):
            f1_classes_str = " ".join([f"{fold_results[method]['f1_per_class'][i]:<10.4f}"
                                      for i in range(num_classes)])
            print(f"  {fold_idx+1:<8} {fold_results[method]['accuracy']:<12.4f} "
                  f"{fold_results[method]['f1_weighted']:<15.4f} {f1_classes_str}")


def main():
    config = Config()

    # Check if model files exist
    model_paths = [f'models/best_bma_mil_model_fold{i}.pth' for i in range(1, config.NUM_FOLDS + 1)]
    missing_models = [path for path in model_paths if not os.path.exists(path)]

    if missing_models:
        print("ERROR: Missing model files:")
        for path in missing_models:
            print(f"  - {path}")
        print("\nPlease train the models first using scripts/train.py with SPLIT_MODE='kfold'")
        return

    print(f"Found all {config.NUM_FOLDS} fold models!")
    print(f"Device: {config.DEVICE}")

    # Load data - SAME WAY AS TRAINING
    df = load_data_splits(config)

    # Get fold splits - EXACT SAME AS TRAINING (using custom split function)
    fold_splits = split_piles_kfold(df, config.NUM_FOLDS, config.RANDOM_STATE)

    all_fold_results = []
    pooling_methods = ['mean', 'max', 'majority']

    # Evaluate each fold - SAME SPLIT AS TRAINING
    for fold, (train_piles, test_piles) in enumerate(fold_splits):
        fold_num = fold + 1

        # Get test piles for this fold (val piles during training)
        test_pile_ids = set(test_piles)

        # Get test images
        test_df = df[df['pile'].isin(test_pile_ids)].reset_index(drop=True)

        print(f"\nFold {fold_num}: Test piles = {len(test_pile_ids)}, Test images = {len(test_df)}")

        # Prepare image data list for BMADataset
        image_data_list = []
        for _, row in test_df.iterrows():
            image_data_list.append({
                'image_path': row['image_path'],
                'pile_id': row['pile'],
                'pile_label': row['BMA_label'] - 1  # Convert to 0-indexed for model
            })

        # Create CLAHE-only augmentation (preprocessing for test)
        test_augmentation = ComposedAugmentation(
            histogram_method=config.HISTOGRAM_METHOD,
            enable_geometric=False,
            enable_color=False,
            enable_noise=False,
            is_training=False,
            target_size=config.TARGET_SIZE
        )

        # Create test dataset with CLAHE preprocessing
        test_dataset = BMADataset(
            image_data_list=image_data_list,
            image_dir=config.IMAGE_DIR,
            augmentation=test_augmentation,
            is_training=False,
            include_original_and_augmented=False,
            num_augmentation_versions=1
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        # Load model
        model_path = f'models/best_bma_mil_model_fold{fold_num}.pth'
        print(f"Loading model: {model_path}")

        # Create feature extractor
        feature_extractor = FeatureExtractor(
            device=config.DEVICE,
            trainable_layers=config.TRAINABLE_FEATURE_LAYERS
        )

        # Create model
        model = BMA_MIL_Classifier(
            feature_extractor=feature_extractor.model,
            feature_dim=config.FEATURE_DIM,
            hidden_dim=config.IMAGE_HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT_RATE,
            trainable_layers=config.TRAINABLE_FEATURE_LAYERS
        ).to(config.DEVICE)

        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate fold
        fold_results = evaluate_fold(model, test_loader, config.DEVICE, fold_num)
        all_fold_results.append(fold_results)

        # Print fold results
        print_fold_results(fold_num, fold_results, pooling_methods)

    # Print summary across all folds
    print_summary_results(all_fold_results, pooling_methods, config.NUM_CLASSES)

    # Save results to file
    results_file = 'results/kfold_test_performance.txt'
    os.makedirs('results', exist_ok=True)

    with open(results_file, 'w') as f:
        f.write(f"K-Fold Cross-Validation Test Performance ({config.NUM_FOLDS} folds)\n")
        f.write(f"{'='*80}\n\n")

        for fold_num, fold_results in enumerate(all_fold_results, 1):
            f.write(f"\nFOLD {fold_num} RESULTS\n")
            f.write(f"{'='*80}\n")

            for method in pooling_methods:
                result = fold_results[method]
                f.write(f"\n{method.upper()} POOLING:\n")
                f.write(f"  Overall Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Overall F1-Score (weighted): {result['f1_weighted']:.4f}\n")
                f.write(f"  F1-Score per class:\n")
                for class_idx, f1 in enumerate(result['f1_per_class']):
                    f.write(f"    Class {class_idx}: {f1:.4f}\n")
                f.write(f"  Number of test piles: {result['num_piles']}\n")

        f.write(f"\n\n{'='*80}\n")
        f.write(f"SUMMARY - AVERAGE ACROSS ALL FOLDS\n")
        f.write(f"{'='*80}\n")

        for method in pooling_methods:
            f.write(f"\n{method.upper()} POOLING:\n")

            accuracies = [fold_results[method]['accuracy'] for fold_results in all_fold_results]
            f1_weighted = [fold_results[method]['f1_weighted'] for fold_results in all_fold_results]
            f1_per_class_all_folds = [fold_results[method]['f1_per_class'] for fold_results in all_fold_results]

            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            avg_f1_weighted = np.mean(f1_weighted)
            std_f1_weighted = np.std(f1_weighted)
            avg_f1_per_class = np.mean(f1_per_class_all_folds, axis=0)
            std_f1_per_class = np.std(f1_per_class_all_folds, axis=0)

            f.write(f"  Overall Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"  Overall F1-Score (weighted): {avg_f1_weighted:.4f} ± {std_f1_weighted:.4f}\n")
            f.write(f"  F1-Score per class:\n")
            for class_idx in range(config.NUM_CLASSES):
                f.write(f"    Class {class_idx}: {avg_f1_per_class[class_idx]:.4f} ± {std_f1_per_class[class_idx]:.4f}\n")

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
