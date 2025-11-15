"""
Evaluation utilities for pile-level metrics with multiple pooling strategies
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from .pooling import aggregate_all_methods, AttentionPooling


def evaluate_model(model, test_loader, device, use_all_pooling_methods=True):
    """
    Evaluate model on test set with pile-level aggregation
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set (returns bags)
        device: torch device
        use_all_pooling_methods: If True, evaluate with all pooling methods side-by-side
    
    Returns:
        dict with pile-level metrics for each pooling method
    """
    model.eval()
    
    pile_predictions = {}  # {pile_id: {'preds': [], 'label': int}}
    bag_predictions = []
    bag_labels = []
    bag_pile_ids = []
    
    print("\nEvaluating model on test set...")
    test_pbar = tqdm(test_loader, desc='Testing', unit='bag')
    
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
                pred_class = torch.argmax(pred_probs).item()
                
                # Store bag-level prediction
                bag_predictions.append(pred_class)
                bag_labels.append(label)
                bag_pile_ids.append(pile_id)
                
                # Store for pile-level aggregation
                if pile_id not in pile_predictions:
                    pile_predictions[pile_id] = {'preds': [], 'label': label}
                pile_predictions[pile_id]['preds'].append(pred_probs.cpu().numpy())
    
    # Bag-level metrics
    bag_acc = accuracy_score(bag_labels, bag_predictions)
    bag_f1 = f1_score(bag_labels, bag_predictions, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"\nBag-level Metrics (each image independently):")
    print(f"  Accuracy: {bag_acc:.4f}")
    print(f"  F1-Score: {bag_f1:.4f}")
    print(f"  Total bags: {len(bag_predictions)}")
    
    # Evaluate with all pooling methods if requested
    if use_all_pooling_methods:
        # Initialize attention pooling model (not trained, just for inference)
        num_classes = pile_predictions[list(pile_predictions.keys())[0]]['preds'][0].shape[0]
        attention_model = AttentionPooling(num_classes=num_classes).to(device)
        attention_model.eval()

        # Aggregate predictions using all methods (excluding untrained attention for bag-level training)
        pooling_methods = ['mean', 'max', 'majority']  # Removed 'attention' - only useful if trained
        results_by_method = {method: {'preds': [], 'labels': [], 'pile_ids': []} 
                            for method in pooling_methods}
        
        for pile_id, data in pile_predictions.items():
            bag_probs_tensor = torch.tensor(np.array(data['preds']), dtype=torch.float32).to(device)
            true_label = data['label']
            
            # Get predictions for all methods
            with torch.no_grad():
                aggregated = aggregate_all_methods(bag_probs_tensor, attention_model=attention_model)
            
            for method, agg_probs in aggregated.items():
                if isinstance(agg_probs, torch.Tensor):
                    pred_class = torch.argmax(agg_probs).item()
                else:
                    pred_class = np.argmax(agg_probs)
                
                results_by_method[method]['preds'].append(pred_class)
                results_by_method[method]['labels'].append(true_label)
                results_by_method[method]['pile_ids'].append(pile_id)
        
        # Calculate metrics for each method and print side-by-side
        print(f"\n{'='*80}")
        print(f"Pile-level Results - All Pooling Methods (side-by-side)")
        print(f"{'='*80}")
        print(f"\nTotal piles: {len(pile_predictions)}\n")
        
        # Summary table
        print(f"{'Method':<20} {'Accuracy':<12} {'F1-Score':<12}")
        print(f"{'-'*50}")
        
        pooling_results = {}
        for method in pooling_methods:
            preds = results_by_method[method]['preds']
            labels = results_by_method[method]['labels']
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
            cm = confusion_matrix(labels, preds)
            
            print(f"{method.capitalize():<20} {acc:<12.4f} {f1:<12.4f}")
            
            pooling_results[method] = {
                'accuracy': acc,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': preds,
                'labels': labels,
                'pile_ids': results_by_method[method]['pile_ids']
            }
        
        print(f"{'-'*50}\n")
        
        # Detailed results for each method
        for method in pooling_methods:
            result = pooling_results[method]
            print(f"\n{method.upper()} POOLING - Detailed Results")
            print(f"{'='*60}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"F1-Score: {result['f1_score']:.4f}")
            print(f"\nConfusion Matrix:")
            print(result['confusion_matrix'])
            print(f"\nClassification Report:")
            all_labels = sorted(set(result['labels']) | set(result['predictions']))
            print(classification_report(result['labels'], result['predictions'], 
                                       labels=all_labels,
                                       target_names=[f'Class {i}' for i in all_labels],
                                       zero_division=0))
        
        # Return comprehensive results
        return {
            'bag_accuracy': bag_acc,
            'bag_f1': bag_f1,
            'bag_predictions': bag_predictions,
            'bag_labels': bag_labels,
            'bag_pile_ids': bag_pile_ids,
            'pooling_results': pooling_results,
            'methods_evaluated': pooling_methods
        }
    
    else:
        # Original behavior: only mean pooling
        pile_true_labels = []
        pile_pred_labels = []
        pile_ids_list = []
        
        for pile_id, data in pile_predictions.items():
            # Average probabilities across all bags in pile
            avg_probs = np.mean(data['preds'], axis=0)  # [num_classes]
            pred_class = np.argmax(avg_probs)
            
            pile_pred_labels.append(pred_class)
            pile_true_labels.append(data['label'])
            pile_ids_list.append(pile_id)
        
        pile_acc = accuracy_score(pile_true_labels, pile_pred_labels)
        pile_f1 = f1_score(pile_true_labels, pile_pred_labels, average='weighted')
        pile_cm = confusion_matrix(pile_true_labels, pile_pred_labels)
        
        print(f"\nPile-level Metrics (Mean Pooling):")
        print(f"  Accuracy: {pile_acc:.4f}")
        print(f"  F1-Score: {pile_f1:.4f}")
        print(f"  Total piles: {len(pile_pred_labels)}")
        
        print(f"\nPile-level Confusion Matrix:")
        print(pile_cm)
        
        print(f"\nPile-level Classification Report:")
        all_labels = sorted(set(pile_true_labels) | set(pile_pred_labels))
        print(classification_report(pile_true_labels, pile_pred_labels, 
                                    labels=all_labels,
                                    target_names=[f'Class {i}' for i in all_labels],
                                    zero_division=0))
        
        return {
            'bag_accuracy': bag_acc,
            'bag_f1': bag_f1,
            'pile_accuracy': pile_acc,
            'pile_f1': pile_f1,
            'pile_confusion_matrix': pile_cm,
            'pile_predictions': pile_pred_labels,
            'pile_labels': pile_true_labels,
            'pile_ids': pile_ids_list,
            'bag_predictions': bag_predictions,
            'bag_labels': bag_labels,
            'bag_pile_ids': bag_pile_ids
        }
