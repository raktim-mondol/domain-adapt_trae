"""
Test script to demonstrate different pooling methods
Shows how each pooling strategy works with sample data
"""

import torch
import numpy as np
from src.utils.pooling import (
    mean_pooling, 
    max_pooling, 
    majority_voting,
    AttentionPooling,
    aggregate_all_methods,
    get_predictions_all_methods
)


def test_pooling_methods():
    """
    Test all pooling methods with sample bag probabilities
    """
    print("="*70)
    print("Testing Pooling Methods for Pile-Level Aggregation")
    print("="*70)
    
    # Sample bag probabilities for a pile with 5 bags
    # Each bag has probabilities for 3 classes
    bag_probs_numpy = np.array([
        [0.1, 0.7, 0.2],  # Bag 1: Predicts Class 1 (0.7)
        [0.2, 0.6, 0.2],  # Bag 2: Predicts Class 1 (0.6)
        [0.3, 0.5, 0.2],  # Bag 3: Predicts Class 1 (0.5)
        [0.4, 0.4, 0.2],  # Bag 4: Tie between Class 0 and 1
        [0.2, 0.3, 0.5],  # Bag 5: Predicts Class 2 (0.5)
    ])
    
    bag_probs_tensor = torch.tensor(bag_probs_numpy, dtype=torch.float32)
    
    print("\nSample Bag Probabilities (5 bags, 3 classes):")
    print("-" * 70)
    for i, probs in enumerate(bag_probs_numpy):
        pred_class = np.argmax(probs)
        print(f"Bag {i+1}: {probs} -> Predicted Class: {pred_class}")
    
    print("\n" + "="*70)
    print("Individual Pooling Method Results")
    print("="*70)
    
    # 1. Mean Pooling
    print("\n1. MEAN POOLING (Average)")
    print("-" * 70)
    mean_result = mean_pooling(bag_probs_tensor)
    mean_pred = torch.argmax(mean_result).item()
    print(f"Aggregated probabilities: {mean_result.numpy()}")
    print(f"Predicted class: {mean_pred}")
    print(f"Interpretation: Average of all bag probabilities")
    
    # 2. Max Pooling
    print("\n2. MAX POOLING")
    print("-" * 70)
    max_result = max_pooling(bag_probs_tensor)
    max_pred = torch.argmax(max_result).item()
    print(f"Aggregated probabilities: {max_result.numpy()}")
    print(f"Predicted class: {max_pred}")
    print(f"Interpretation: Maximum probability for each class across all bags")
    
    # 3. Majority Voting
    print("\n3. MAJORITY VOTING")
    print("-" * 70)
    majority_result = majority_voting(bag_probs_tensor)
    majority_pred = np.argmax(majority_result)
    print(f"Aggregated probabilities: {majority_result}")
    print(f"Predicted class: {majority_pred}")
    
    # Show voting breakdown
    bag_votes = [np.argmax(bag) for bag in bag_probs_numpy]
    from collections import Counter
    vote_counts = Counter(bag_votes)
    print(f"Vote breakdown: {dict(vote_counts)}")
    print(f"Interpretation: Each bag votes, majority class wins")
    
    # 4. Attention Pooling
    print("\n4. ATTENTION POOLING (with random initialization)")
    print("-" * 70)
    attention_model = AttentionPooling(num_classes=3)
    attention_model.eval()
    
    with torch.no_grad():
        attention_result, attention_weights = attention_model(bag_probs_tensor)
    
    attention_pred = torch.argmax(attention_result).item()
    print(f"Aggregated probabilities: {attention_result.numpy()}")
    print(f"Predicted class: {attention_pred}")
    print(f"Attention weights: {attention_weights.numpy()}")
    print(f"Interpretation: Learned weights for each bag (currently random)")
    print(f"Note: Attention weights would be trained during model training")
    
    # All methods together
    print("\n" + "="*70)
    print("All Methods - Side-by-Side Comparison")
    print("="*70)
    
    all_results = aggregate_all_methods(bag_probs_tensor, attention_model=attention_model)
    all_preds = get_predictions_all_methods(bag_probs_tensor, attention_model=attention_model)
    
    print(f"\n{'Method':<20} {'Predicted Class':<20} {'Confidence':<15}")
    print("-" * 70)
    
    for method, probs in all_results.items():
        pred = all_preds[method]
        if isinstance(probs, torch.Tensor):
            confidence = probs[pred].item()
            probs_np = probs.detach().numpy()
            probs_str = f"[{', '.join([f'{p:.3f}' for p in probs_np])}]"
        else:
            confidence = probs[pred]
            probs_str = f"[{', '.join([f'{p:.3f}' for p in probs])}]"
        
        print(f"{method.capitalize():<20} {pred:<20} {confidence:<15.3f}")
        print(f"{'':20} Probs: {probs_str}")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("\nIn this example:")
    print(f"  - {sum(1 for v in bag_votes if v == 1)}/5 bags predict Class 1")
    print(f"  - {sum(1 for v in bag_votes if v == 0)}/5 bags predict Class 0")
    print(f"  - {sum(1 for v in bag_votes if v == 2)}/5 bags predict Class 2")
    print(f"\nAgreement across methods:")
    
    method_names = list(all_preds.keys())
    predictions = [all_preds[m] for m in method_names]
    
    if len(set(predictions)) == 1:
        print(f"  [AGREE] All methods agree: Class {predictions[0]}")
    else:
        print(f"  [DISAGREE] Methods disagree:")
        for method in method_names:
            print(f"    - {method.capitalize()}: Class {all_preds[method]}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


def test_edge_cases():
    """
    Test edge cases for pooling methods
    """
    print("\n\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)
    
    # Case 1: All bags agree
    print("\nCase 1: All bags strongly predict the same class")
    print("-" * 70)
    unanimous_probs = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.1, 0.9, 0.0],
        [0.2, 0.7, 0.1],
    ])
    
    results = aggregate_all_methods(unanimous_probs)
    preds = get_predictions_all_methods(unanimous_probs)
    
    print("All methods should predict Class 1:")
    for method, pred in preds.items():
        status = "[OK]" if pred == 1 else "[FAIL]"
        print(f"  {status} {method.capitalize()}: Class {pred}")
    
    # Case 2: Strong disagreement
    print("\nCase 2: Bags strongly disagree")
    print("-" * 70)
    disagree_probs = torch.tensor([
        [0.9, 0.05, 0.05],  # Strong Class 0
        [0.05, 0.9, 0.05],  # Strong Class 1
        [0.05, 0.05, 0.9],  # Strong Class 2
    ])
    
    results = aggregate_all_methods(disagree_probs)
    preds = get_predictions_all_methods(disagree_probs)
    
    print("Methods may produce different results:")
    for method, pred in preds.items():
        print(f"  - {method.capitalize()}: Class {pred}")
    
    # Case 3: Single bag
    print("\nCase 3: Single bag (no aggregation needed)")
    print("-" * 70)
    single_bag = torch.tensor([[0.2, 0.5, 0.3]])
    
    results = aggregate_all_methods(single_bag)
    preds = get_predictions_all_methods(single_bag)
    
    print("All methods should give same result for single bag:")
    for method, pred in preds.items():
        status = "[OK]" if pred == 1 else "[FAIL]"
        print(f"  {status} {method.capitalize()}: Class {pred}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_pooling_methods()
    test_edge_cases()
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)
    print("\nYou can now use these pooling methods in your training:")
    print("  1. Set POOLING_METHOD in configs/config.py")
    print("  2. Train your model")
    print("  3. Evaluation will show all methods side-by-side")
    print("="*70)

