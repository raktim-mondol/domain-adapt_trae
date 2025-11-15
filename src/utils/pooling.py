"""
Pooling strategies for pile-level prediction aggregation
Supports: Mean Pooling, Max Pooling, Attention Pooling, and Majority Voting
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating bag predictions
    Learns to weight different bag predictions based on their importance
    """
    def __init__(self, num_classes=3):
        super(AttentionPooling, self).__init__()
        self.num_classes = num_classes
        # Attention network: takes class probabilities and outputs attention weights
        self.attention = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, bag_probs):
        """
        Args:
            bag_probs: Tensor or list of tensors [num_bags, num_classes]
        
        Returns:
            aggregated_probs: [num_classes]
            attention_weights: [num_bags]
        """
        if isinstance(bag_probs, list):
            bag_probs = torch.stack(bag_probs)  # [num_bags, num_classes]
        
        # Compute attention scores
        attention_scores = self.attention(bag_probs)  # [num_bags, 1]
        attention_weights = torch.softmax(attention_scores, dim=0)  # [num_bags, 1]
        
        # Weighted sum of bag probabilities
        aggregated_probs = (bag_probs * attention_weights).sum(dim=0)  # [num_classes]
        
        return aggregated_probs, attention_weights.squeeze()


def mean_pooling(bag_probs):
    """
    Mean pooling: Average probabilities across all bags
    
    Args:
        bag_probs: List or array of bag probability predictions [num_bags, num_classes]
    
    Returns:
        aggregated_probs: [num_classes]
    """
    if isinstance(bag_probs, list):
        if isinstance(bag_probs[0], torch.Tensor):
            bag_probs = torch.stack(bag_probs)
        else:
            bag_probs = np.array(bag_probs)
    
    if isinstance(bag_probs, torch.Tensor):
        return bag_probs.mean(dim=0)
    else:
        return np.mean(bag_probs, axis=0)


def max_pooling(bag_probs):
    """
    Max pooling: Take maximum probability for each class across all bags
    
    Args:
        bag_probs: List or array of bag probability predictions [num_bags, num_classes]
    
    Returns:
        aggregated_probs: [num_classes]
    """
    if isinstance(bag_probs, list):
        if isinstance(bag_probs[0], torch.Tensor):
            bag_probs = torch.stack(bag_probs)
        else:
            bag_probs = np.array(bag_probs)
    
    if isinstance(bag_probs, torch.Tensor):
        return bag_probs.max(dim=0)[0]
    else:
        return np.max(bag_probs, axis=0)


def majority_voting(bag_probs):
    """
    Majority voting: Take the most common predicted class across all bags
    Returns one-hot encoded probabilities based on majority vote
    
    Args:
        bag_probs: List or array of bag probability predictions [num_bags, num_classes]
    
    Returns:
        aggregated_probs: [num_classes] - one-hot encoded based on majority vote
    """
    if isinstance(bag_probs, list):
        if isinstance(bag_probs[0], torch.Tensor):
            bag_probs = torch.stack(bag_probs)
        else:
            bag_probs = np.array(bag_probs)
    
    # Get predicted class for each bag
    if isinstance(bag_probs, torch.Tensor):
        bag_predictions = torch.argmax(bag_probs, dim=1).cpu().numpy()
        num_classes = bag_probs.shape[1]
    else:
        bag_predictions = np.argmax(bag_probs, axis=1)
        num_classes = bag_probs.shape[1]
    
    # Count votes for each class
    vote_counts = Counter(bag_predictions)
    majority_class = vote_counts.most_common(1)[0][0]
    
    # Create one-hot encoded result with majority class
    if isinstance(bag_probs, torch.Tensor):
        aggregated_probs = torch.zeros(num_classes)
        aggregated_probs[majority_class] = 1.0
    else:
        aggregated_probs = np.zeros(num_classes)
        aggregated_probs[majority_class] = 1.0
    
    return aggregated_probs


def aggregate_pile_predictions(bag_probs, method='mean', attention_model=None):
    """
    Aggregate bag predictions for a pile using specified method
    
    Args:
        bag_probs: List or array of bag probability predictions [num_bags, num_classes]
        method: One of ['mean', 'max', 'attention', 'majority']
        attention_model: AttentionPooling model (required if method='attention')
    
    Returns:
        aggregated_probs: [num_classes]
        attention_weights: [num_bags] if method='attention', else None
    """
    if method == 'mean':
        return mean_pooling(bag_probs), None
    elif method == 'max':
        return max_pooling(bag_probs), None
    elif method == 'attention':
        if attention_model is None:
            raise ValueError("attention_model required for attention pooling")
        return attention_model(bag_probs)
    elif method == 'majority':
        return majority_voting(bag_probs), None
    else:
        raise ValueError(f"Unknown pooling method: {method}. Choose from ['mean', 'max', 'attention', 'majority']")


def aggregate_all_methods(bag_probs, attention_model=None):
    """
    Aggregate using all pooling methods
    
    Args:
        bag_probs: List or array of bag probability predictions [num_bags, num_classes]
        attention_model: AttentionPooling model (optional, for attention pooling)
    
    Returns:
        dict with keys: 'mean', 'max', 'attention' (if model provided), 'majority'
        Each value is the aggregated probabilities [num_classes]
    """
    results = {}
    
    # Mean pooling
    results['mean'], _ = aggregate_pile_predictions(bag_probs, method='mean')
    
    # Max pooling
    results['max'], _ = aggregate_pile_predictions(bag_probs, method='max')
    
    # Majority voting
    results['majority'], _ = aggregate_pile_predictions(bag_probs, method='majority')
    
    # Attention pooling (if model provided)
    if attention_model is not None:
        results['attention'], _ = aggregate_pile_predictions(
            bag_probs, method='attention', attention_model=attention_model
        )
    
    return results


def get_predictions_all_methods(bag_probs, attention_model=None):
    """
    Get predicted classes using all pooling methods
    
    Args:
        bag_probs: List or array of bag probability predictions [num_bags, num_classes]
        attention_model: AttentionPooling model (optional)
    
    Returns:
        dict with keys: 'mean', 'max', 'attention' (if model provided), 'majority'
        Each value is the predicted class index
    """
    aggregated = aggregate_all_methods(bag_probs, attention_model)
    
    predictions = {}
    for method, probs in aggregated.items():
        if isinstance(probs, torch.Tensor):
            predictions[method] = torch.argmax(probs).item()
        else:
            predictions[method] = np.argmax(probs)
    
    return predictions

