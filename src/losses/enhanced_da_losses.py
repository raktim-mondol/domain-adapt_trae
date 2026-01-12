"""
Enhanced Domain Adaptation Losses
Implementation of advanced domain adaptation techniques from domain_adaptation_improvements.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.domain_discriminator import GradientReversal, DomainDiscriminator


def coral_loss(source_features, target_features):
    """
    Correlation Alignment Loss
    Aligns the second-order statistics between source and target features
    """
    d = source_features.size(1)
    
    # Calculate covariance matrices
    source_mean = torch.mean(source_features, dim=0, keepdim=True)
    target_mean = torch.mean(target_features, dim=0, keepdim=True)
    
    source_centered = source_features - source_mean
    target_centered = target_features - target_mean
    
    source_cov = torch.mm(source_centered.t(), source_centered) / (source_features.size(0) - 1)
    target_cov = torch.mm(target_centered.t(), target_centered) / (target_features.size(0) - 1)
    
    # Frobenius norm of the difference
    coral_loss = torch.mean((source_cov - target_cov) ** 2)
    
    return coral_loss


def cka_loss(source_features, target_features):
    """
    Centered Kernel Alignment Loss
    Measures similarity between feature representations
    """
    # Center the features
    source_centered = source_features - torch.mean(source_features, dim=0, keepdim=True)
    target_centered = target_features - torch.mean(target_features, dim=0, keepdim=True)
    
    # Compute Gram matrices
    gram_source = torch.mm(source_centered, source_centered.t())
    gram_target = torch.mm(target_centered, target_centered.t())
    
    # Normalize
    norm_source = torch.norm(gram_source)
    norm_target = torch.norm(gram_target)
    
    cka = torch.sum(gram_source * gram_target) / (norm_source * norm_target)
    
    return 1 - cka  # Minimize 1 - CKA


def compute_prediction_uncertainty(logits):
    """Compute uncertainty of predictions using entropy and margin methods"""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    # Margin-based uncertainty
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    
    return entropy, margin


def adaptive_pseudo_labeling(logits, threshold=0.9, uncertainty_weight=0.1):
    """Adaptive pseudo-labeling with uncertainty estimation"""
    probs = torch.softmax(logits, dim=-1)
    max_probs, pseudo_labels = torch.max(probs, dim=-1)
    
    # Compute uncertainty
    entropy, margin = compute_prediction_uncertainty(logits)
    
    # Combine confidence and uncertainty
    uncertainty_score = entropy * (1 - margin)
    adaptive_threshold = threshold - uncertainty_weight * uncertainty_score
    
    mask = max_probs >= adaptive_threshold
    return pseudo_labels, mask


def mixup_consistency_loss(features, labels, alpha=0.2):
    """Consistency regularization using mixup"""
    batch_size = features.size(0)
    indices = torch.randperm(batch_size)
    
    lambda_val = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1))
    lambda_val = lambda_val.to(features.device)
    
    mixed_features = lambda_val * features + (1 - lambda_val) * features[indices]
    mixed_labels = lambda_val * labels + (1 - lambda_val) * labels[indices]
    
    return mixed_features, mixed_labels


class AdaptiveLossWeighting:
    def __init__(self, num_losses, initial_weights=None):
        self.num_losses = num_losses
        self.weights = torch.ones(num_losses) if initial_weights is None else torch.tensor(initial_weights)
        self.loss_history = [[] for _ in range(num_losses)]
    
    def update_weights(self, current_losses, method='uncertainty'):
        """
        Update loss weights based on different strategies
        """
        current_losses = torch.tensor(current_losses)
        
        if method == 'uncertainty':
            # Weight based on inverse of recent loss variance
            for i in range(self.num_losses):
                self.loss_history[i].append(current_losses[i].item())
                if len(self.loss_history[i]) > 10:  # Keep last 10 values
                    self.loss_history[i] = self.loss_history[i][-10:]
            
            recent_losses = [torch.tensor(h[-5:]) for h in self.loss_history if len(h) >= 5]
            if len(recent_losses) == self.num_losses:
                variances = [torch.var(losses) for losses in recent_losses]
                variances = torch.tensor(variances)
                # Inverse weighting
                weights = 1.0 / (variances + 1e-8)
                self.weights = weights / torch.sum(weights)
        
        return self.weights


class ProgressiveDAScheduler:
    def __init__(self, start_ratio=0.1, end_ratio=1.0, total_epochs=100):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.total_epochs = total_epochs
    
    def get_current_ratio(self, epoch):
        """Linearly increase the ratio of target domain samples"""
        progress = min(1.0, epoch / self.total_epochs)
        return self.start_ratio + progress * (self.end_ratio - self.start_ratio)
    
    def get_domain_weight(self, epoch):
        """Return domain adaptation weight based on curriculum"""
        return min(1.0, epoch / (self.total_epochs * 0.3))  # Start DA after 30% of training


class MultiLevelDomainDiscriminator(nn.Module):
    def __init__(self, feature_dims, hidden_ratio=0.5, dropout=0.3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DomainDiscriminator(dim, hidden_ratio, dropout) for dim in feature_dims
        ])
    
    def forward(self, features_list):
        domain_logits = []
        for disc, feat in zip(self.discriminators, features_list):
            domain_logits.append(disc(feat))
        return domain_logits


class ConditionalDomainAdversarialLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, max_iter=1000, alpha=1.0):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.grl = GradientReversal()
        self.domain_discriminator = DomainDiscriminator(feature_dim * num_classes)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, feature, output, weight=None):
        softmax_output = torch.softmax(output, dim=1)
        batch_size = softmax_output.size(0)
        
        # Expand features with class predictions
        expanded_features = feature.unsqueeze(1).expand(-1, self.num_classes, -1)
        softmax_expanded = softmax_output.unsqueeze(2).expand(-1, -1, feature.size(1))
        conditional_features = (expanded_features * softmax_expanded.view(batch_size, self.num_classes, -1)).view(batch_size, -1)
        
        conditional_features = self.grl(conditional_features)
        domain_output = self.domain_discriminator(conditional_features)
        
        if weight is None:
            weight = torch.ones(batch_size, device=feature.device)
        
        true_labels = torch.zeros(batch_size, device=feature.device).long()
        false_labels = torch.ones(batch_size, device=feature.device).long()
        
        source_weight = torch.ones_like(domain_output)
        target_weight = torch.ones_like(domain_output) * weight
        
        source_loss = torch.sum(self.cross_entropy(domain_output, true_labels) * source_weight) / torch.sum(source_weight)
        target_loss = torch.sum(self.cross_entropy(domain_output, false_labels) * target_weight) / torch.sum(target_weight)
        
        return source_loss + target_loss


class DomainSpecificFeatureExtractor(nn.Module):
    def __init__(self, base_dim, num_domains):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Domain-specific layers
        self.domain_specific = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim // 2, base_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_domains)
        ])
        
        self.domain_classifier = nn.Linear(base_dim // 2, num_domains)
    
    def forward(self, x, domain_id=None):
        shared_features = self.shared(x)
        
        if domain_id is not None:
            domain_features = self.domain_specific[domain_id](shared_features)
        else:
            # Average across all domain-specific features
            domain_features = torch.stack([
                self.domain_specific[i](shared_features) 
                for i in range(len(self.domain_specific))
            ], dim=0).mean(dim=0)
        
        return domain_features


def compute_domain_gap(source_acc, target_acc):
    """Compute the domain gap between source and target performance"""
    return abs(source_acc - target_acc)


def compute_adaptation_efficiency(initial_gap, final_gap):
    """Compute how much of the domain gap was closed"""
    if initial_gap == 0:
        return 1.0
    return (initial_gap - final_gap) / initial_gap