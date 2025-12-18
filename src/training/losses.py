"""
Loss Functions
Various loss functions for Siamese and Triplet networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary Cross-Entropy Loss for similarity classification"""
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()
    
    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted similarities (0-1)
            labels: Ground truth labels (0 or 1)
            
        Returns:
            Loss value
        """
        return self.loss_fn(predictions, labels)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning.
    Forces similar pairs to be close and dissimilar pairs to be far apart.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for negative pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1: Features from first image
            feat2: Features from second image
            labels: 1 for similar, 0 for dissimilar
            
        Returns:
            Loss value
        """
        # Euclidean distance
        distances = F.pairwise_distance(feat1, feat2)
        
        # Contrastive loss
        loss_positive = labels * torch.pow(distances, 2)
        loss_negative = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        return torch.mean(loss_positive + loss_negative)


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    Ensures anchor-positive distance < anchor-negative distance by margin.
    """
    
    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Margin between positive and negative pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class)
            
        Returns:
            Triplet loss value
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class CombinedLoss(nn.Module):
    """
    Combines BCE loss with contrastive loss for stronger training signal.
    """
    
    def __init__(self, margin: float = 1.0, alpha: float = 0.5):
        """
        Args:
            margin: Margin for contrastive loss
            alpha: Weight for BCE loss (1-alpha for contrastive)
        """
        super().__init__()
        self.bce_loss = BCELoss()
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.alpha = alpha
    
    def forward(self, predictions: torch.Tensor, feat1: torch.Tensor, 
                feat2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted similarities from classifier
            feat1: Normalized features from first image
            feat2: Normalized features from second image
            labels: Ground truth labels
            
        Returns:
            Combined loss value
        """
        bce = self.bce_loss(predictions, labels)
        contrastive = self.contrastive_loss(feat1, feat2, labels)
        
        return self.alpha * bce + (1 - self.alpha) * contrastive