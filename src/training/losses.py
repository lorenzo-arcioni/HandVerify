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
    Contrastive Loss using Cosine Similarity.
    Forces similar pairs to have high cosine similarity (close to 1)
    and dissimilar pairs to have low cosine similarity (below margin).
    """
    
    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Margin for negative pairs (in cosine similarity space)
                   Typical values: 0.2-0.5
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1: Features from first image (will be normalized)
            feat2: Features from second image (will be normalized)
            labels: 1 for similar, 0 for dissimilar
            
        Returns:
            Loss value
        """
        # L2 normalize features (project onto unit hypersphere)
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Cosine similarity ∈ [-1, 1]
        cosine_sim = F.cosine_similarity(feat1, feat2)
        
        # Convert to distance: higher similarity = lower distance
        # distance ∈ [0, 2], where 0 = identical, 2 = opposite
        distances = 1 - cosine_sim
        
        # Contrastive loss
        # Positive pairs: minimize distance (maximize similarity)
        loss_positive = labels * torch.pow(distances, 2)
        
        # Negative pairs: enforce distance > margin
        # (enforce similarity < 1 - margin)
        loss_negative = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        
        return torch.mean(loss_positive + loss_negative)

class TripletLoss(nn.Module):
    """
    Triplet Loss directly on Cosine Similarity.
    Maximizes sim(anchor, positive) - sim(anchor, negative).
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Cosine similarities
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        
        # We want: pos_sim > neg_sim + margin
        # Loss: max(0, margin - (pos_sim - neg_sim))
        losses = F.relu(
            self.margin - (pos_sim - neg_sim)
        )
        
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