# src/training/__init__.py
"""
Training module
Aggregates trainers and loss functions for easy import.
"""

# Loss functions
from .losses import BCELoss, ContrastiveLoss, TripletLoss, CombinedLoss

# Trainers
from .trainer_bce import BCETrainer
from .trainer_triplet import TripletTrainer
from .trainer_contrastive import ContrastiveTrainer

__all__ = [
    # Losses
    'BCELoss',
    'ContrastiveLoss',
    'TripletLoss',
    'CombinedLoss',
    
    # Trainers
    'BCETrainer',
    'TripletTrainer',
    'ContrastiveTrainer',
]