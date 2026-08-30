# src/data/__init__.py
"""
Data module - Refactored with modular structure
"""

from .base_dataset import BaseWriterDataset
from .siamese_dataset import SiameseDataset
from .contrastive_dataset import ContrastiveDataset
from .triplet_dataset import TripletDataset

from .dataloader_factory import (
    create_dataloaders,
    create_cross_dataset_dataloaders,
    create_kfold_dataloaders,
    # Convenience wrappers - Siamese
    create_siamese_dataloaders,
    create_siamese_cross_dataset_dataloaders,
    create_siamese_kfold_dataloaders,
    # Convenience wrappers - Contrastive
    create_contrastive_dataloaders,
    create_contrastive_cross_dataset_dataloaders,
    create_contrastive_kfold_dataloaders,
    # Convenience wrappers - Triplet
    create_triplet_dataloaders,
    create_triplet_cross_dataset_dataloaders,
    create_triplet_kfold_dataloaders,
)

from .transforms import get_train_transforms, get_test_transforms

__all__ = [
    # Base class
    'BaseWriterDataset',

    # Dataset classes
    'SiameseDataset',
    'ContrastiveDataset',
    'TripletDataset',

    # Generic factory functions
    'create_dataloaders',                  # train/val/test, stesso dominio
    'create_cross_dataset_dataloaders',    # train/val/test, cross-dataset
    'create_kfold_dataloaders',

    # Type-specific convenience functions (backward compatibility)
    'create_siamese_dataloaders',
    'create_siamese_cross_dataset_dataloaders',
    'create_siamese_kfold_dataloaders',
    'create_contrastive_dataloaders',
    'create_contrastive_cross_dataset_dataloaders',
    'create_contrastive_kfold_dataloaders',
    'create_triplet_dataloaders',
    'create_triplet_cross_dataset_dataloaders',
    'create_triplet_kfold_dataloaders',

    # Transforms
    'get_train_transforms',
    'get_test_transforms',
]