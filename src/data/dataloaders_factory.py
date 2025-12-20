# ============================================================================
# dataloader_factory.py
# ============================================================================

import os

from .siamese_dataset import SiameseDataset
from .triplet_dataset import TripletDataset
from .contrastive_dataset import ContrastiveDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

"""
DataLoader Factory
Unified functions to create dataloaders for any dataset type.
"""

def create_dataloaders(
    dataset_class,
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    test_size: float = 0.2,
    samples_per_writer: int = 100,
    target_size: int = 448,
    random_state: int = 42,
    **dataset_kwargs
):
    """
    Generic dataloader creator for any dataset type.
    
    Args:
        dataset_class: Dataset class to instantiate (SiameseDataset, etc.)
        data_root: Root directory with writer subdirectories
        batch_size: Batch size
        num_workers: Number of workers
        test_size: Train/test split ratio (0.0 = all train, 1.0 = all test)
        samples_per_writer: Samples to generate per writer
        target_size: Image size
        random_state: Random seed
        **dataset_kwargs: Additional args for dataset constructor
    
    Returns:
        (train_loader, test_loader, train_dataset, test_dataset)
    """
    # Get writer directories
    writer_dirs = [
        os.path.join(data_root, d) 
        for d in sorted(os.listdir(data_root)) 
        if os.path.isdir(os.path.join(data_root, d))
    ]
    
    # Split directories
    if test_size == 0.0:
        train_dirs, test_dirs = writer_dirs, []
    elif test_size == 1.0:
        train_dirs, test_dirs = [], writer_dirs
    else:
        train_dirs, test_dirs = train_test_split(
            writer_dirs, test_size=test_size, random_state=random_state
        )
    
    # Create datasets
    train_dataset = dataset_class(
        train_dirs, 
        train=True, 
        samples_per_writer=samples_per_writer,
        target_size=target_size,
        **dataset_kwargs
    ) if train_dirs else None
    
    test_dataset = dataset_class(
        test_dirs, 
        train=False, 
        samples_per_writer=samples_per_writer,
        target_size=target_size,
        **dataset_kwargs
    ) if test_dirs else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    ) if train_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if test_dataset else None
    
    return train_loader, test_loader, train_dataset, test_dataset


def create_kfold_dataloaders(
    dataset_class,
    data_root: str,
    n_splits: int = 5,
    current_fold: int = 0,
    batch_size: int = 16,
    num_workers: int = 4,
    samples_per_writer: int = 100,
    target_size: int = 448,
    random_state: int = 42,
    **dataset_kwargs
):
    """
    Generic K-Fold dataloader creator.
    
    Args:
        dataset_class: Dataset class to instantiate
        data_root: Root directory with writer subdirectories
        n_splits: Number of folds
        current_fold: Current fold index (0 to n_splits-1)
        batch_size: Batch size
        num_workers: Number of workers
        samples_per_writer: Samples to generate per writer
        target_size: Image size
        random_state: Random seed
        **dataset_kwargs: Additional args for dataset constructor
    
    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Get writer directories
    writer_dirs = [
        os.path.join(data_root, d) 
        for d in sorted(os.listdir(data_root)) 
        if os.path.isdir(os.path.join(data_root, d))
    ]
    
    # K-Fold split
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kfold.split(writer_dirs))
    train_idx, val_idx = splits[current_fold]
    
    train_dirs = [writer_dirs[i] for i in train_idx]
    val_dirs = [writer_dirs[i] for i in val_idx]
    
    # Create datasets
    train_dataset = dataset_class(
        train_dirs, 
        train=True, 
        samples_per_writer=samples_per_writer,
        target_size=target_size,
        **dataset_kwargs
    )
    
    val_dataset = dataset_class(
        val_dirs, 
        train=False, 
        samples_per_writer=samples_per_writer,
        target_size=target_size,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


# ============================================================================
# Convenience wrappers (backward compatibility)
# ============================================================================

# Siamese
def create_siamese_dataloaders(*args, **kwargs):
    """Create Siamese dataloaders."""
    return create_dataloaders(SiameseDataset, *args, **kwargs)

def create_siamese_kfold_dataloaders(*args, **kwargs):
    """Create Siamese K-Fold dataloaders."""
    return create_kfold_dataloaders(SiameseDataset, *args, **kwargs)


# Contrastive
def create_contrastive_dataloaders(*args, **kwargs):
    """Create Contrastive dataloaders."""
    return create_dataloaders(ContrastiveDataset, *args, **kwargs)

def create_contrastive_kfold_dataloaders(*args, **kwargs):
    """Create Contrastive K-Fold dataloaders."""
    return create_kfold_dataloaders(ContrastiveDataset, *args, **kwargs)


# Triplet
def create_triplet_dataloaders(*args, **kwargs):
    """Create Triplet dataloaders."""
    return create_dataloaders(TripletDataset, *args, **kwargs)

def create_triplet_kfold_dataloaders(*args, **kwargs):
    """Create Triplet K-Fold dataloaders."""
    return create_kfold_dataloaders(TripletDataset, *args, **kwargs)