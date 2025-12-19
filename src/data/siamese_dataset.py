"""
Siamese Dataset
Dataset class for loading image pairs for Siamese networks.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from  .transforms import get_train_transforms, get_test_transforms


class SiameseDataset(Dataset):
    """
    Dataset for Siamese network training.
    Generates positive (same writer) and negative (different writer) pairs.
    """
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        positive_ratio: float = 0.5,
        pairs_per_writer: int = 50,
        target_size: int = 448,
    ):
        """
        Args:
            writer_dirs: List of directories, each containing images from one writer
            train: Whether this is training set (applies augmentation if True)
            positive_ratio: Proportion of positive pairs to generate
            pairs_per_writer: Number of pairs to generate per writer
            target_size: Target image size for resizing
        """
        self.writer_dirs = writer_dirs
        self.positive_ratio = positive_ratio
        self.train = train
        self.target_size = target_size
        
        # Define transforms
        if self.train:
            self.transform = get_train_transforms(self.target_size)
        else:
            self.transform = get_test_transforms(self.target_size)
        
        # Load image paths for each writer
        self.writer_images = {}
        for writer_dir in writer_dirs:
            writer_id = Path(writer_dir).name
            images = [
                os.path.join(writer_dir, f) 
                for f in os.listdir(writer_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            if len(images) >= 2:
                self.writer_images[writer_id] = images
        
        self.writer_ids = list(self.writer_images.keys())
        self.num_pairs = len(self.writer_ids) * pairs_per_writer
    
    def __len__(self) -> int:
        return self.num_pairs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (image1, image2, label) where label=1 for same writer, 0 otherwise
        """
        is_positive = random.random() < self.positive_ratio
        
        if is_positive:
            # Same writer - select two different images from same writer
            writer_id = random.choice(self.writer_ids)
            img1_path, img2_path = random.sample(self.writer_images[writer_id], 2)
            label = 1.0
        else:
            # Different writers
            writer1, writer2 = random.sample(self.writer_ids, 2)
            img1_path = random.choice(self.writer_images[writer1])
            img2_path = random.choice(self.writer_images[writer2])
            label = 0.0
        
        # Load images as grayscale
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        
        # Apply transforms
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    test_size: float = 0.2,
    pairs_per_writer: int = 100,
    target_size: int = 448,
    random_state: int = 42,
):
    """
    Create train and test dataloaders.
    
    Args:
        data_root: Root directory containing writer subdirectories
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        test_size: Proportion of writers to use for testing
        pairs_per_writer: Number of pairs to generate per writer
        target_size: Target image size
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, test_loader, train_dirs, test_dirs)
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    
    # Get all writer directories
    writer_dirs = [
        os.path.join(data_root, d) 
        for d in sorted(os.listdir(data_root)) 
        if os.path.isdir(os.path.join(data_root, d))
    ]
    
    # Split into train and test
    train_dirs, test_dirs = train_test_split(
        writer_dirs, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create datasets
    train_dataset = SiameseDataset(
        train_dirs, 
        train=True, 
        pairs_per_writer=pairs_per_writer,
        target_size=target_size
    )
    
    test_dataset = SiameseDataset(
        test_dirs, 
        train=False, 
        pairs_per_writer=pairs_per_writer,
        target_size=target_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset

def create_dataloaders_kfold(
    data_root: str,
    n_splits: int = 5,
    current_fold: int = 0,
    batch_size: int = 16,
    num_workers: int = 4,
    pairs_per_writer: int = 100,
    target_size: int = 448,
    random_state: int = 42,
):
    """
    Create train and test dataloaders for a specific K-Fold split.
    
    Args:
        data_root: Root directory containing writer subdirectories
        n_splits: Number of folds for K-Fold
        current_fold: Current fold index (0 to n_splits-1)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pairs_per_writer: Number of pairs to generate per writer
        target_size: Target image size
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader
    
    # Get all writer directories
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
    train_dataset = SiameseDataset(
        train_dirs, 
        train=True, 
        pairs_per_writer=pairs_per_writer,
        target_size=target_size
    )
    
    val_dataset = SiameseDataset(
        val_dirs, 
        train=False, 
        pairs_per_writer=pairs_per_writer,
        target_size=target_size
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
