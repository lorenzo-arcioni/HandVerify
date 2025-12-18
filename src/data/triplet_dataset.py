"""
Triplet Dataset
Dataset class for loading image triplets for metric learning.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from .transforms import get_train_transforms, get_test_transforms


class TripletDataset(Dataset):
    """
    Dataset for Triplet network training.
    Generates triplets: (anchor, positive, negative)
    - anchor and positive are from the same writer
    - negative is from a different writer
    """
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        triplets_per_writer: int = 100,
        target_size: int = 448,
    ):
        """
        Args:
            writer_dirs: List of directories, each containing images from one writer
            train: Whether this is training set (applies augmentation if True)
            triplets_per_writer: Number of triplets to generate per writer
            target_size: Target image size for resizing
        """
        self.writer_dirs = writer_dirs
        self.train = train
        self.target_size = target_size
        self.triplets_per_writer = triplets_per_writer
        
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
            if len(images) >= 2:  # Need at least 2 images for anchor/positive
                self.writer_images[writer_id] = images
        
        self.writer_ids = list(self.writer_images.keys())
        self.num_triplets = len(self.writer_ids) * triplets_per_writer
        
        print(f"{'TRAIN' if train else 'TEST'}: {len(self.writer_ids)} writers, "
              f"{self.num_triplets} triplets")
    
    def __len__(self) -> int:
        return self.num_triplets
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (anchor, positive, negative) images
        """
        # Select anchor writer
        anchor_writer = random.choice(self.writer_ids)
        
        # Get anchor and positive from same writer
        anchor_path, positive_path = random.sample(self.writer_images[anchor_writer], 2)
        
        # Get negative from different writer
        negative_writer = random.choice([w for w in self.writer_ids if w != anchor_writer])
        negative_path = random.choice(self.writer_images[negative_writer])
        
        # Load images as grayscale
        anchor = Image.open(anchor_path).convert("L")
        positive = Image.open(positive_path).convert("L")
        negative = Image.open(negative_path).convert("L")
        
        # Apply transforms
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)
        
        return anchor, positive, negative


def create_triplet_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    test_size: float = 0.2,
    triplets_per_writer: int = 100,
    target_size: int = 448,
    random_state: int = 42,
):
    """
    Create train and test dataloaders for triplet learning.
    
    Args:
        data_root: Root directory containing writer subdirectories
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        test_size: Proportion of writers to use for testing
        triplets_per_writer: Number of triplets to generate per writer
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
    train_dataset = TripletDataset(
        train_dirs, 
        train=True, 
        triplets_per_writer=triplets_per_writer,
        target_size=target_size
    )
    
    test_dataset = TripletDataset(
        test_dirs, 
        train=False, 
        triplets_per_writer=triplets_per_writer,
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