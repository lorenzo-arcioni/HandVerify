"""
Base Dataset Classes
Provides common functionality for all dataset types.
"""

import os
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_test_transforms


class BaseWriterDataset(Dataset, ABC):
    """
    Base class for writer verification datasets.
    Handles common operations like loading writer directories and transforms.
    """
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        samples_per_writer: int = 100,
        target_size: int = 448,
        min_images_per_writer: int = 2,
    ):
        """
        Args:
            writer_dirs: List of directories, each containing images from one writer
            train: Whether this is training set (applies augmentation if True)
            samples_per_writer: Number of samples to generate per writer
            target_size: Target image size for resizing
            min_images_per_writer: Minimum images required per writer
        """
        self.writer_dirs = writer_dirs
        self.train = train
        self.target_size = target_size
        self.samples_per_writer = samples_per_writer
        self.min_images_per_writer = min_images_per_writer
        
        # Define transforms
        self.transform = (get_train_transforms(target_size) if train 
                         else get_test_transforms(target_size))
        
        # Load writer images
        self.writer_images = self._load_writer_images()
        self.writer_ids = list(self.writer_images.keys())
        self.num_samples = len(self.writer_ids) * samples_per_writer
        
        print(f"{'TRAIN' if train else 'VAL'}: {len(self.writer_ids)} writers, "
              f"{self.num_samples} samples")
    
    def _load_writer_images(self) -> dict:
        """Load image paths for each writer."""
        writer_images = {}
        
        for writer_dir in self.writer_dirs:
            writer_id = Path(writer_dir).name
            images = [
                os.path.join(writer_dir, f) 
                for f in os.listdir(writer_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if len(images) >= self.min_images_per_writer:
                writer_images[writer_id] = images
        
        return writer_images
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and transform a single image."""
        img = Image.open(path).convert("L")
        return self.transform(img)
    
    def __len__(self) -> int:
        return self.num_samples
    
    @abstractmethod
    def __getitem__(self, idx: int):
        """Generate a sample (must be implemented by subclasses)."""
        pass
