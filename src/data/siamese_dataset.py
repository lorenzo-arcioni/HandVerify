# ============================================================================
# siamese_dataset.py
# ============================================================================

import random
from pathlib import Path
from typing import List, Tuple

import torch

from .base_dataset import BaseWriterDataset

"""Siamese Dataset - Binary pairs with labels"""

class SiameseDataset(BaseWriterDataset):
    """Dataset for Siamese network training (positive/negative pairs)."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        positive_ratio: float = 0.5,
        pairs_per_writer: int = 100,
        target_size: int = 448,
    ):
        self.positive_ratio = positive_ratio
        super().__init__(
            writer_dirs=writer_dirs,
            train=train,
            samples_per_writer=pairs_per_writer,
            target_size=target_size,
            min_images_per_writer=2
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a pair with label."""
        is_positive = random.random() < self.positive_ratio
        
        if is_positive:
            # Same writer
            writer_id = random.choice(self.writer_ids)
            img1_path, img2_path = random.sample(self.writer_images[writer_id], 2)
            label = 1.0
        else:
            # Different writers
            writer1, writer2 = random.sample(self.writer_ids, 2)
            img1_path = random.choice(self.writer_images[writer1])
            img2_path = random.choice(self.writer_images[writer2])
            label = 0.0
        
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)