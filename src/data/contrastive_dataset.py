# ============================================================================
# contrastive_dataset.py
# ============================================================================
"""Contrastive Dataset - Similar to Siamese but 50/50 ratio"""

import random
from typing import List, Tuple

import torch

from .base_dataset import BaseWriterDataset

class ContrastiveDataset(BaseWriterDataset):
    """Dataset for Contrastive Learning (balanced positive/negative samples)."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        samples_per_writer: int = 100,
        target_size: int = 448,
    ):
        super().__init__(
            writer_dirs=writer_dirs,
            train=train,
            samples_per_writer=samples_per_writer,
            target_size=target_size,
            min_images_per_writer=2
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a pair with label (50/50 positive/negative)."""
        is_positive = random.random() < 0.5
        
        if is_positive:
            writer_id = random.choice(self.writer_ids)
            img1_path, img2_path = random.sample(self.writer_images[writer_id], 2)
            label = 1.0
        else:
            writer1, writer2 = random.sample(self.writer_ids, 2)
            img1_path = random.choice(self.writer_images[writer1])
            img2_path = random.choice(self.writer_images[writer2])
            label = 0.0
        
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
