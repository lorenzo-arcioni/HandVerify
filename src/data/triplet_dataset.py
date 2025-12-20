# ============================================================================
# triplet_dataset.py
# ============================================================================

import random
from typing import List, Tuple

import torch

from .base_dataset import BaseWriterDataset

"""Triplet Dataset - Anchor, Positive, Negative"""

class TripletDataset(BaseWriterDataset):
    """Dataset for Triplet network training."""
    
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
        """Generate a triplet (anchor, positive, negative)."""
        # Anchor writer
        anchor_writer = random.choice(self.writer_ids)
        
        # Anchor and positive from same writer
        anchor_path, positive_path = random.sample(
            self.writer_images[anchor_writer], 2
        )
        
        # Negative from different writer
        negative_writer = random.choice(
            [w for w in self.writer_ids if w != anchor_writer]
        )
        negative_path = random.choice(self.writer_images[negative_writer])
        
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative
