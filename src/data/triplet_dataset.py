# ============================================================================
# triplet_dataset.py
# ============================================================================
from typing import List, Tuple
import torch
from .base_dataset import BaseWriterDataset

class TripletDataset(BaseWriterDataset):
    """Dataset for Triplet network training."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        target_size: int = 448,
    ):
        super().__init__(writer_dirs=writer_dirs, train=train, target_size=target_size)
        
        self.triplets = self._generate_all_samples(triplet=True)
        self.num_samples = len(self.triplets)
        
        print(f"{'TRAIN' if train else 'VAL'}: {len(self.triplets)} triplets")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, positive_path, negative_path = self.triplets[idx]
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        return anchor, positive, negative
    
    def __len__(self) -> int:
        return len(self.triplets)
