# ============================================================================
# siamese_dataset.py
# ============================================================================
from typing import List, Tuple
import torch
from .base_dataset import BaseWriterDataset

class SiameseDataset(BaseWriterDataset):
    """Dataset for Siamese network training (positive/negative pairs)."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        positive_ratio: float = 0.5,
        target_size: int = 448,
    ):
        self.positive_ratio = positive_ratio
        super().__init__(writer_dirs=writer_dirs, train=train, target_size=target_size)
        
        self.pairs = self._generate_all_samples(positive_ratio=self.positive_ratio, triplet=False)
        self.num_samples = len(self.pairs)
        
        num_positive = sum(1 for p in self.pairs if p[2] == 1.0)
        print(f"{'TRAIN' if train else 'VAL'}: {len(self.pairs)} pairs "
              f"({num_positive} positive, {len(self.pairs)-num_positive} negative)")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.pairs[idx]
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        return img1, img2, torch.tensor(label, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.pairs)