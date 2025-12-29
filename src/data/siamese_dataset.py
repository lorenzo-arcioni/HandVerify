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
        resample_negatives_every_n_epochs: int = 1,
    ):
        print(f"\n{'='*60}")
        print(f"Initializing {'TRAIN' if train else 'VAL'} Siamese Dataset")
        print(f"{'='*60}")
        print(f"  Writers: {len(writer_dirs)}")
        print(f"  Positive ratio: {positive_ratio:.2f}")
        print(f"  Resample negatives every: {resample_negatives_every_n_epochs} epoch(s)")
        
        super().__init__(
            writer_dirs=writer_dirs,
            train=train,
            target_size=target_size,
            positive_ratio=positive_ratio,
            resample_negatives_every_n_epochs=resample_negatives_every_n_epochs
        )
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.samples[idx]
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        return img1, img2, torch.tensor(label, dtype=torch.float32)
