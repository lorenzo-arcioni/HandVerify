# ============================================================================
# contrastive_dataset.py
# ============================================================================
"""Contrastive Dataset - Similar to Siamese but 50/50 ratio"""
import random
from itertools import combinations
from typing import List, Tuple
import torch
from .base_dataset import BaseWriterDataset

class ContrastiveDataset(BaseWriterDataset):
    """Dataset for Contrastive Learning (balanced positive/negative samples)."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        target_size: int = 448,
    ):
        super().__init__(
            writer_dirs=writer_dirs,
            train=train,
            target_size=target_size,
        )
        
        # Pre-genera tutte le coppie con ratio 50/50
        self.pairs = self._generate_all_possible_pairs()
        self.num_samples = len(self.pairs)
        
        num_positive = sum(1 for p in self.pairs if p[2] == 1.0)
        num_negative = sum(1 for p in self.pairs if p[2] == 0.0)
        actual_ratio = num_positive / len(self.pairs) * 100 if self.pairs else 0
        
        print(f"{'TRAIN' if train else 'VAL'}: Generated {len(self.pairs)} pairs")
        print(f"  - Positive: {num_positive} ({actual_ratio:.1f}%)")
        print(f"  - Negative: {num_negative} ({100-actual_ratio:.1f}%)")
        print(f"  - Writers: {len(self.writer_ids)} total")
        print(f"    - With ≥2 images: {sum(1 for w in self.writer_ids if len(self.writer_images[w]) >= 2)}")
        print(f"    - With 1 image: {sum(1 for w in self.writer_ids if len(self.writer_images[w]) == 1)}")
    
    def _generate_all_possible_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Genera tutte le coppie possibili con ratio 50/50.
        
        Returns:
            Lista di tuple (img1_path, img2_path, label)
        """
        positive_pairs = []
        negative_pairs = []
        negative_pairs_set = set()
        
        # Separa autori per numero di immagini
        multi_image_writers = [w for w in self.writer_ids 
                              if len(self.writer_images[w]) >= 2]
        single_image_writers = [w for w in self.writer_ids 
                                if len(self.writer_images[w]) == 1]
        
        print(f"\nGenerating pairs from {len(multi_image_writers)} multi-image writers "
              f"and {len(single_image_writers)} single-image writers...")
        
        # ===================================================================
        # STEP 1: Genera TUTTE le coppie positive possibili
        # ===================================================================
        for writer_id in multi_image_writers:
            images = self.writer_images[writer_id]
            
            for i in range(len(images) - 1):
                for j in range(i + 1, len(images)):
                    positive_pairs.append((images[i], images[j], 1.0))
        
        print(f"Generated {len(positive_pairs)} positive pairs (all possible combinations)")
        
        # ===================================================================
        # STEP 2: Genera STESSO NUMERO di coppie negative (ratio 50/50)
        # ===================================================================
        num_negative_needed = len(positive_pairs)
        
        # Calcola il numero totale di coppie negative possibili
        total_negative_possible = 0
        for writer1_id, writer2_id in combinations(self.writer_ids, 2):
            writer1_images = self.writer_images[writer1_id]
            writer2_images = self.writer_images[writer2_id]
            total_negative_possible += len(writer1_images) * len(writer2_images)
        
        print(f"Need {num_negative_needed} negative pairs for 50% positive ratio")
        print(f"Total possible negative pairs: {total_negative_possible}")
        
        if total_negative_possible > 0:
            print(f"Using {num_negative_needed / total_negative_possible * 100:.2f}% of possible negative pairs")
        
        # ===================================================================
        # STEP 3: Genera coppie negative
        # ===================================================================
        for writer1_id, writer2_id in combinations(self.writer_ids, 2):
            if len(negative_pairs) >= num_negative_needed:
                break
            
            writer1_images = self.writer_images[writer1_id]
            writer2_images = self.writer_images[writer2_id]
            
            for img1 in writer1_images:
                for img2 in writer2_images:
                    if len(negative_pairs) >= num_negative_needed:
                        break
                    
                    pair_key = tuple(sorted([img1, img2]))
                    if pair_key not in negative_pairs_set:
                        negative_pairs_set.add(pair_key)
                        negative_pairs.append((img1, img2, 0.0))
                
                if len(negative_pairs) >= num_negative_needed:
                    break
        
        print(f"Generated {len(negative_pairs)} negative pairs")
        
        # ===================================================================
        # STEP 4: Combina e mescola
        # ===================================================================
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        return all_pairs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Carica e restituisce una coppia pre-generata."""
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.pairs)