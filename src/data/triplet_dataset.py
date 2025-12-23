# ============================================================================
# triplet_dataset.py
# ============================================================================
import random
from itertools import combinations
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
        target_size: int = 448,
    ):
        super().__init__(
            writer_dirs=writer_dirs,
            train=train,
            target_size=target_size,
        )
        
        # Pre-genera tutte le triple
        self.triplets = self._generate_all_triplets()
        self.num_samples = len(self.triplets)
        
        print(f"{'TRAIN' if train else 'VAL'}: Generated {len(self.triplets)} triplets")
        print(f"  - Writers: {len(self.writer_ids)} total")
        print(f"    - With ≥2 images: {sum(1 for w in self.writer_ids if len(self.writer_images[w]) >= 2)}")
        print(f"    - With 1 image: {sum(1 for w in self.writer_ids if len(self.writer_images[w]) == 1)}")
    
    def _generate_all_triplets(self) -> List[Tuple[str, str, str]]:
        """
        Genera tutte le triple possibili.
        Per ogni coppia positiva (anchor, positive), genera triple con tutti i possibili negativi.
        
        Returns:
            Lista di tuple (anchor_path, positive_path, negative_path)
        """
        triplets = []
        triplets_set = set()
        
        # Separa autori per numero di immagini
        multi_image_writers = [w for w in self.writer_ids 
                              if len(self.writer_images[w]) >= 2]
        single_image_writers = [w for w in self.writer_ids 
                                if len(self.writer_images[w]) == 1]
        
        print(f"\nGenerating triplets from {len(multi_image_writers)} multi-image writers "
              f"and {len(single_image_writers)} single-image writers...")
        
        # ===================================================================
        # STEP 1: Per ogni writer con ≥2 immagini, genera coppie positive
        # ===================================================================
        positive_pairs = []
        for writer_id in multi_image_writers:
            images = self.writer_images[writer_id]
            for i in range(len(images) - 1):
                for j in range(i + 1, len(images)):
                    positive_pairs.append((writer_id, images[i], images[j]))
        
        print(f"Generated {len(positive_pairs)} positive pairs (anchor-positive)")
        
        # ===================================================================
        # STEP 2: Per ogni coppia positiva, genera triple con negativi
        # ===================================================================
        # Conta i negativi possibili
        total_triplets_possible = 0
        for writer_id, anchor, positive in positive_pairs:
            for other_writer_id in self.writer_ids:
                if other_writer_id != writer_id:
                    total_triplets_possible += len(self.writer_images[other_writer_id])
        
        print(f"Total possible triplets: {total_triplets_possible}")
        
        # Genera tutte le triple
        for writer_id, anchor, positive in positive_pairs:
            for other_writer_id in self.writer_ids:
                if other_writer_id == writer_id:
                    continue
                
                negative_images = self.writer_images[other_writer_id]
                
                for negative in negative_images:
                    # Crea chiave per evitare duplicati
                    triplet_key = (anchor, positive, negative)
                    
                    if triplet_key not in triplets_set:
                        triplets_set.add(triplet_key)
                        triplets.append((anchor, positive, negative))
        
        print(f"Generated {len(triplets)} unique triplets")
        
        return triplets
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Carica e restituisce una tripla pre-generata."""
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative
    
    def __len__(self) -> int:
        return len(self.triplets)