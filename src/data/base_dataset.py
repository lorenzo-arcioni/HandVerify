"""
Base Dataset Classes
Provides common functionality for all dataset types.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from itertools import combinations, product

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_train_transforms, get_test_transforms


class BaseWriterDataset(Dataset, ABC):
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        target_size: int = 448,
    ):
        self.writer_dirs = writer_dirs
        self.train = train
        self.target_size = target_size
        
        self.transform = (get_train_transforms(target_size) if train 
                         else get_test_transforms(target_size))
        
        self.writer_images = self._load_writer_images()
        self.writer_ids = list(self.writer_images.keys())
    
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
            writer_images[writer_id] = images
            
        return writer_images
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and transform a single image."""
        img = Image.open(path).convert("L")
        return self.transform(img)
    
    def _generate_all_samples(
        self, 
        positive_ratio: float = 0.5, 
        triplet: bool = False
    ) -> List[Union[Tuple[str, str, float], Tuple[str, str, str]]]:
        """
        Genera tutte le coppie o triple possibili.
        
        Args:
            positive_ratio: Rapporto di coppie positive (ignorato se triplet=True)
            triplet: Se True genera triple (anchor, positive, negative)
                    Se False genera coppie (img1, img2, label)
        
        Returns:
            Lista di tuple (coppie con label o triple)
        """
        multi_image_writers = [w for w in self.writer_ids 
                              if len(self.writer_images[w]) >= 2]
        
        print(f"\nGenerating {'triplets' if triplet else 'pairs'} from "
              f"{len(multi_image_writers)} multi-image writers...")
        
        # Genera tutte le coppie positive (stesso writer)
        positive_pairs = []
        for writer_id in multi_image_writers:
            images = self.writer_images[writer_id]
            for img1, img2 in combinations(images, 2):
                positive_pairs.append((writer_id, img1, img2))
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        
        if triplet:
            # Genera triple: per ogni coppia positiva, aggiungi tutti i negativi possibili
            triplets = []
            for writer_id, anchor, positive in positive_pairs:
                for other_writer_id in self.writer_ids:
                    if other_writer_id != writer_id:
                        for negative in self.writer_images[other_writer_id]:
                            triplets.append((anchor, positive, negative))
            
            print(f"Generated {len(triplets)} triplets")
            return triplets
        
        else:
            # Genera coppie con label
            pairs = [(img1, img2, 1.0) for _, img1, img2 in positive_pairs]
            
            # Calcola quante negative servono
            num_negative_needed = int(len(pairs) * (1 - positive_ratio) / positive_ratio)
            
            # Genera coppie negative
            negative_pairs = []
            for w1, w2 in combinations(self.writer_ids, 2):
                if len(negative_pairs) >= num_negative_needed:
                    break
                for img1, img2 in product(self.writer_images[w1], self.writer_images[w2]):
                    if len(negative_pairs) >= num_negative_needed:
                        break
                    negative_pairs.append((img1, img2, 0.0))
            
            print(f"Generated {len(negative_pairs)} negative pairs")
            
            all_pairs = pairs + negative_pairs
            random.shuffle(all_pairs)
            return all_pairs
    
    def __len__(self) -> int:
        return self.num_samples
    
    @abstractmethod
    def __getitem__(self, idx: int):
        """Generate a sample (must be implemented by subclasses)."""
        pass