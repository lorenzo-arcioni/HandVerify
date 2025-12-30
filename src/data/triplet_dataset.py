# ============================================================================
# triplet_dataset.py
# ============================================================================
from typing import List, Tuple
import torch
import random
from itertools import combinations


class TripletDataset:
    """Dataset for Triplet network training."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        target_size: int = 448,
        resample_negatives_every_n_epochs: int = 1,
        positive_ratio: float = 0.5,  # For validation pairs
    ):
        print(f"\n{'='*60}")
        print(f"Initializing {'TRAIN' if train else 'VAL'} Triplet Dataset")
        print(f"{'='*60}")
        print(f"  Writers: {len(writer_dirs)}")
        print(f"  Resample negatives every: {resample_negatives_every_n_epochs} epoch(s)")

        self.resample_negatives_every_n_epochs = resample_negatives_every_n_epochs
        self.writer_dirs = writer_dirs
        self.train = train
        self.target_size = target_size
        self.positive_ratio = positive_ratio
        self.current_epoch = 0
        
        from .transforms import get_train_transforms, get_test_transforms
        self.transform = (get_train_transforms(target_size) if train 
                         else get_test_transforms(target_size))
        
        self.writer_images = self._load_writer_images()
        self.writer_ids = list(self.writer_images.keys())
        
        # Genera tutte le genuine pairs (base per le triple E per validation)
        self.all_genuine_pairs = self._generate_all_genuine_pairs()
        print(f"  Generated {len(self.all_genuine_pairs)} genuine pairs (base for triplets)")
        
        # Pool di immagini negative per writer
        self._create_negative_pool()
        
        # Genera le triple iniziali per training
        self._resample_triplets()
        
        # Crea validation pairs (FIXED, non cambiano ad ogni epoca)
        self._create_validation_pairs()
    
    def _load_writer_images(self) -> dict:
        """Load image paths for each writer."""
        from pathlib import Path
        import os
        
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
        from PIL import Image
        img = Image.open(path).convert("L")
        return self.transform(img)
    
    def _generate_all_genuine_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Genera TUTTE le coppie genuine possibili (stesso writer).
        Returns: Lista di (writer_id, img1_path, img2_path)
        """
        multi_image_writers = [w for w in self.writer_ids 
                              if len(self.writer_images[w]) >= 2]
        
        genuine_pairs = []
        for writer_id in multi_image_writers:
            images = self.writer_images[writer_id]
            for img1, img2 in combinations(images, 2):
                genuine_pairs.append((writer_id, img1, img2))
        
        return genuine_pairs
    
    def _create_negative_pool(self):
        """Crea un pool di immagini negative per ogni writer."""
        self.negative_pool = {
            writer_id: [
                img
                for other_id, images in self.writer_images.items()
                if other_id != writer_id
                for img in images
            ]
            for writer_id in self.writer_ids
        }
    
    def _resample_triplets(self):
        """Genera le triple campionando casualmente i negativi dal pool."""
        self.triplets = []
        
        for writer_id, anchor_path, positive_path in self.all_genuine_pairs:
            # Campiona un'immagine negativa casuale da un altro writer
            negative_path = random.choice(self.negative_pool[writer_id])
            self.triplets.append((anchor_path, positive_path, negative_path))
        
        # Shuffle per randomizzare l'ordine
        random.shuffle(self.triplets)
        
        print(f"  Generated {len(self.triplets)} triplets")
    
    def _create_validation_pairs(self):
        """
        Crea pairs (img1, img2, label) per validation comprehensive.
        Usa tutte le genuine + campiona impostors per mantenere positive_ratio.
        """
        # Genuine pairs (tutte)
        genuine_samples = [(img1, img2, 1.0) for _, img1, img2 in self.all_genuine_pairs]
        
        # Genera tutte le possibili impostor pairs
        all_impostor_pairs = []
        for w1, w2 in combinations(self.writer_ids, 2):
            for img1 in self.writer_images[w1]:
                for img2 in self.writer_images[w2]:
                    all_impostor_pairs.append((img1, img2))
        
        # Calcola quanti impostors servono per mantenere positive_ratio
        num_impostors_needed = int(
            len(genuine_samples) * (1 - self.positive_ratio) / self.positive_ratio
        )
        
        # Campiona impostors
        if len(all_impostor_pairs) <= num_impostors_needed:
            impostor_samples = [(img1, img2, 0.0) for img1, img2 in all_impostor_pairs]
        else:
            sampled_impostors = random.sample(all_impostor_pairs, num_impostors_needed)
            impostor_samples = [(img1, img2, 0.0) for img1, img2 in sampled_impostors]
        
        # Combina e shuffla
        self.validation_pairs = genuine_samples + impostor_samples
        random.shuffle(self.validation_pairs)
        
        print(f"  Created {len(self.validation_pairs)} validation pairs "
              f"({len(genuine_samples)} genuine + {len(impostor_samples)} impostor)")
    
    def on_epoch_end(self, epoch: int):
        """
        Chiamata alla fine di ogni epoca dal trainer.
        Ricampiona i negativi se necessario.
        
        Args:
            epoch: Numero dell'epoca appena completata (0-indexed)
        """
        if self.resample_negatives_every_n_epochs == 0:
            # Disabilitato
            return
        
        if (epoch + 1) % self.resample_negatives_every_n_epochs == 0:
            print(f"\n🔄 Resampling triplet negatives for next epoch...")
            self._resample_triplets()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a triplet (anchor, positive, negative) for training."""
        anchor_path, positive_path, negative_path = self.triplets[idx]
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        return anchor, positive, negative
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def get_validation_pair(self, idx: int) -> Tuple[str, str, float]:
        """
        Get a validation pair for comprehensive evaluation.
        
        For TripletDataset, this returns pairs from validation_pairs which are
        created once during initialization and remain fixed.
        
        Args:
            idx: Index of the validation pair
            
        Returns:
            Tuple of (img1_path, img2_path, label)
        """
        return self.validation_pairs[idx]