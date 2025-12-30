# ============================================================================
# triplet_dataset.py
# ============================================================================
from typing import List, Tuple
import torch
import random
from .base_dataset import BaseWriterDataset


class TripletDataset(BaseWriterDataset):
    """Dataset for Triplet network training."""
    
    def __init__(
        self,
        writer_dirs: List[str],
        train: bool = True,
        target_size: int = 448,
        resample_negatives_every_n_epochs: int = 1,
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
        self.current_epoch = 0
        
        from .transforms import get_train_transforms, get_test_transforms
        self.transform = (get_train_transforms(target_size) if train 
                         else get_test_transforms(target_size))
        
        self.writer_images = self._load_writer_images()
        self.writer_ids = list(self.writer_images.keys())
        
        # Genera tutte le genuine pairs (base per le triple)
        self.all_genuine_pairs = self._generate_all_genuine_pairs()
        print(f"  Generated {len(self.all_genuine_pairs)} genuine pairs (base for triplets)")
        
        # Pool di immagini negative per writer
        self._create_negative_pool()
        
        # Genera le triple iniziali
        self._resample_triplets()
    
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
        anchor_path, positive_path, negative_path = self.triplets[idx]
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        return anchor, positive, negative
    
    def __len__(self) -> int:
        return len(self.triplets)