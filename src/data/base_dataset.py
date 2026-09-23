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
        positive_ratio: float = 0.5,
        resample_negatives_every_n_epochs: int = 1,
        max_genuine_pairs: int = None,
        random_seed: int = 42
    ):
        """
        Args:
            writer_dirs: Lista di directory degli writers
            train: Se True, applica augmentation
            target_size: Dimensione target delle immagini
            positive_ratio: Rapporto genuine/(genuine+impostors)
            resample_negatives_every_n_epochs: Ogni quante epoche ricampionare i negativi
                                               (0 = mai, 1 = ogni epoca, N = ogni N epoche)
            max_genuine_pairs: Numero massimo di coppie genuine usate per epoca.
                               None = usa tutte le genuine disponibili (comportamento
                               precedente). Se impostato e minore del totale disponibile,
                               ad ogni epoca viene ricampionato un sottoinsieme random di
                               questa dimensione, con lo stesso meccanismo di resampling
                               già usato per gli impostors.
            random_seed: Seed base per il resampling deterministico per epoca
        """
        self.writer_dirs = writer_dirs
        self.train = train
        self.target_size = target_size
        self.positive_ratio = positive_ratio
        self.resample_negatives_every_n_epochs = resample_negatives_every_n_epochs
        self.max_genuine_pairs = max_genuine_pairs
        self.random_seed = random_seed

        self.transform = (get_train_transforms(target_size) if train
                         else get_test_transforms(target_size))

        self.writer_images = self._load_writer_images()
        self.writer_ids = sorted(list(self.writer_images.keys()))

        # Pool FISSO di tutte le genuine e tutte le impostor possibili
        self.all_genuine_pairs = self._generate_all_genuine_pairs()
        self.all_impostor_pairs = self._generate_all_impostor_pairs()

        print(f"  Generated {len(self.all_genuine_pairs)} genuine pairs (pool)")
        print(f"  Generated {len(self.all_impostor_pairs)} total impostor pairs (pool)")

        # NOTA: num_impostors_needed NON viene piu' calcolato qui una volta
        # per tutte, perche' dipende da quante genuine vengono campionate
        # ad OGNI epoca (vedi _resample_pairs). Il calcolo e' stato spostato
        # dentro _resample_pairs.

        self.current_genuine_pairs = []
        self.current_impostor_pairs = []
        self.current_epoch = 0
        self._resample_pairs()

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

    def _generate_all_impostor_pairs(self) -> List[Tuple[str, str]]:
        """
        Genera TUTTE le coppie impostors possibili (writers diversi).
        Returns: Lista di (img1_path, img2_path)
        """
        impostor_pairs = []

        for w1, w2 in combinations(self.writer_ids, 2):
            for img1 in self.writer_images[w1]:
                for img2 in self.writer_images[w2]:
                    impostor_pairs.append((img1, img2))

        return impostor_pairs

    def _resample_pairs(self, epoch: int = 0):
        """
        Ricampiona sia le genuine (se max_genuine_pairs e' impostato) sia gli
        impostors dal rispettivo pool completo, mantenendo il positive_ratio
        desiderato rispetto al numero di genuine EFFETTIVAMENTE usate in
        questa epoca (non rispetto al pool totale di genuine).
        """
        epoch_rng = random.Random(self.random_seed + epoch)

        # --- Genuine ---
        if self.max_genuine_pairs is not None and \
                len(self.all_genuine_pairs) > self.max_genuine_pairs:
            self.current_genuine_pairs = epoch_rng.sample(
                self.all_genuine_pairs, self.max_genuine_pairs
            )
        else:
            self.current_genuine_pairs = self.all_genuine_pairs.copy()

        # --- Impostors (calcolati sulle genuine di QUESTA epoca) ---
        num_impostors_needed = int(
            len(self.current_genuine_pairs) * (1 - self.positive_ratio) / self.positive_ratio
        )

        if len(self.all_impostor_pairs) <= num_impostors_needed:
            # Se abbiamo meno impostors disponibili di quelli necessari, usali tutti
            self.current_impostor_pairs = self.all_impostor_pairs.copy()
        else:
            self.current_impostor_pairs = epoch_rng.sample(
                self.all_impostor_pairs,
                num_impostors_needed
            )

        print(f"  Sampled {len(self.current_genuine_pairs)} genuine "
              f"(from pool of {len(self.all_genuine_pairs)}) + "
              f"{len(self.current_impostor_pairs)} impostors "
              f"(from pool of {len(self.all_impostor_pairs)})")

        # Crea il dataset completo per questa epoca
        self._create_current_samples()

    def _create_current_samples(self):
        """
        Crea il dataset attuale combinando le genuine e gli impostors
        campionati per l'epoca corrente.
        """
        genuine_samples = [(img1, img2, 1.0) for _, img1, img2 in self.current_genuine_pairs]
        impostor_samples = [(img1, img2, 0.0) for img1, img2 in self.current_impostor_pairs]

        self.samples = genuine_samples + impostor_samples
        random.shuffle(self.samples)

        actual_ratio = len(genuine_samples) / len(self.samples) * 100
        print(f"  Created epoch dataset: {len(genuine_samples)} genuine + {len(impostor_samples)} impostors")
        print(f"  Actual positive ratio: {actual_ratio:.1f}%")

    def on_epoch_end(self, epoch: int):
        """
        Chiamata alla fine di ogni epoca dal trainer.
        Ricampiona genuine (se cappate) e negativi se necessario.

        Args:
            epoch: Numero dell'epoca appena completata (0-indexed)
        """
        if self.resample_negatives_every_n_epochs == 0:
            # Disabilitato
            return

        if (epoch + 1) % self.resample_negatives_every_n_epochs == 0:
            print(f"\n🔄 Resampling genuine + negatives for next epoch...")
            self._resample_pairs(epoch + 1)

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int):
        """Generate a sample (must be implemented by subclasses)."""
        pass

    @abstractmethod
    def get_validation_pair(self, idx: int) -> Tuple[str, str, float]:
        """
        Get a validation pair for comprehensive evaluation.

        This method provides a consistent interface for validation across all dataset types.
        Returns pairs in format (img1_path, img2_path, label) where label is 1.0 for
        genuine pairs and 0.0 for impostor pairs.

        Args:
            idx: Index of the pair to retrieve

        Returns:
            Tuple of (img1_path, img2_path, label)
        """
        pass