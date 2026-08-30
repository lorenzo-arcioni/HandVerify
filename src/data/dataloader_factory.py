# ============================================================================
# dataloader_factory.py
# ============================================================================

import os
import numpy as np
import random
import torch

from .siamese_dataset import SiameseDataset
from .triplet_dataset import TripletDataset
from .contrastive_dataset import ContrastiveDataset

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold

"""
DataLoader Factory
Unified functions to create dataloaders for any dataset type.

Split a 3 vie (train / val / test) sui writer:
  - train: usato per il training vero e proprio
  - val:   usato per l'early stopping / model selection durante il training
  - test:  held-out, mai visto durante training o model selection.
           Va usato SOLO per il report finale delle metriche, per evitare
           che lo stesso set che sceglie il checkpoint migliore sia anche
           quello su cui si dichiarano i risultati (bias di selezione).
"""


def _list_writer_dirs(data_root: str):
    return [
        os.path.join(data_root, d)
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
    ]


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(dataset, batch_size, shuffle, num_workers, random_state, drop_last=False):
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        generator=torch.Generator().manual_seed(random_state),
        pin_memory=True,
        drop_last=drop_last,
    )


def create_dataloaders(
    dataset_class,
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    val_size: float = 0.15,
    test_size: float = 0.15,
    target_size: int = 448,
    random_state: int = 42,
    **dataset_kwargs
):
    """
    Split a 3 vie (train/val/test) sui writer, stesso dominio dati.

    Args:
        dataset_class: Dataset class da istanziare (SiameseDataset, etc.)
        data_root: Directory radice con le sottocartelle per writer
        batch_size: Batch size
        num_workers: Numero di worker
        val_size: Frazione di writer riservata a validation (early stopping)
        test_size: Frazione di writer riservata a test (held-out, report finale)
        target_size: Dimensione immagine
        random_state: Seed
        **dataset_kwargs: Argomenti aggiuntivi per il costruttore del dataset

    Returns:
        (train_loader, val_loader, test_loader,
         train_dataset, val_dataset, test_dataset)

        Un loader/dataset è None se la relativa frazione è 0.0.
    """
    if val_size < 0 or test_size < 0 or (val_size + test_size) >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) deve essere < 1.0 "
            f"per lasciare una porzione di dati al training."
        )

    writer_dirs = _list_writer_dirs(data_root)

    holdout_size = val_size + test_size
    if holdout_size == 0.0:
        train_dirs, val_dirs, test_dirs = writer_dirs, [], []
    else:
        train_dirs, temp_dirs = train_test_split(
            writer_dirs, test_size=holdout_size, random_state=random_state
        )
        if val_size == 0.0:
            val_dirs, test_dirs = [], temp_dirs
        elif test_size == 0.0:
            val_dirs, test_dirs = temp_dirs, []
        else:
            rel_test = test_size / holdout_size
            val_dirs, test_dirs = train_test_split(
                temp_dirs, test_size=rel_test, random_state=random_state
            )

    train_dataset = dataset_class(
        train_dirs, train=True, target_size=target_size, **dataset_kwargs
    ) if train_dirs else None

    val_dataset = dataset_class(
        val_dirs, train=False, target_size=target_size, **dataset_kwargs
    ) if val_dirs else None

    test_dataset = dataset_class(
        test_dirs, train=False, target_size=target_size, **dataset_kwargs
    ) if test_dirs else None

    train_loader = _make_loader(train_dataset, batch_size, True, num_workers,
                                 random_state, drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, False, num_workers, random_state)
    test_loader = _make_loader(test_dataset, batch_size, False, num_workers, random_state)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_cross_dataset_dataloaders(
    dataset_class,
    train_root: str,
    target_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    val_size: float = 0.5,
    target_size: int = 448,
    random_state: int = 42,
    **dataset_kwargs
):
    """
    Split per esperimenti cross-dataset: tutti i writer di train_root vanno
    in training; i writer di target_root vengono divisi in val (early
    stopping) e test (held-out, report finale), così il test set del
    dominio target resta comunque mai visto prima della valutazione finale.

    Args:
        dataset_class: Dataset class da istanziare
        train_root: Directory radice del dominio di training (usato per intero)
        target_root: Directory radice del dominio target, diviso in val/test
        batch_size: Batch size
        num_workers: Numero di worker
        val_size: Frazione dei writer di target_root riservata a val
                  (il resto, 1 - val_size, va a test)
        target_size: Dimensione immagine
        random_state: Seed
        **dataset_kwargs: Argomenti aggiuntivi per il costruttore del dataset

    Returns:
        (train_loader, val_loader, test_loader,
         train_dataset, val_dataset, test_dataset)
    """
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size deve essere strettamente tra 0 e 1, ricevuto {val_size}")

    train_dirs = _list_writer_dirs(train_root)
    target_dirs = _list_writer_dirs(target_root)

    val_dirs, test_dirs = train_test_split(
        target_dirs, test_size=(1.0 - val_size), random_state=random_state
    )

    train_dataset = dataset_class(
        train_dirs, train=True, target_size=target_size, **dataset_kwargs
    )
    val_dataset = dataset_class(
        val_dirs, train=False, target_size=target_size, **dataset_kwargs
    )
    test_dataset = dataset_class(
        test_dirs, train=False, target_size=target_size, **dataset_kwargs
    )

    train_loader = _make_loader(train_dataset, batch_size, True, num_workers,
                                 random_state, drop_last=True)
    val_loader = _make_loader(val_dataset, batch_size, False, num_workers, random_state)
    test_loader = _make_loader(test_dataset, batch_size, False, num_workers, random_state)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_kfold_dataloaders(
    dataset_class,
    data_root: str,
    n_splits: int = 5,
    current_fold: int = 0,
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: int = 448,
    random_state: int = 42,
    **dataset_kwargs
):
    """
    Generic K-Fold dataloader creator.

    Args:
        dataset_class: Dataset class to instantiate
        data_root: Root directory with writer subdirectories
        n_splits: Number of folds
        current_fold: Current fold index (0 to n_splits-1)
        batch_size: Batch size
        num_workers: Number of workers
        target_size: Image size
        random_state: Random seed
        **dataset_kwargs: Additional args for dataset constructor

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Get writer directories
    writer_dirs = _list_writer_dirs(data_root)

    # K-Fold split
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kfold.split(writer_dirs))
    train_idx, val_idx = splits[current_fold]

    train_dirs = [writer_dirs[i] for i in train_idx]
    val_dirs = [writer_dirs[i] for i in val_idx]

    # Create datasets
    train_dataset = dataset_class(
        train_dirs,
        train=True,
        target_size=target_size,
        **dataset_kwargs
    )

    val_dataset = dataset_class(
        val_dirs,
        train=False,
        target_size=target_size,
        **dataset_kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset
