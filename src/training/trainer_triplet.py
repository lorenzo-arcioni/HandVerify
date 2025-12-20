"""
Triplet Loss Trainer
Trainer class for triplet networks with comprehensive biometric evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from .base_trainer import BaseTrainer
from .losses import TripletLoss


class TripletTrainer(BaseTrainer):
    """Trainer for triplet networks."""
    
    def __init__(
        self,
        model,
        model_name: str,
        device: torch.device,
        margin: float = 0.5,
        results_dir: str = "results",
    ):
        super().__init__(model, model_name, device, results_dir)
        self.margin = margin
        self.criterion = TripletLoss(margin=margin)
        self._setup_optimizer()
    
    def _setup_optimizer(self):
        """Setup optimizer for triplet learning."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        
        for anchor, positive, negative in tqdm(train_loader, desc="Training"):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            self.optimizer.zero_grad()
            
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def _get_embeddings(self, img1, img2):
        """Get embeddings from triplet network."""
        emb1 = self.model(img1)
        emb2 = self.model(img2)
        return emb1, emb2
    
    def train_kfold(
        self,
        data_root: str,
        n_splits: int = 5,
        batch_size: int = 16,
        num_workers: int = 4,
        triplets_per_writer: int = 100,
        target_size: int = 448,
        epochs: int = 50,
        patience: int = 7,
        random_state: int = 42,
    ):
        """K-Fold Cross-Validation for Triplet."""
        from ..data import create_triplet_dataloaders_kfold
        
        print(f"\n{'='*70}")
        print(f"K-FOLD TRAINING: {self.model_name} ({n_splits} folds)")
        print(f"{'='*70}\n")
        
        all_fold_histories = []
        fold_summaries = []
        
        for fold in range(n_splits):
            print(f"\n{'#'*70}")
            print(f"# FOLD {fold+1}/{n_splits}")
            print(f"{'#'*70}\n")
            
            train_loader, val_loader, train_dataset, val_dataset = create_triplet_dataloaders_kfold(
                data_root=data_root,
                n_splits=n_splits,
                current_fold=fold,
                batch_size=batch_size,
                num_workers=num_workers,
                triplets_per_writer=triplets_per_writer,
                target_size=target_size,
                random_state=random_state
            )
            
            # Reset model and optimizer
            self.model.apply(self._reset_weights)
            self._setup_optimizer()
            self.history = self._init_history()
            self.best_eer = float('inf')
            
            # Train this fold
            fold_history = self.train(
                train_loader=train_loader,
                val_dataset=val_dataset,
                epochs=epochs,
                patience=patience,
                fold=fold + 1  # Pass fold number for checkpoint naming
            )
            
            # Save fold results
            fold_df = pd.DataFrame(fold_history)
            fold_df['fold'] = fold + 1
            all_fold_histories.append(fold_df)
            
            print(f"\n✓ Fold {fold+1} - Best EER={self.best_eer:.4f}\n")
        
        # Save only detailed results (summary derivable from this)
        combined_history = pd.concat(all_fold_histories, ignore_index=True)
        combined_history.to_csv(
            f"{self.results_dir}/{self.model_name}_kfold_detailed.csv",
            index=False
        )
        
        # Compute aggregated stats
        fold_best_eers = combined_history.groupby('fold')['val_eer'].min().values
        
        aggregated = {
            'mean_eer': np.mean(fold_best_eers),
            'std_eer': np.std(fold_best_eers),
            'all_folds_eer': fold_best_eers.tolist()
        }
        
        print(f"\n{'='*70}")
        print(f"K-FOLD RESULTS")
        print(f"{'='*70}")
        print(f"EER: {aggregated['mean_eer']:.4f} ± {aggregated['std_eer']:.4f}")
        print(f"Per-fold: {fold_best_eers}")
        print(f"{'='*70}\n")
        
        return aggregated
    
    def _init_history(self):
        """Initialize empty history dict."""
        return {k: [] for k in self.history.keys()}
