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
            weight_decay=1e-2,
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
    
    @torch.no_grad()
    def validate_loss(self, val_loader: DataLoader) -> float:
        """Calculate Triplet validation loss."""
        self.model.eval()
        val_loss = 0.0

        for anchor, positive, negative in tqdm(val_loader, desc="Validation"):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Get embeddings
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            negative_emb = self.model(negative)
            
            # Triplet loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            val_loss += loss.item()
        
        return val_loss / len(val_loader)

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
        target_size: int = 448,
        resample_negatives_every_n_epochs: int = 1,
        epochs: int = 50,
        patience: int = 7,
        random_state: int = 42,
    ):
        """K-Fold Cross-Validation for Triplet."""
        from ..data import create_triplet_kfold_dataloaders
        
        print(f"\n{'='*70}")
        print(f"K-FOLD TRAINING: {self.model_name} ({n_splits} folds)")
        print(f"{'='*70}\n")
        
        all_fold_histories = []
        all_fold_metrics = []
        
        for fold in range(n_splits):
            print(f"\n{'#'*70}")
            print(f"# FOLD {fold+1}/{n_splits}")
            print(f"{'#'*70}\n")
            
            train_loader, val_loader, train_dataset, val_dataset = create_triplet_kfold_dataloaders(
                data_root=data_root,
                n_splits=n_splits,
                current_fold=fold,
                batch_size=batch_size,
                num_workers=num_workers,
                target_size=target_size,
                resample_negatives_every_n_epochs=resample_negatives_every_n_epochs,
                random_state=random_state
            )
            
            # Reset model and optimizer
            self.model.apply(self._reset_weights)
            self._setup_optimizer()
            self.history = self._init_history()
            self.best_loss = float('inf')
            
            # Train this fold
            fold_history, fold_metrics = self.train(
                train_loader=train_loader,
                val_loader=val_loader,
                val_dataset=val_dataset,
                epochs=epochs,
                patience=patience,
                fold=fold + 1
            )
            
            # Save fold results
            fold_df = pd.DataFrame(fold_history)
            fold_df['fold'] = fold + 1
            all_fold_histories.append(fold_df)
            
            # Save fold metrics
            fold_metrics['fold'] = fold + 1
            all_fold_metrics.append(fold_metrics)
            
            print(f"\n✓ Fold {fold+1} - Best Val Loss={self.best_loss:.4f}, Final EER={fold_metrics['eer']:.4f}\n")
        
        # Save detailed results
        combined_history = pd.concat(all_fold_histories, ignore_index=True)
        combined_history.to_csv(
            f"{self.results_dir}/{self.model_name}_kfold_detailed.csv",
            index=False
        )
        
        # Compute aggregated stats
        fold_eers = [m['eer'] for m in all_fold_metrics]
        
        aggregated = {
            'mean_eer': np.mean(fold_eers),
            'std_eer': np.std(fold_eers),
            'all_folds_eer': fold_eers
        }
        
        print(f"\n{'='*70}")
        print(f"K-FOLD RESULTS")
        print(f"{'='*70}")
        print(f"EER: {aggregated['mean_eer']:.4f} ± {aggregated['std_eer']:.4f}")
        print(f"Per-fold: {fold_eers}")
        print(f"{'='*70}\n")
        
        return aggregated