import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from .base_trainer import BaseTrainer
from .losses import BCELoss


class BCETrainer(BaseTrainer):
    """Trainer for Siamese networks using BCE loss."""
    
    def __init__(
        self,
        model,
        model_name: str,
        device: torch.device,
        results_dir: str = "results",
    ):
        super().__init__(model, model_name, device, results_dir)
        self.criterion = BCELoss()
        self._setup_optimizer()
    
    def _setup_optimizer(self):
        """Setup optimizer based on model size."""
        num_params = sum(p.numel() for p in self.model.parameters())
        lr = 3e-5 if num_params > 15e6 else 5e-5
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        train_loss = 0
        
        for img1, img2, labels in tqdm(train_loader, desc="Training", leave=False):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(img1, img2).squeeze()
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def _get_embeddings(self, img1, img2):
        """Get embeddings from Siamese network."""
        emb1, emb2 = self.model(img1, img2, return_embeddings=True)
        return emb1, emb2
    
    def train_kfold(
        self,
        data_root: str,
        n_splits: int = 5,
        batch_size: int = 16,
        num_workers: int = 4,
        pairs_per_writer: int = 100,
        target_size: int = 448,
        epochs: int = 50,
        patience: int = 5,
        random_state: int = 42,
    ):
        """K-Fold Cross-Validation for BCE."""
        from ..data import create_dataloaders_kfold
        
        print(f"\n{'='*70}")
        print(f"K-FOLD TRAINING: {self.model_name} ({n_splits} folds)")
        print(f"{'='*70}\n")
        
        all_fold_histories = []
        fold_summaries = []
        
        for fold in range(n_splits):
            print(f"\n{'#'*70}")
            print(f"# FOLD {fold+1}/{n_splits}")
            print(f"{'#'*70}\n")
            
            train_loader, val_loader, train_dataset, val_dataset = create_dataloaders_kfold(
                data_root=data_root,
                n_splits=n_splits,
                current_fold=fold,
                batch_size=batch_size,
                num_workers=num_workers,
                pairs_per_writer=pairs_per_writer,
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
    
    def _save_kfold_results(self, all_fold_histories, fold_summaries):
        """Save k-fold detailed and summary results."""
        combined_history = pd.concat(all_fold_histories, ignore_index=True)
        combined_history.to_csv(
            f"{self.results_dir}/{self.model_name}_kfold_detailed.csv",
            index=False
        )
        
        summary_df = pd.DataFrame(fold_summaries)
        summary_df.to_csv(
            f"{self.results_dir}/{self.model_name}_kfold_summary.csv",
            index=False
        )
    
    def _aggregate_kfold_results(self, fold_summaries):
        """Aggregate k-fold results."""
        all_eers = [f['best_eer'] for f in fold_summaries]
        
        aggregated = {
            'mean_eer': np.mean(all_eers),
            'std_eer': np.std(all_eers),
            'fold_summaries': fold_summaries
        }
        
        print(f"\n{'='*70}")
        print(f"K-FOLD RESULTS")
        print(f"{'='*70}")
        print(f"EER: {aggregated['mean_eer']:.4f} ± {aggregated['std_eer']:.4f}")
        print(f"{'='*70}\n")
        
        return aggregated
