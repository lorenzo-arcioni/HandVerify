# src/training/base_trainer.py
"""
Base Trainer Class
Common functionality for all trainer types.
"""

import os
import gc
import random
from typing import Dict, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics, print_results


class BaseTrainer(ABC):
    """Base class for all trainers with common validation and saving logic."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        device: torch.device,
        results_dir: str = "results",
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Training history (only loss per epoch)
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
        }
        
        self.best_loss = float('inf')
    
    @abstractmethod
    def _setup_optimizer(self):
        """Setup optimizer and scheduler (implemented by subclasses)."""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader):
        """Train for one epoch (implemented by subclasses)."""
        pass
    
    @abstractmethod
    def validate_loss(self, val_loader) -> float:
        """
        Calculate validation loss (implemented by subclasses).
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Average validation loss
        """
        pass
    
    @torch.no_grad()
    def validate_comprehensive(self, val_dataset) -> Dict[str, float]:
        """
        Comprehensive validation computing all verification metrics.
        Uses validation pairs from val_dataset.get_validation_pair().

        If the subclass implements _get_embeddings, it's used for computing
        similarity metrics (contrastive/triplet). Otherwise, assumes BCE
        Siamese and uses model output directly.
        
        Args:
            val_dataset: Validation dataset with get_validation_pair() method
        
        Returns:
            Dictionary with all metrics
        """
        self.model.eval()
        
        genuine_scores = []
        impostor_scores = []

        # Check if subclass has implemented _get_embeddings
        use_embeddings = hasattr(self, "_get_embeddings") and callable(getattr(self, "_get_embeddings"))

        # Determine total number of validation pairs
        # For TripletDataset, use validation_pairs; for others, use samples
        if hasattr(val_dataset, 'validation_pairs'):
            num_pairs = len(val_dataset.validation_pairs)
        elif hasattr(val_dataset, 'samples'):
            num_pairs = len(val_dataset.samples)
        else:
            raise AttributeError("Dataset must have either 'validation_pairs' or 'samples' attribute")

        print(f"\n🔍 Computing verification metrics on {num_pairs} pairs...")
        
        for idx in tqdm(range(num_pairs), desc="  Evaluating pairs"):
            # Use the unified interface
            img1_path, img2_path, label = val_dataset.get_validation_pair(idx)
            
            # Load images
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            if use_embeddings:
                # Metric learning: contrastive / triplet
                emb1, emb2 = self._get_embeddings(img1, img2)
                score = F.cosine_similarity(emb1, emb2).item()
            else:
                # BCE Siamese: direct output
                score = self.model(img1, img2).item()
            
            if label == 1.0:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
        
        # Decide if scores are already similarity or need to be inverted
        distances_are_similarity = not use_embeddings  # BCE → already similarity

        metrics = compute_metrics(
            np.array(genuine_scores),
            np.array(impostor_scores),
            distances_are_similarity= True#distances_are_similarity
        )
        
        print(f"\n  ✓ Evaluated {len(genuine_scores)} genuine + {len(impostor_scores)} impostor pairs")
        actual_ratio = len(genuine_scores) / (len(genuine_scores) + len(impostor_scores)) * 100
        print(f"  Actual ratio: {actual_ratio:.1f}% genuine")
        
        return metrics
    
    def _update_history(self, epoch: int, train_loss: float, val_loss: float):
        """Update history with losses only."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
    
    def _print_epoch_summary(self, epoch: int, epochs: int, train_loss: float, val_loss: float):
        """Print formatted epoch summary."""
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    def train(
        self,
        train_loader,
        val_loader,
        val_dataset,
        epochs: int = 50,
        patience: int = 7,
        fold: int = None,
    ) -> Dict:
        """
        Main training loop (same for all trainers).
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (for loss calculation)
            val_dataset: Validation dataset (for metric calculation)
            epochs: Number of epochs
            patience: Early stopping patience
            fold: Fold number (for K-Fold)
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        train_dataset = train_loader.dataset
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate loss only
            val_loss = self.validate_loss(val_loader)
            
            # Update scheduler (if exists)
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self._update_history(epoch + 1, train_loss, val_loss)
            
            # Print summary
            self._print_epoch_summary(epoch, epochs, train_loss, val_loss)
            
            # Early stopping on validation loss
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(is_best=True, fold=fold)
                print(f"  ✓ Saved best model (Val Loss={self.best_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("  ⚠ Early stopping triggered")
                    break

            # === NEGATIVE RESAMPLING ===
            # Notify dataset that epoch has ended
            if hasattr(train_dataset, 'on_epoch_end'):
                train_dataset.on_epoch_end(epoch)
        
        # Final comprehensive validation
        print(f"\n{'='*60}")
        print("FINAL COMPREHENSIVE VALIDATION")
        print(f"{'='*60}")
        
        final_metrics = self.validate_comprehensive(val_dataset)
        print_results(final_metrics)
        
        # Save final checkpoint and metrics
        self.save_checkpoint(is_best=False, fold=fold)
        if fold is None: self.save_history()
        self.save_final_metrics(final_metrics, fold=fold)
        
        print(f"\n✓ Training completed! Best Val Loss={self.best_loss:.4f}\n")
        
        return self.history, final_metrics
    
    def train_kfold(
        self,
        data_root: str,
        n_splits: int = 5,
        batch_size: int = 16,
        num_workers: int = 4,
        samples_per_writer: int = 100,
        target_size: int = 448,
        epochs: int = 50,
        patience: int = 7,
        random_state: int = 42,
    ) -> Dict:
        """K-Fold Cross-Validation (implemented by subclasses with specific dataloader)."""
        raise NotImplementedError("Subclasses must implement train_kfold with appropriate dataloader")
    
    def _reset_weights(self, m):
        """Reset model weights."""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def save_checkpoint(self, is_best: bool = False, fold: int = None):
        """Save model checkpoint."""
        suffix = "best" if is_best else "final"
        if fold is not None:
            path = os.path.join(self.results_dir, f"{self.model_name}_fold{fold}_{suffix}.pth")
        else:
            path = os.path.join(self.results_dir, f"{self.model_name}_{suffix}.pth")
        torch.save(self.model.state_dict(), path)
    
    def save_history(self):
        """Save training history to CSV."""
        path = os.path.join(self.results_dir, f"{self.model_name}_history.csv")
        pd.DataFrame(self.history).to_csv(path, index=False)
    
    def save_final_metrics(self, metrics: Dict, fold: int = None):
        """Save final comprehensive metrics to CSV."""
        
        if fold is not None:
            path = os.path.join(self.results_dir, f"{self.model_name}_fold{fold}_final_metrics.csv")
        else:
            path = os.path.join(self.results_dir, f"{self.model_name}_final_metrics.csv")
        
        pd.DataFrame([metrics]).to_csv(path, index=False)

    def _init_history(self):
        """Initialize empty history dict."""
        return {k: [] for k in self.history.keys()}
    
    def cleanup(self):
        """Cleanup resources."""
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()
        gc.collect()