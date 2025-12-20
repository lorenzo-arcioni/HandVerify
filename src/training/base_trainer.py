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

from ..evaluation.metrics import compute_verification_metrics


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
        
        # All trainers track the same comprehensive metrics
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],  # ✅ AGGIUNTO
            # Primary biometric metrics
            'val_eer': [],
            'val_auc': [],
            'val_eer_threshold': [],
            # Classification metrics (at EER threshold)
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            # Secondary metrics
            'val_d_prime': [],
            'val_decidability': [],
            # Accuracy at fixed FARs
            'val_acc_far_0.001': [],
            'val_acc_far_0.01': [],
            'val_frr_far_0.001': [],
            'val_frr_far_0.01': [],
            'val_threshold_far_0.001': [],
            'val_threshold_far_0.01': [],
            # Distribution statistics
            'val_mu_genuine': [],
            'val_mu_impostor': [],
            'val_sigma_genuine': [],
            'val_sigma_impostor': [],
        }
        
        self.best_eer = float('inf')
    
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
    def validate_comprehensive(self, val_dataset, num_pairs: int = 1000) -> Dict[str, float]:
        """
        Comprehensive validation computing all verification metrics.
        
        Args:
            val_dataset: Validation dataset
            num_pairs: Number of pairs to evaluate
            
        Returns:
            Dictionary with all metrics
        """
        self.model.eval()
        genuine_dists = []
        impostor_dists = []
        
        writer_ids = val_dataset.writer_ids
        writer_images = val_dataset.writer_images
        
        print("  Computing genuine pairs...", end=" ", flush=True)
        for _ in range(num_pairs // 2):
            writer = random.choice(writer_ids)
            if len(writer_images[writer]) < 2:
                continue
            
            img1_path, img2_path = random.sample(writer_images[writer], 2)
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            emb1, emb2 = self._get_embeddings(img1, img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            genuine_dists.append(dist)
        
        print("Done.", end=" ", flush=True)
        print("Computing impostor pairs...", end=" ", flush=True)
        
        for _ in range(num_pairs // 2):
            w1, w2 = random.sample(writer_ids, 2)
            img1_path = random.choice(writer_images[w1])
            img2_path = random.choice(writer_images[w2])
            
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            emb1, emb2 = self._get_embeddings(img1, img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            impostor_dists.append(dist)
        
        print("Done.")
        
        # Compute all metrics at once
        metrics = compute_verification_metrics(
            np.array(genuine_dists),
            np.array(impostor_dists)
        )
        
        return metrics
    
    @abstractmethod
    def _get_embeddings(self, img1, img2):
        """Get embeddings from images (implemented by subclasses)."""
        pass
    
    def _update_history(self, epoch: int, train_loss: float, val_loss: float, val_metrics: Dict):
        """Update history with all metrics."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)  # ✅ AGGIUNTO
        
        # Primary metrics
        self.history['val_eer'].append(val_metrics['eer'])
        self.history['val_auc'].append(val_metrics['auc'])
        self.history['val_eer_threshold'].append(val_metrics['eer_threshold'])
        
        # Classification metrics
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        
        # Secondary metrics
        self.history['val_d_prime'].append(val_metrics['d_prime'])
        self.history['val_decidability'].append(val_metrics['decidability'])
        
        # FAR-based metrics
        self.history['val_acc_far_0.001'].append(val_metrics['acc_far_0.001'])
        self.history['val_acc_far_0.01'].append(val_metrics['acc_far_0.01'])
        self.history['val_frr_far_0.001'].append(val_metrics['frr_far_0.001'])
        self.history['val_frr_far_0.01'].append(val_metrics['frr_far_0.01'])
        self.history['val_threshold_far_0.001'].append(val_metrics['threshold_far_0.001'])
        self.history['val_threshold_far_0.01'].append(val_metrics['threshold_far_0.01'])
        
        # Distribution statistics
        self.history['val_mu_genuine'].append(val_metrics['mu_genuine'])
        self.history['val_mu_impostor'].append(val_metrics['mu_impostor'])
        self.history['val_sigma_genuine'].append(val_metrics['sigma_genuine'])
        self.history['val_sigma_impostor'].append(val_metrics['sigma_impostor'])
    
    def _print_epoch_summary(self, epoch: int, epochs: int, train_loss: float, 
                            val_loss: float, val_metrics: Dict):
        """Print formatted epoch summary with all metrics."""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} Summary")
        print(f"{'='*70}")
        
        print(f"\n📉 LOSSES:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")  # ✅ AGGIUNTO
        
        print(f"\n🎯 PRIMARY BIOMETRIC METRICS:")
        print(f"  EER:              {val_metrics['eer']:.4f} ({val_metrics['eer']*100:.2f}%)")
        print(f"  AUC:              {val_metrics['auc']:.4f}")
        print(f"  EER Threshold:    {val_metrics['eer_threshold']:.4f}")
        
        print(f"\n📊 CLASSIFICATION METRICS (@ EER threshold):")
        print(f"  Accuracy:         {val_metrics['accuracy']:.4f}")
        print(f"  Precision:        {val_metrics['precision']:.4f}")
        print(f"  Recall:           {val_metrics['recall']:.4f}")
        print(f"  F1-Score:         {val_metrics['f1']:.4f}")
        
        print(f"\n📈 DISCRIMINABILITY:")
        print(f"  d-prime (d'):     {val_metrics['d_prime']:.4f}")
        print(f"  Decidability:     {val_metrics['decidability']:.4f}")
        
        print(f"\n⚙️ FAR-BASED METRICS:")
        print(f"  Acc @ FAR=0.1%:   {val_metrics['acc_far_0.001']:.4f} (FRR={val_metrics['frr_far_0.001']:.4f}, Thr={val_metrics['threshold_far_0.001']:.4f})")
        print(f"  Acc @ FAR=1.0%:   {val_metrics['acc_far_0.01']:.4f} (FRR={val_metrics['frr_far_0.01']:.4f}, Thr={val_metrics['threshold_far_0.01']:.4f})")
        
        print(f"\n📏 DISTRIBUTION STATISTICS:")
        print(f"  Genuine:          μ={val_metrics['mu_genuine']:.4f}, σ={val_metrics['sigma_genuine']:.4f}")
        print(f"  Impostor:         μ={val_metrics['mu_impostor']:.4f}, σ={val_metrics['sigma_impostor']:.4f}")
        
        print(f"{'='*70}")
    
    def train(
        self,
        train_loader,
        val_loader,  # ✅ AGGIUNTO: serve per calcolare val_loss
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
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            print(f"\nValidation Epoch {epoch+1}...")
            
            # Calculate val_loss
            val_loss = self.validate_loss(val_loader)
            
            # Calculate biometric metrics
            val_metrics = self.validate_comprehensive(val_dataset, num_pairs=1000)
            
            # Update scheduler (if exists)
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['eer'])
                else:
                    self.scheduler.step()
            
            # Update history
            self._update_history(epoch + 1, train_loss, val_loss, val_metrics)
            
            # Print summary
            self._print_epoch_summary(epoch, epochs, train_loss, val_loss, val_metrics)
            
            # Early stopping on EER (all trainers)
            if val_metrics['eer'] < self.best_eer:
                self.best_eer = val_metrics['eer']
                patience_counter = 0
                self.save_checkpoint(is_best=True, fold=fold)
                print(f"  ✓ Saved best model (EER={self.best_eer:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("  ⚠ Early stopping triggered")
                    break
        
        self.save_checkpoint(is_best=False, fold=fold)
        self.save_history()
        
        print(f"\n✓ Training completed! Best EER={self.best_eer:.4f}\n")
        
        return self.history
    
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
    
    def cleanup(self):
        """Cleanup resources."""
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()
        gc.collect()