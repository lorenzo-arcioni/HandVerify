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
    def validate_comprehensive(self, val_dataset, num_pairs: int = 1000) -> Dict[str, float]:
        """
        Comprehensive validation computing all verification metrics.
        WITH DEBUG PRINTS
        """
        self.model.eval()
        genuine_dists = []
        impostor_dists = []
        
        writer_ids = val_dataset.writer_ids
        writer_images = val_dataset.writer_images
        
        print("\n🔍 Computing verification metrics...")
        
        # Genuine pairs
        for _ in tqdm(range(num_pairs // 2), desc="  Genuine pairs"):
            writer = random.choice(writer_ids)
            if len(writer_images[writer]) < 2:
                continue
            
            img1_path, img2_path = random.sample(writer_images[writer], 2)
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            emb1, emb2 = self._get_embeddings(img1, img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            genuine_dists.append(dist)
        
        # Impostor pairs
        for _ in tqdm(range(num_pairs // 2), desc="  Impostor pairs"):
            w1, w2 = random.sample(writer_ids, 2)
            img1_path = random.choice(writer_images[w1])
            img2_path = random.choice(writer_images[w2])
            
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            emb1, emb2 = self._get_embeddings(img1, img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            impostor_dists.append(dist)
        
        # ========================================================================
        # DEBUG PRINTS
        # ========================================================================
        genuine_dists = np.array(genuine_dists)
        impostor_dists = np.array(impostor_dists)
        
        print("\n" + "="*70)
        print("🐛 DEBUG: Distance Distributions")
        print("="*70)
        print(f"\nGenuine distances:")
        print(f"  Count:  {len(genuine_dists)}")
        print(f"  Min:    {genuine_dists.min():.4f}")
        print(f"  Max:    {genuine_dists.max():.4f}")
        print(f"  Mean:   {genuine_dists.mean():.4f}")
        print(f"  Std:    {genuine_dists.std():.4f}")
        print(f"  Median: {np.median(genuine_dists):.4f}")
        
        print(f"\nImpostor distances:")
        print(f"  Count:  {len(impostor_dists)}")
        print(f"  Min:    {impostor_dists.min():.4f}")
        print(f"  Max:    {impostor_dists.max():.4f}")
        print(f"  Mean:   {impostor_dists.mean():.4f}")
        print(f"  Std:    {impostor_dists.std():.4f}")
        print(f"  Median: {np.median(impostor_dists):.4f}")
        
        print(f"\nDistribution overlap:")
        overlap_min = max(genuine_dists.min(), impostor_dists.min())
        overlap_max = min(genuine_dists.max(), impostor_dists.max())
        print(f"  Overlap range: [{overlap_min:.4f}, {overlap_max:.4f}]")
        print(f"  Genuine in overlap: {((genuine_dists >= overlap_min) & (genuine_dists <= overlap_max)).sum()}/{len(genuine_dists)}")
        print(f"  Impostor in overlap: {((impostor_dists >= overlap_min) & (impostor_dists <= overlap_max)).sum()}/{len(impostor_dists)}")
        print("="*70 + "\n")
        
        # Compute metrics
        metrics = compute_verification_metrics(genuine_dists, impostor_dists)
        
        # ========================================================================
        # DEBUG: Verifica threshold e predictions
        # ========================================================================
        print("\n" + "="*70)
        print("🐛 DEBUG: Threshold & Predictions")
        print("="*70)
        print(f"\nEER Threshold: {metrics['eer_threshold']:.4f}")
        
        # Simula predictions @ EER threshold
        all_dists = np.concatenate([genuine_dists, impostor_dists])
        all_labels = np.concatenate([np.ones(len(genuine_dists)), np.zeros(len(impostor_dists))])
        predictions = (all_dists <= metrics['eer_threshold']).astype(int)  # dist < thresh = same writer
        
        print(f"\nPredictions @ EER threshold:")
        print(f"  Total samples: {len(predictions)}")
        print(f"  Predicted SAME (1): {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
        print(f"  Predicted DIFF (0): {(1-predictions).sum()} ({(1-predictions).sum()/len(predictions)*100:.1f}%)")
        
        print(f"\nActual labels:")
        print(f"  Genuine (1): {all_labels.sum()} ({all_labels.sum()/len(all_labels)*100:.1f}%)")
        print(f"  Impostor (0): {(1-all_labels).sum()} ({(1-all_labels).sum()/len(all_labels)*100:.1f}%)")
        
        # Confusion matrix manual
        tp = ((predictions == 1) & (all_labels == 1)).sum()
        tn = ((predictions == 0) & (all_labels == 0)).sum()
        fp = ((predictions == 1) & (all_labels == 0)).sum()
        fn = ((predictions == 0) & (all_labels == 1)).sum()
        
        print(f"\nManual Confusion Matrix:")
        print(f"  TP (genuine correctly as SAME):  {tp}")
        print(f"  TN (impostor correctly as DIFF): {tn}")
        print(f"  FP (impostor wrongly as SAME):   {fp}")
        print(f"  FN (genuine wrongly as DIFF):    {fn}")
        
        manual_acc = (tp + tn) / len(predictions)
        manual_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        manual_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        manual_f1 = 2 * manual_prec * manual_rec / (manual_prec + manual_rec) if (manual_prec + manual_rec) > 0 else 0
        
        print(f"\nManual metrics:")
        print(f"  Accuracy:  {manual_acc:.4f}")
        print(f"  Precision: {manual_prec:.4f}")
        print(f"  Recall:    {manual_rec:.4f}")
        print(f"  F1:        {manual_f1:.4f}")
        
        print(f"\nMetrics from compute_verification_metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print("="*70 + "\n")
        
        return metrics
    
    @abstractmethod
    def _get_embeddings(self, img1, img2):
        """Get embeddings from images (implemented by subclasses)."""
        pass
    
    def _update_history(self, epoch: int, train_loss: float, val_loss: float):
        """Update history with losses only."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
    
    def _print_epoch_summary(self, epoch: int, epochs: int, train_loss: float, val_loss: float):
        """Print formatted epoch summary."""
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    def _print_final_metrics(self, metrics: Dict):
        """Print comprehensive metrics at end of training."""
        print(f"\n{'='*70}")
        print(f"FINAL VALIDATION METRICS")
        print(f"{'='*70}")
        
        print(f"\n🎯 PRIMARY BIOMETRIC METRICS:")
        print(f"  EER:              {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
        print(f"  AUC:              {metrics['auc']:.4f}")
        print(f"  EER Threshold:    {metrics['eer_threshold']:.4f}")
        
        print(f"\n📊 CLASSIFICATION METRICS (@ EER threshold):")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1-Score:         {metrics['f1']:.4f}")
        
        print(f"\n📈 DISCRIMINABILITY:")
        print(f"  d-prime (d'):     {metrics['d_prime']:.4f}")
        print(f"  Decidability:     {metrics['decidability']:.4f}")
        
        print(f"\n⚙️ FAR-BASED METRICS:")
        print(f"  Acc @ FAR=0.1%:   {metrics['acc_far_0.001']:.4f} (FRR={metrics['frr_far_0.001']:.4f})")
        print(f"  Acc @ FAR=1.0%:   {metrics['acc_far_0.01']:.4f} (FRR={metrics['frr_far_0.01']:.4f})")
        
        print(f"\n📏 DISTRIBUTION STATISTICS:")
        print(f"  Genuine:          μ={metrics['mu_genuine']:.4f}, σ={metrics['sigma_genuine']:.4f}")
        print(f"  Impostor:         μ={metrics['mu_impostor']:.4f}, σ={metrics['sigma_impostor']:.4f}")
        
        print(f"{'='*70}\n")
    
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
        
        # Final comprehensive validation
        print(f"\n{'='*60}")
        print("FINAL COMPREHENSIVE VALIDATION")
        print(f"{'='*60}")
        
        final_metrics = self.validate_comprehensive(val_dataset, num_pairs=1000)
        self._print_final_metrics(final_metrics)
        
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