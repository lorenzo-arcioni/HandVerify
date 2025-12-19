"""
Supervised Contrastive Loss Trainer
Trainer class for supervised contrastive learning with comprehensive biometric evaluation.
"""

import os
import gc
from typing import Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Extends contrastive learning to use label information.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Normalized embeddings [batch_size, embedding_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Supervised contrastive loss value
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class ContrastiveTrainer:
    """Trainer for supervised contrastive learning"""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device,
        temperature: float = 0.5,
        results_dir: str = "results",
    ):
        """
        Args:
            model: Contrastive network model
            model_name: Name for saving checkpoints
            device: Device to train on
            temperature: Temperature parameter for contrastive loss
            results_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Loss function
        self.criterion = SupConLoss(temperature=temperature)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_eer': [],
            'val_auc': []
        }
        
        self.best_eer = float('inf')
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        train_loss = 0.0
        
        for img1, img2, labels in tqdm(train_loader, desc="Training", leave=False):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get embeddings for both views
            emb1 = self.model(img1)
            emb2 = self.model(img2)
            
            # Stack embeddings and duplicate labels
            embeddings = torch.cat([emb1, emb2], dim=0)
            labels_repeated = torch.cat([labels, labels], dim=0)
            
            # Compute loss
            loss = self.criterion(embeddings, labels_repeated)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    @torch.no_grad()
    def validate_quick(self, val_dataset, num_pairs: int = 1000) -> Dict:
        """
        Quick validation to compute EER and AUC.
        
        Args:
            val_dataset: Validation dataset
            num_pairs: Number of pairs to evaluate
            
        Returns:
            Dictionary with 'eer' and 'auc' metrics
        """
        from ..evaluation.metrics import compute_eer
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        genuine_dists = []
        impostor_dists = []
        
        writer_ids = val_dataset.writer_ids
        writer_images = val_dataset.writer_images
        
        # Genuine pairs
        for _ in range(num_pairs // 2):
            writer = random.choice(writer_ids)
            if len(writer_images[writer]) < 2:
                continue
            
            img1_path, img2_path = random.sample(writer_images[writer], 2)
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            emb1 = self.model(img1)
            emb2 = self.model(img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            genuine_dists.append(dist)
        
        # Impostor pairs
        for _ in range(num_pairs // 2):
            w1, w2 = random.sample(writer_ids, 2)
            img1_path = random.choice(writer_images[w1])
            img2_path = random.choice(writer_images[w2])
            
            img1 = val_dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(self.device)
            img2 = val_dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(self.device)
            
            emb1 = self.model(img1)
            emb2 = self.model(img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            impostor_dists.append(dist)
        
        # Convert distances to similarity scores
        labels = np.array([1] * len(genuine_dists) + [0] * len(impostor_dists))
        distances = np.array(genuine_dists + impostor_dists)
        
        # Compute EER
        eer, eer_threshold = compute_eer(1 - distances, labels)
        
        # Compute AUC
        auc = roc_auc_score(labels, -distances)
        
        return {'eer': eer, 'auc': auc}
    
    def train(
        self,
        train_loader: DataLoader,
        val_dataset,
        epochs: int = 50,
        patience: int = 7,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training dataloader
            val_dataset: Validation dataset
            epochs: Number of epochs to train
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} with Supervised Contrastive Loss")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            print(f"\nValidation Epoch {epoch+1}...")
            val_metrics = self.validate_quick(val_dataset, num_pairs=1000)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save metrics
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_eer'].append(val_metrics['eer'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f} | "
                  f"Val EER={val_metrics['eer']:.4f} AUC={val_metrics['auc']:.4f}")
            
            # Early stopping and model saving
            if val_metrics['eer'] < self.best_eer:
                self.best_eer = val_metrics['eer']
                patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"✓ Saved best model (EER: {val_metrics['eer']:.4f})")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("⚠ Early stopping triggered")
                    break
        
        # Save final model and history
        self.save_checkpoint(is_best=False)
        self.save_history()
        
        print(f"\n✓ {self.model_name} training completed! "
              f"Best EER: {self.best_eer:.4f}\n")
        
        return self.history
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        suffix = "best" if is_best else "final"
        path = os.path.join(self.results_dir, f"{self.model_name}_{suffix}.pth")
        torch.save(self.model.state_dict(), path)
    
    def save_history(self):
        """Save training history to CSV"""
        path = os.path.join(self.results_dir, f"{self.model_name}_history.csv")
        pd.DataFrame(self.history).to_csv(path, index=False)
    
    def cleanup(self):
        """Clean up GPU memory"""
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()
        gc.collect()