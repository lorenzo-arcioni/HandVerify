"""
BCE Trainer
Trainer class for Siamese networks with Binary Cross-Entropy loss.
"""

import os
import gc
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score

from ..models.base import BaseSiameseNetwork


class BCETrainer:
    """Trainer for Siamese networks using BCE loss"""
    
    def __init__(
        self,
        model: BaseSiameseNetwork,
        model_name: str,
        device: torch.device,
        results_dir: str = "results",
    ):
        """
        Args:
            model: Siamese network model
            model_name: Name for saving checkpoints
            device: Device to train on
            results_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        self.best_test_loss = float('inf')
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Adaptive learning rate based on model size
        num_params = sum(p.numel() for p in self.model.parameters())
        lr = 3e-5 if num_params > 15e6 else 5e-5
        
        # AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # ReduceLROnPlateau scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for img1, img2, labels in tqdm(train_loader, desc="Training"):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(img1, img2).squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        avg_loss = train_loss / len(train_loader)
        accuracy = accuracy_score(train_labels, train_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, test_loader: DataLoader) -> tuple:
        """
        Validate on test set.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        test_loss = 0
        test_preds = []
        test_labels = []
        
        for img1, img2, labels in tqdm(test_loader, desc="Validating"):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(img1, img2).squeeze()
            loss = self.criterion(outputs, labels)
            
            test_loss += loss.item()
            test_preds.extend((outputs > 0.5).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
        
        avg_loss = test_loss / len(test_loader)
        accuracy = accuracy_score(test_labels, test_preds)
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        patience: int = 5,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training dataloader
            test_loader: Test dataloader
            epochs: Number of epochs to train
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            test_loss, test_acc = self.validate(test_loader)
            
            # Update scheduler
            self.scheduler.step(test_loss)
            
            # Save metrics
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Test Loss={test_loss:.4f} Acc={test_acc:.4f}")
            
            # Early stopping and model saving
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"✓ Saved best model (test_loss: {test_loss:.4f})")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Save final model and history
        self.save_checkpoint(is_best=False)
        self.save_history()
        
        print(f"\n✓ {self.model_name} training completed! "
              f"Best test loss: {self.best_test_loss:.4f}\n")
        
        return self.history
    
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
    ) -> Dict:
        """
        Train with K-Fold Cross-Validation.
        
        Args:
            data_root: Root directory with writer subdirectories
            n_splits: Number of folds
            batch_size: Batch size
            num_workers: Number of data loading workers
            pairs_per_writer: Pairs per writer
            target_size: Image size
            epochs: Epochs per fold
            patience: Early stopping patience
            random_state: Random seed
            
        Returns:
            Dictionary with aggregated results across all folds
        """
        from ..data import create_dataloaders_kfold
        import numpy as np
        
        print(f"\n{'='*70}")
        print(f"K-FOLD TRAINING: {self.model_name} ({n_splits} folds)")
        print(f"{'='*70}\n")
        
        fold_results = []
        
        for fold in range(n_splits):
            print(f"\n{'#'*70}")
            print(f"# FOLD {fold+1}/{n_splits}")
            print(f"{'#'*70}\n")
            
            # Create dataloaders for this fold
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
            
            # Reset model for this fold
            self.model.apply(self._reset_weights)
            self._setup_optimizer()
            
            # Train this fold
            fold_history = self.train(
                train_loader=train_loader,
                test_loader=val_loader,
                epochs=epochs,
                patience=patience
            )
            
            fold_results.append({
                'fold': fold + 1,
                'best_loss': self.best_test_loss,
                'history': fold_history
            })
            
            print(f"\n✓ Fold {fold+1} completed - Best Loss: {self.best_test_loss:.4f}\n")
        
        # Aggregate results
        all_losses = [f['best_loss'] for f in fold_results]
        aggregated = {
            'fold_results': fold_results,
            'mean_loss': np.mean(all_losses),
            'std_loss': np.std(all_losses),
            'min_loss': np.min(all_losses),
            'max_loss': np.max(all_losses)
        }
        
        print(f"\n{'='*70}")
        print(f"K-FOLD RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Mean Loss: {aggregated['mean_loss']:.4f} ± {aggregated['std_loss']:.4f}")
        print(f"Min Loss:  {aggregated['min_loss']:.4f}")
        print(f"Max Loss:  {aggregated['max_loss']:.4f}")
        print(f"{'='*70}\n")
        
        # Save aggregated results
        pd.DataFrame({
            'fold': [f['fold'] for f in fold_results],
            'best_loss': [f['best_loss'] for f in fold_results]
        }).to_csv(os.path.join(self.results_dir, f"{self.model_name}_kfold_summary.csv"), index=False)
        
        return aggregated
    
    def _reset_weights(self, m):
        """Reset model weights for new fold"""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    
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


def train_multiple_models(
    models_dict: Dict[str, type],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
    results_dir: str = "results",
) -> Dict:
    """
    Train multiple models sequentially.
    
    Args:
        models_dict: Dictionary mapping model names to model classes
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: Device to train on
        epochs: Number of epochs per model
        patience: Early stopping patience
        results_dir: Directory to save results
        
    Returns:
        Dictionary with results for each model
    """
    results = {}
    
    for model_name, ModelClass in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Starting training: {model_name}")
        print(f"{'='*60}\n")
        
        # Create model instance
        model = ModelClass()
        
        # Create trainer
        trainer = BCETrainer(
            model=model,
            model_name=model_name,
            device=device,
            results_dir=results_dir
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            patience=patience
        )
        
        # Store results
        results[model_name] = {
            'history': history,
            'best_loss': trainer.best_test_loss
        }
        
        # Cleanup
        trainer.cleanup()
        print(f"✓ GPU memory cleared after {model_name}\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name:20s}: Best Test Loss = {result['best_loss']:.4f}")
    
    return results
