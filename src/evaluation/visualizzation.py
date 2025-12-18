"""
Visualization utilities
Functions for visualizing training progress and results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List],
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with 'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = history['epoch']
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, history['test_loss'], 'r-o', label='Test Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', markersize=4)
    ax2.plot(epochs, history['test_acc'], 'r-o', label='Test Accuracy', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multiple_models_comparison(
    results: Dict[str, Dict],
    metric: str = 'test_loss',
    save_path: Optional[str] = None
):
    """
    Compare multiple models on a single metric.
    
    Args:
        results: Dictionary mapping model names to their history dictionaries
        metric: Metric to plot ('train_loss', 'test_loss', 'train_acc', 'test_acc')
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    for model_name, result in results.items():
        history = result['history']
        epochs = history['epoch']
        values = history[metric]
        plt.plot(epochs, values, '-o', label=model_name, markersize=3)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: 2x2 confusion matrix
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Different', 'Same'],
           yticklabels=['Different', 'Same'],
           title=f'{model_name} - Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_results_summary(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Create summary bar chart of best losses for all models.
    
    Args:
        results: Dictionary mapping model names to result dictionaries
        save_path: Optional path to save the figure
    """
    model_names = list(results.keys())
    best_losses = [results[name]['best_loss'] for name in model_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), best_losses, color='steelblue')
    
    # Highlight best model
    best_idx = np.argmin(best_losses)
    bars[best_idx].set_color('green')
    
    plt.xlabel('Model')
    plt.ylabel('Best Test Loss')
    plt.title('Model Performance Comparison')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (name, loss) in enumerate(zip(model_names, best_losses)):
        plt.text(i, loss, f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_report(
    results: Dict[str, Dict],
    save_dir: str = "results"
):
    """
    Create comprehensive training report with all visualizations.
    
    Args:
        results: Dictionary mapping model names to result dictionaries
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Individual model plots
    for model_name, result in results.items():
        plot_training_history(
            history=result['history'],
            model_name=model_name,
            save_path=os.path.join(save_dir, f"{model_name}_history.png")
        )
    
    # Comparison plots
    for metric in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
        plot_multiple_models_comparison(
            results=results,
            metric=metric,
            save_path=os.path.join(save_dir, f"comparison_{metric}.png")
        )
    
    # Summary plot
    plot_results_summary(
        results=results,
        save_path=os.path.join(save_dir, "summary.png")
    )
    
    print(f"✓ Training report saved to {save_dir}")