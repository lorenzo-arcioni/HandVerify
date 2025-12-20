"""
Visualization Functions
Unified visualization for training history and biometric evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional
import os


# ============================================================================
# TRAINING VISUALIZATIONS
# ============================================================================

def plot_training_history(
    history: Dict[str, List],
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot training and validation metrics (for BCE/Contrastive training).
    
    Args:
        history: Dict with 'epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'
        model_name: Name of the model
        save_path: Optional path to save figure
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


def plot_triplet_training_history(
    history: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot training history for triplet networks.
    
    Args:
        history: Dict with 'epoch', 'train_loss', 'val_eer', 'val_auc'
        model_name: Name of the model
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    epochs = history['epoch']
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Triplet Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # EER
    ax2.plot(epochs, history['val_eer'], 'r-o', label='Validation EER', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('EER')
    ax2.set_title(f'{model_name} - Equal Error Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AUC
    ax3.plot(epochs, history['val_auc'], 'g-o', label='Validation AUC', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC')
    ax3.set_title(f'{model_name} - AUC-ROC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
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
        results: Dict mapping model names to their history dicts
        metric: Metric to plot
        save_path: Optional path to save figure
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


# ============================================================================
# BIOMETRIC VISUALIZATIONS
# ============================================================================

def plot_biometric_results(
    metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Comprehensive biometric evaluation visualization.
    
    Args:
        metrics: Dictionary with all metrics from evaluate_comprehensive
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. ROC Curve with operating points
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(metrics['fpr'], metrics['tpr'], 'darkorange', lw=2,
             label=f'AUC={metrics["auc"]:.4f}')
    ax1.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(metrics['fpr'] - (1 - metrics['tpr'])))
    ax1.scatter(metrics['fpr'][eer_idx], metrics['tpr'][eer_idx],
                color='red', s=100, zorder=5, label=f'EER={metrics["eer"]:.4f}')
    
    # Mark FAR operating points
    for far_target in [0.001, 0.01]:
        idx = np.argmin(np.abs(metrics['fpr'] - far_target))
        ax1.scatter(metrics['fpr'][idx], metrics['tpr'][idx], s=80, zorder=5,
                   label=f'FAR={far_target*100:.1f}%')
    
    ax1.set_xlabel('False Accept Rate (FAR)')
    ax1.set_ylabel('True Accept Rate (TAR)')
    ax1.set_title('ROC Curve with Operating Points')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Distance distributions
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(metrics['genuine_dists'], bins=50, alpha=0.6,
             label='Genuine', color='green', density=True)
    ax2.hist(metrics['impostor_dists'], bins=50, alpha=0.6,
             label='Impostor', color='red', density=True)
    ax2.axvline(metrics['eer_threshold'], color='black', linestyle='--',
                label=f'EER Threshold', linewidth=2)
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Density')
    ax2.set_title('Distance Distributions')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. FAR vs FRR curves
    ax3 = plt.subplot(2, 3, 3)
    frr = 1 - metrics['tpr']
    ax3.plot(metrics['thresholds'], metrics['fpr'], label='FAR', color='red', lw=2)
    ax3.plot(metrics['thresholds'], frr, label='FRR', color='blue', lw=2)
    ax3.axvline(metrics['eer_threshold'], color='black', linestyle='--',
                label=f'EER Point', linewidth=2)
    ax3.set_xlabel('Threshold (Distance)')
    ax3.set_ylabel('Error Rate')
    ax3.set_title('FAR vs FRR')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Gaussian fits
    ax4 = plt.subplot(2, 3, 4)
    x = np.linspace(
        0,
        max(max(metrics['genuine_dists']), max(metrics['impostor_dists'])),
        200
    )
    
    genuine_pdf = norm.pdf(x, metrics['mu_genuine'], metrics['sigma_genuine'])
    impostor_pdf = norm.pdf(x, metrics['mu_impostor'], metrics['sigma_impostor'])
    
    ax4.plot(x, genuine_pdf, 'g-', lw=2, label='Genuine (fitted)')
    ax4.plot(x, impostor_pdf, 'r-', lw=2, label='Impostor (fitted)')
    ax4.fill_between(x, 0, genuine_pdf, alpha=0.3, color='green')
    ax4.fill_between(x, 0, impostor_pdf, alpha=0.3, color='red')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Probability Density')
    ax4.set_title(f'Gaussian Fits (d\'={metrics["d_prime"]:.2f})')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Detection Error Tradeoff (DET) curve
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(metrics['fpr'] * 100, frr * 100, 'b-', lw=2)
    ax5.scatter(metrics['fpr'][eer_idx] * 100, frr[eer_idx] * 100,
                color='red', s=100, zorder=5, label=f'EER={metrics["eer"]*100:.2f}%')
    ax5.set_xlabel('False Accept Rate (%)')
    ax5.set_ylabel('False Reject Rate (%)')
    ax5.set_title('Detection Error Tradeoff (DET)')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(alpha=0.3, which='both')
    
    # 6. Metrics summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['EER', f'{metrics["eer"]:.4f} ({metrics["eer"]*100:.2f}%)'],
        ['AUC-ROC', f'{metrics["auc"]:.4f}'],
        ['Acc @ FAR=0.1%', f'{metrics["acc_far_0.1%"]:.4f}'],
        ['Acc @ FAR=1.0%', f'{metrics["acc_far_1.0%"]:.4f}'],
        ['d-prime', f'{metrics["d_prime"]:.4f}'],
        ['Decidability', f'{metrics["decidability"]:.4f}'],
    ]
    
    if metrics.get('rank1_identification') is not None:
        summary_data.append(['Rank-1 ID Rate', f'{metrics["rank1_identification"]:.4f}'])
    
    table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Metrics Summary', fontweight='bold', fontsize=12, pad=20)
    
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
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
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


# ============================================================================
# COMPREHENSIVE REPORTS
# ============================================================================

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
        if all(metric in res['history'] for res in results.values()):
            plot_multiple_models_comparison(
                results=results,
                metric=metric,
                save_path=os.path.join(save_dir, f"comparison_{metric}.png")
            )
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for model_name, result in results.items():
        best = result.get('best_loss') or result.get('best_eer', 'N/A')
        print(f"{model_name:20s}: {best:.4f}" if isinstance(best, float) else f"{model_name:20s}: {best}")
    
    print(f"\n✓ Training report saved to {save_dir}")
