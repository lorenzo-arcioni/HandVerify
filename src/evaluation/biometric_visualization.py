"""
Biometric Visualization
Visualization functions for biometric evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


def plot_comprehensive_results(
    metrics: Dict,
    save_path: Optional[str] = None
):
    """
    Create comprehensive visualization of all biometric metrics.
    
    Args:
        metrics: Dictionary with all metrics from evaluate_comprehensive
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. ROC CURVE with operating points
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


def plot_training_history_triplet(
    history: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot training history for triplet networks.
    
    Args:
        history: Dictionary with 'epoch', 'train_loss', 'val_eer', 'val_auc'
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history['epoch']
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Triplet Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # EER plot
    ax2.plot(epochs, history['val_eer'], 'r-o', label='Validation EER', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('EER')
    ax2.set_title(f'{model_name} - Equal Error Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AUC plot
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