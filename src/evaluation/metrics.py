"""
Evaluation Metrics for Writer Verification
Comprehensive metrics including classification and biometric verification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from typing import Dict, Tuple


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate from ROC curve.
    
    Args:
        fpr: False Positive Rates
        tpr: True Positive Rates  
        thresholds: Decision thresholds
        
    Returns:
        (EER value, threshold at EER)
    """
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer, thresholds[eer_idx]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute classification metrics at a given threshold.
    
    Args:
        y_true: True labels (1 for genuine, 0 for impostor)
        y_scores: Similarity scores (higher = more similar)
        threshold: Decision threshold
        
    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def compute_verification_metrics(
    genuine_dists: np.ndarray,
    impostor_dists: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive verification metrics from distance distributions.
    """
    # Convert distances to similarity scores (lower distance = higher similarity)
    y_true = np.array([1] * len(genuine_dists) + [0] * len(impostor_dists))
    y_scores = np.concatenate([-genuine_dists, -impostor_dists])  # ✅ FIX: usa concatenate
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    thresholds = -thresholds  # Convert back to distances
    
    # Primary biometric metrics
    metrics = {
        'auc': auc(fpr, tpr),
    }
    
    # Equal Error Rate
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    metrics['eer'] = eer
    metrics['eer_threshold'] = eer_threshold
    
    # Classification metrics at EER threshold
    eer_sim_threshold = -eer_threshold  # Convert to similarity
    eer_classification = compute_classification_metrics(
        y_true,
        y_scores,  # ✅ Già è similarity scores
        eer_sim_threshold
    )
    
    metrics['accuracy'] = eer_classification['accuracy']
    metrics['precision'] = eer_classification['precision']
    metrics['recall'] = eer_classification['recall']
    metrics['f1'] = eer_classification['f1']
    
    # Accuracy at specific FAR operating points
    fnr = 1 - tpr
    for target_far in [0.001, 0.01]:
        idx = np.argmin(np.abs(fpr - target_far))
        frr_at_far = fnr[idx]
        accuracy = 1 - ((fpr[idx] + frr_at_far) / 2)
        
        metrics[f'acc_far_{target_far}'] = accuracy
        metrics[f'frr_far_{target_far}'] = frr_at_far
        metrics[f'threshold_far_{target_far}'] = thresholds[idx]
    
    # D-prime (discriminability index)
    mu_genuine = np.mean(genuine_dists)
    mu_impostor = np.mean(impostor_dists)
    sigma_genuine = np.std(genuine_dists)
    sigma_impostor = np.std(impostor_dists)
    
    pooled_std = np.sqrt((sigma_genuine**2 + sigma_impostor**2) / 2)
    d_prime = (mu_impostor - mu_genuine) / pooled_std if pooled_std > 0 else 0
    
    metrics['d_prime'] = d_prime
    metrics['decidability'] = d_prime * np.sqrt(2)
    
    # Distribution statistics
    metrics['mu_genuine'] = mu_genuine
    metrics['mu_impostor'] = mu_impostor
    metrics['sigma_genuine'] = sigma_genuine
    metrics['sigma_impostor'] = sigma_impostor
    
    # Store raw distributions for plotting
    metrics['genuine_dists'] = genuine_dists
    metrics['impostor_dists'] = impostor_dists
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['thresholds'] = thresholds
    
    return metrics

def print_verification_results(metrics: Dict[str, float], dataset_name: str = "Validation"):
    """Print formatted verification results."""
    print(f"\n{'='*70}")
    print(f"VERIFICATION METRICS: {dataset_name}")
    print(f"{'='*70}")
    print(f"📊 PRIMARY METRICS:")
    print(f"  EER (Equal Error Rate):     {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"  AUC-ROC:                     {metrics['auc']:.4f}")
    print(f"  Accuracy @ EER threshold:    {metrics['accuracy']:.4f}")
    
    print(f"\n🎯 CLASSIFICATION METRICS (@ EER threshold):")
    print(f"  Precision:                   {metrics['precision']:.4f}")
    print(f"  Recall:                      {metrics['recall']:.4f}")
    print(f"  F1-Score:                    {metrics['f1']:.4f}")
    
    print(f"\n📈 DISCRIMINABILITY:")
    print(f"  d-prime (d'):                {metrics['d_prime']:.4f}")
    print(f"  Decidability Index:          {metrics['decidability']:.4f}")
    
    print(f"\n📏 DISTANCE STATISTICS:")
    print(f"  Genuine:  μ={metrics['mu_genuine']:.4f}, σ={metrics['sigma_genuine']:.4f}")
    print(f"  Impostor: μ={metrics['mu_impostor']:.4f}, σ={metrics['sigma_impostor']:.4f}")
    
    print(f"\n⚙️ OPERATING POINTS:")
    print(f"  EER Threshold:               {metrics['eer_threshold']:.4f}")
    print(f"  Accuracy @ FAR=0.1%:         {metrics['acc_far_0.001']:.4f}")
    print(f"  Accuracy @ FAR=1.0%:         {metrics['acc_far_0.01']:.4f}")
    print(f"  Threshold @ FAR=0.1%:        {metrics['threshold_far_0.001']:.4f}")
    print(f"  Threshold @ FAR=1.0%:        {metrics['threshold_far_0.01']:.4f}")
    print(f"{'='*70}\n")