"""
Evaluation Metrics
Metrics for evaluating Siamese network performance.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, Tuple


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted probabilities (0-1)
        labels: Ground truth labels (0 or 1)
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    binary_preds = (predictions > threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(labels, binary_preds),
        'precision': precision_score(labels, binary_preds, zero_division=0),
        'recall': recall_score(labels, binary_preds, zero_division=0),
        'f1': f1_score(labels, binary_preds, zero_division=0),
        'auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0,
    }


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    binary_preds = (predictions > threshold).astype(int)
    return confusion_matrix(labels, binary_preds)


def compute_eer(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    The EER is the point where False Accept Rate = False Reject Rate.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels (1 for positive, 0 for negative)
        
    Returns:
        Tuple of (EER value, threshold at EER)
    """
    # Sort by predictions
    sorted_indices = np.argsort(predictions)
    sorted_preds = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Compute FAR and FRR at each threshold
    n_impostor = np.sum(labels == 0)
    n_genuine = np.sum(labels == 1)
    
    if n_impostor == 0 or n_genuine == 0:
        return 0.0, 0.5
    
    far = []
    frr = []
    thresholds = []
    
    for threshold in np.linspace(0, 1, 100):
        # False Accept: impostor pairs predicted as genuine
        fa = np.sum((predictions >= threshold) & (labels == 0))
        far_val = fa / n_impostor
        
        # False Reject: genuine pairs predicted as impostor
        fr = np.sum((predictions < threshold) & (labels == 1))
        frr_val = fr / n_genuine
        
        far.append(far_val)
        frr.append(frr_val)
        thresholds.append(threshold)
    
    # Find point where FAR ≈ FRR
    far = np.array(far)
    frr = np.array(frr)
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def evaluate_model_comprehensive(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Comprehensive evaluation with all metrics.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary with all metrics
    """
    basic_metrics = compute_metrics(predictions, labels, threshold)
    cm = compute_confusion_matrix(predictions, labels, threshold)
    eer, eer_threshold = compute_eer(predictions, labels)
    
    tn, fp, fn, tp = cm.ravel()
    
    return {
        **basic_metrics,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }