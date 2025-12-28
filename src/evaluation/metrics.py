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

def compute_metrics(
    genuine_vals: np.ndarray,
    impostor_vals: np.ndarray,
    distances_are_similarity: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive verification metrics for biometric verification systems.

    Args:
        genuine_vals (np.ndarray):
            Genuine pair values.
            - If distances_are_similarity=False → distances (lower = more similar)
            - If distances_are_similarity=True  → similarity scores (higher = more similar)

        impostor_vals (np.ndarray):
            Impostor pair values (same convention as genuine_vals)

        distances_are_similarity (bool):
            True  → values are similarity scores (e.g. BCE, cosine similarity)
            False → values are distances (e.g. Euclidean, contrastive, triplet)

    Returns:
        Dict[str, float]:
            Contains ROC/AUC, EER, classification metrics, FAR/FRR operating points,
            d-prime, decidability index, and raw distributions.
    """

    # ------------------------------------------------------------------
    # 1. Ground truth
    # ------------------------------------------------------------------
    y_true = np.array(
        [1] * len(genuine_vals) +
        [0] * len(impostor_vals)
    )

    # ------------------------------------------------------------------
    # 2. Convert EVERYTHING to similarity scores for ROC & classification
    #    (higher score = more genuine)
    # ------------------------------------------------------------------
    if distances_are_similarity:
        genuine_scores = genuine_vals
        impostor_scores = impostor_vals
    else:
        genuine_scores = -genuine_vals
        impostor_scores = -impostor_vals

    y_scores = np.concatenate([genuine_scores, impostor_scores])

    # ------------------------------------------------------------------
    # 3. ROC / AUC
    # ------------------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # ------------------------------------------------------------------
    # 4. Equal Error Rate (EER)
    # ------------------------------------------------------------------
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    # ------------------------------------------------------------------
    # 5. Classification metrics @ EER threshold
    # ------------------------------------------------------------------
    y_pred = (y_scores >= eer_threshold).astype(int)

    cls_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ------------------------------------------------------------------
    # 6. FAR operating points
    # ------------------------------------------------------------------
    # DA CHIARIRE BENE
    operating_points = {}
    for far in [0.001, 0.01]:
        idx = np.argmin(np.abs(fpr - far))
        frr = fnr[idx]
        acc = 1 - (far + frr) / 2

        operating_points[f"acc_far_{far}"] = acc
        operating_points[f"frr_far_{far}"] = frr
        operating_points[f"threshold_far_{far}"] = thresholds[idx]

    # ------------------------------------------------------------------
    # 7. d-prime & decidability (ALWAYS computed on DISTANCES)
    # ------------------------------------------------------------------
    if distances_are_similarity:
        # Convert similarity → distance for biometric statistics
        genuine_dist = 1.0 - genuine_vals
        impostor_dist = 1.0 - impostor_vals
    else:
        genuine_dist = genuine_vals
        impostor_dist = impostor_vals

    mu_g = np.mean(genuine_dist)
    mu_i = np.mean(impostor_dist)
    sigma_g = np.std(genuine_dist)
    sigma_i = np.std(impostor_dist)

    pooled_std = np.sqrt((sigma_g ** 2 + sigma_i ** 2) / 2)
    d_prime = (mu_i - mu_g) / pooled_std if pooled_std > 0 else 0.0
    decidability = d_prime * np.sqrt(2)

    # ------------------------------------------------------------------
    # 8. Collect metrics
    # ------------------------------------------------------------------
    metrics = {
        # Primary
        "auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,

        # Classification
        **cls_metrics,

        # Discriminability
        "d_prime": d_prime,
        "decidability": decidability,

        # Distance statistics
        "mu_genuine": mu_g,
        "mu_impostor": mu_i,
        "sigma_genuine": sigma_g,
        "sigma_impostor": sigma_i,

        # ROC curves
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,

        # Raw distributions (for plots)
        "genuine_vals": genuine_vals,
        "impostor_vals": impostor_vals,
    }

    metrics.update(operating_points)

    return metrics

def print_results(metrics: Dict[str, float], dataset_name: str = "Validation"):
    """Print formatted verification results."""
    print(f"\n{'='*70}")
    print(f"VERIFICATION METRICS: {dataset_name}")
    print(f"{'='*70}")
    print(f"📊 PRIMARY METRICS:")
    print(f"  EER (Equal Error Rate):     {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"  AUC-ROC:                     {metrics['auc']:.4f}")
    
    print(f"\n🎯 CLASSIFICATION METRICS (@ EER threshold):")
    print(f"  Accuracy @ EER threshold:    {metrics['accuracy']:.4f}")
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