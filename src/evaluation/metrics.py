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

    for far_target in [0.001, 0.01]:
        key_prefix = f"far_{far_target}"

        # inizializza tutto a None
        operating_points[f"{key_prefix}_effective"] = None
        operating_points[f"frr_at_{key_prefix}"] = None
        operating_points[f"gar_at_{key_prefix}"] = None
        operating_points[f"threshold_at_{key_prefix}"] = None

        # cerca soglie che rispettano FAR <= target
        valid_idxs = np.where(fpr <= far_target)[0]
        if len(valid_idxs) == 0:
            continue

        idx = valid_idxs[-1]

        far_eff = fpr[idx]
        frr = fnr[idx]
        gar = 1.0 - frr

        operating_points[f"{key_prefix}_effective"] = far_eff
        operating_points[f"frr_at_{key_prefix}"] = frr
        operating_points[f"gar_at_{key_prefix}"] = gar
        operating_points[f"threshold_at_{key_prefix}"] = thresholds[idx]

    # ------------------------------------------------------------------
    # 7. d-prime & decidability (computed on DISTANCES) ### DA CONTROLLARE
    # ------------------------------------------------------------------
    # mu_genuine / mu_impostor restano sulla STESSA scala di genuine_vals /
    # impostor_vals (score grezzi, alto = piu' simile). Nessuna conversione
    # in distanza: cosi' sono coerenti con i valori salvati e con
    # calculate_biometric_metrics del notebook di analisi.
    mu_g = np.mean(genuine_vals)
    mu_i = np.mean(impostor_vals)

    sigma_g = np.std(genuine_vals, ddof=1)
    sigma_i = np.std(impostor_vals, ddof=1)

    pooled_std = np.sqrt(0.5 * (sigma_g**2 + sigma_i**2))

    if pooled_std > 0:
        d_prime = (mu_g - mu_i) / pooled_std
        decidability = abs(d_prime)
    else:
        d_prime = 0.0
        decidability = 0.0

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

        # Score statistics
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
    
    print(f"\n📏 SCORE STATISTICS:")
    print(f"  Genuine:  μ={metrics['mu_genuine']:.4f}, σ={metrics['sigma_genuine']:.4f}")
    print(f"  Impostor: μ={metrics['mu_impostor']:.4f}, σ={metrics['sigma_impostor']:.4f}")
    
    print(f"\n⚙️ OPERATING POINTS:")
    print(f"  EER Threshold:               {metrics['eer_threshold']:.4f}")

    for far_target in [0.001, 0.01]:
        key_prefix = f"far_{far_target}"

        far_eff = metrics.get(f"{key_prefix}_effective")
        frr = metrics.get(f"frr_at_{key_prefix}")
        gar = metrics.get(f"gar_at_{key_prefix}")
        thr = metrics.get(f"threshold_at_{key_prefix}")

        print(f"\n  FAR Target = {far_target*100:.2f}%")

        if far_eff is None:
            print(f"    ❌ Operating point not reachable")
        else:
            print(f"    FAR (effective):           {far_eff*100:.4f}%")
            print(f"    FRR @ FAR:                 {frr*100:.4f}%")
            print(f"    GAR @ FAR:                 {gar*100:.4f}%")
            print(f"    Threshold @ FAR:           {thr:.4f}")

    print(f"{'='*70}\n")