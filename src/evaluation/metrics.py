"""
Evaluation Metrics
Unified metrics for classification and biometric evaluation.
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix
)
from scipy.stats import norm
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from PIL import Image


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def compute_basic_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute basic classification metrics.
    
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


# ============================================================================
# BIOMETRIC METRICS
# ============================================================================

def compute_eer(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).
    
    The EER is the point where False Accept Rate = False Reject Rate.
    
    Args:
        predictions: Predicted probabilities/similarities
        labels: Ground truth labels (1 for positive, 0 for negative)
        
    Returns:
        Tuple of (EER value, threshold at EER)
    """
    sorted_indices = np.argsort(predictions)
    sorted_labels = labels[sorted_indices]
    
    n_impostor = np.sum(labels == 0)
    n_genuine = np.sum(labels == 1)
    
    if n_impostor == 0 or n_genuine == 0:
        return 0.0, 0.5
    
    far, frr, thresholds = [], [], []
    
    for threshold in np.linspace(0, 1, 100):
        fa = np.sum((predictions >= threshold) & (labels == 0))
        fr = np.sum((predictions < threshold) & (labels == 1))
        
        far.append(fa / n_impostor)
        frr.append(fr / n_genuine)
        thresholds.append(threshold)
    
    far = np.array(far)
    frr = np.array(frr)
    eer_idx = np.argmin(np.abs(far - frr))
    
    return (far[eer_idx] + frr[eer_idx]) / 2, thresholds[eer_idx]


def compute_biometric_metrics(
    genuine_dists: np.ndarray,
    impostor_dists: np.ndarray
) -> Dict:
    """
    Compute comprehensive biometric metrics from distance distributions.
    
    Args:
        genuine_dists: Distances for genuine pairs (same writer)
        impostor_dists: Distances for impostor pairs (different writers)
        
    Returns:
        Dictionary with all biometric metrics including ROC, EER, d-prime, etc.
    """
    # Convert distances to similarity scores (lower dist = higher similarity)
    y_true = [1] * len(genuine_dists) + [0] * len(impostor_dists)
    y_scores = [-d for d in (list(genuine_dists) + list(impostor_dists))]
    
    # ROC Curve & AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    thresholds = -thresholds  # Convert back to distances
    
    metrics = {
        'auc': auc(fpr, tpr),
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
    }
    
    # Equal Error Rate
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    metrics['eer'] = (fpr[eer_idx] + fnr[eer_idx]) / 2
    metrics['eer_threshold'] = thresholds[eer_idx]
    
    # Accuracy at specific FAR operating points
    for target_far in [0.001, 0.01]:
        idx = np.argmin(np.abs(fpr - target_far))
        frr_at_far = fnr[idx]
        accuracy = 1 - ((fpr[idx] + frr_at_far) / 2)
        
        metrics[f'acc_far_{target_far*100:.1f}%'] = accuracy
        metrics[f'frr_far_{target_far*100:.1f}%'] = frr_at_far
        metrics[f'threshold_far_{target_far*100:.1f}%'] = thresholds[idx]
    
    # D-prime (discriminability index)
    mu_genuine = np.mean(genuine_dists)
    mu_impostor = np.mean(impostor_dists)
    sigma_genuine = np.std(genuine_dists)
    sigma_impostor = np.std(impostor_dists)
    
    pooled_std = np.sqrt((sigma_genuine**2 + sigma_impostor**2) / 2)
    metrics['d_prime'] = (mu_impostor - mu_genuine) / pooled_std if pooled_std > 0 else 0
    metrics['decidability'] = metrics['d_prime'] * np.sqrt(2)
    
    # Store distributions and statistics
    metrics['genuine_dists'] = genuine_dists
    metrics['impostor_dists'] = impostor_dists
    metrics['mu_genuine'] = mu_genuine
    metrics['mu_impostor'] = mu_impostor
    metrics['sigma_genuine'] = sigma_genuine
    metrics['sigma_impostor'] = sigma_impostor
    
    return metrics


def compute_rank1_identification(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    num_queries: int = 100
) -> float:
    """
    Compute Rank-1 Identification Rate.
    
    Args:
        model: Trained model
        dataset: Dataset with writer_ids and writer_images
        device: Device to run on
        num_queries: Number of queries to test
        
    Returns:
        Rank-1 identification rate
    """
    model.eval()
    writer_ids = dataset.writer_ids
    writer_images = dataset.writer_images
    
    # Build gallery: one embedding per writer
    gallery_embeddings = {}
    with torch.no_grad():
        for writer_id in writer_ids:
            img_path = random.choice(writer_images[writer_id])
            img = dataset.transform(Image.open(img_path).convert("L")).unsqueeze(0).to(device)
            emb = model.get_embedding(img).cpu().numpy()[0]
            gallery_embeddings[writer_id] = emb
    
    # Test queries
    correct = 0
    with torch.no_grad():
        for _ in range(num_queries):
            query_writer = random.choice(writer_ids)
            query_img_path = random.choice(writer_images[query_writer])
            query_img = dataset.transform(Image.open(query_img_path).convert("L")).unsqueeze(0).to(device)
            query_emb = model.get_embedding(query_img).cpu().numpy()[0]
            
            # Find closest gallery writer
            min_dist = float('inf')
            predicted_writer = None
            
            for writer_id, gallery_emb in gallery_embeddings.items():
                dist = np.linalg.norm(query_emb - gallery_emb)
                if dist < min_dist:
                    min_dist = dist
                    predicted_writer = writer_id
            
            if predicted_writer == query_writer:
                correct += 1
    
    return correct / num_queries


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_comprehensive(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    num_pairs: int = 2000,
    dataset_name: str = "Test"
) -> Dict:
    """
    Comprehensive biometric evaluation on a dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device to run on
        num_pairs: Number of pairs to evaluate
        dataset_name: Name for logging
        
    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EVALUATION: {dataset_name}")
    print(f"{'='*70}\n")
    
    model.eval()
    genuine_dists = []
    impostor_dists = []
    
    writer_ids = dataset.writer_ids
    writer_images = dataset.writer_images
    
    with torch.no_grad():
        # Genuine pairs
        for _ in tqdm(range(num_pairs // 2), desc="Genuine pairs"):
            writer = random.choice(writer_ids)
            if len(writer_images[writer]) < 2:
                continue
            
            img1_path, img2_path = random.sample(writer_images[writer], 2)
            img1 = dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(device)
            img2 = dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(device)
            
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            genuine_dists.append(dist)
        
        # Impostor pairs
        for _ in tqdm(range(num_pairs // 2), desc="Impostor pairs"):
            w1, w2 = random.sample(writer_ids, 2)
            img1_path = random.choice(writer_images[w1])
            img2_path = random.choice(writer_images[w2])
            
            img1 = dataset.transform(Image.open(img1_path).convert("L")).unsqueeze(0).to(device)
            img2 = dataset.transform(Image.open(img2_path).convert("L")).unsqueeze(0).to(device)
            
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            dist = F.pairwise_distance(emb1, emb2).item()
            impostor_dists.append(dist)
    
    # Compute all metrics
    metrics = compute_biometric_metrics(
        np.array(genuine_dists),
        np.array(impostor_dists)
    )
    
    # Rank-1 identification (if enough writers)
    if len(writer_ids) >= 10:
        rank1 = compute_rank1_identification(
            model, dataset, device,
            num_queries=min(100, len(writer_ids) * 5)
        )
        metrics['rank1_identification'] = rank1
    else:
        metrics['rank1_identification'] = None
    
    # Print results
    print(f"\n📊 PRIMARY METRICS:")
    print(f"  EER (Equal Error Rate):     {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"  AUC-ROC:                     {metrics['auc']:.4f}")
    print(f"  Accuracy @ FAR=0.1%:         {metrics['acc_far_0.1%']:.4f}")
    print(f"  Accuracy @ FAR=1.0%:         {metrics['acc_far_1.0%']:.4f}")
    
    print(f"\n📈 SECONDARY METRICS:")
    print(f"  d-prime (d'):                {metrics['d_prime']:.4f}")
    print(f"  Decidability Index:          {metrics['decidability']:.4f}")
    if metrics['rank1_identification'] is not None:
        print(f"  Rank-1 Identification Rate:  {metrics['rank1_identification']:.4f}")
    
    print(f"\n📏 DISTANCE STATISTICS:")
    print(f"  Genuine:  μ={metrics['mu_genuine']:.4f}, σ={metrics['sigma_genuine']:.4f}")
    print(f"  Impostor: μ={metrics['mu_impostor']:.4f}, σ={metrics['sigma_impostor']:.4f}")
    
    print(f"\n🎯 OPERATING POINTS:")
    print(f"  EER Threshold:          {metrics['eer_threshold']:.4f}")
    print(f"  Threshold @ FAR=0.1%:   {metrics['threshold_far_0.1%']:.4f}")
    print(f"  Threshold @ FAR=1.0%:   {metrics['threshold_far_1.0%']:.4f}")
    
    return metrics
