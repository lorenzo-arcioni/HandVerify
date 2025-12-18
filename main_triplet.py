"""
Main script for running triplet loss experiments on handwriting verification.
Replicates the notebook experiments in a single script.
"""

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data import TripletDataset
from src.models import TripletMobileNetV3Small
from src.training import TripletTrainer
from src.evaluation import (
    evaluate_comprehensive,
    plot_comprehensive_results,
    plot_training_history_triplet
)
from src.utils import set_seed, get_device


def main():
    # Configuration
    IAM_ROOT = "datasets/processed-handwritten/iam_processed"
    RIMES_ROOT = "datasets/processed-handwritten/rimes_processed"
    RESULTS_DIR = "results/triplet"
    
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    TARGET_SIZE = 448
    EMBEDDING_DIM = 128
    MARGIN = 0.5
    EPOCHS = 10
    PATIENCE = 7
    TRIPLETS_PER_WRITER = 100
    
    # Setup
    set_seed(42)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    iam_dirs = [
        os.path.join(IAM_ROOT, d)
        for d in sorted(os.listdir(IAM_ROOT))
        if os.path.isdir(os.path.join(IAM_ROOT, d))
    ]
    
    rimes_dirs = [
        os.path.join(RIMES_ROOT, d)
        for d in sorted(os.listdir(RIMES_ROOT))
        if os.path.isdir(os.path.join(RIMES_ROOT, d))
    ]
    
    print(f"IAM writers: {len(iam_dirs)}")
    print(f"RIMES writers: {len(rimes_dirs)}")
    
    # Split datasets
    iam_train, iam_temp = train_test_split(iam_dirs, test_size=0.2, random_state=42)
    iam_val, iam_test = train_test_split(iam_temp, test_size=0.5, random_state=42)
    
    rimes_train, rimes_temp = train_test_split(rimes_dirs, test_size=0.2, random_state=42)
    rimes_val, rimes_test = train_test_split(rimes_temp, test_size=0.5, random_state=42)
    
    print(f"\nIAM: Train={len(iam_train)}, Val={len(iam_val)}, Test={len(iam_test)}")
    print(f"RIMES: Train={len(rimes_train)}, Val={len(rimes_val)}, Test={len(rimes_test)}")
    
    results_summary = []
    
    # ========== EXPERIMENT 1: Train IAM → Test IAM ==========
    print("\n" + "="*70)
    print("EXPERIMENT 1: Train IAM → Test IAM")
    print("="*70)
    
    train_ds = TripletDataset(iam_train, train=True, 
                             triplets_per_writer=TRIPLETS_PER_WRITER, 
                             target_size=TARGET_SIZE)
    val_ds = TripletDataset(iam_val, train=False, 
                           triplets_per_writer=TRIPLETS_PER_WRITER, 
                           target_size=TARGET_SIZE)
    test_ds = TripletDataset(iam_test, train=False, 
                            triplets_per_writer=TRIPLETS_PER_WRITER, 
                            target_size=TARGET_SIZE)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    model = TripletMobileNetV3Small(embedding_dim=EMBEDDING_DIM)
    trainer = TripletTrainer(model, "iam_to_iam_triplet", device, MARGIN, RESULTS_DIR)
    
    history = trainer.train(train_loader, val_ds, EPOCHS, PATIENCE)
    plot_training_history_triplet(history, "IAM→IAM Triplet",
                                 os.path.join(RESULTS_DIR, "iam_to_iam_history.png"))
    
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "iam_to_iam_triplet_best.pth")))
    metrics = evaluate_comprehensive(model, test_ds, device, 2000, "IAM→IAM Test")
    plot_comprehensive_results(metrics, os.path.join(RESULTS_DIR, "iam_to_iam_comprehensive.png"))
    
    results_summary.append({
        'experiment': 'IAM→IAM',
        **{k: v for k, v in metrics.items() if not isinstance(v, (list, type(None)))}
    })
    
    trainer.cleanup()
    
    # ========== EXPERIMENT 2: Train IAM → Test RIMES ==========
    print("\n" + "="*70)
    print("EXPERIMENT 2: Train IAM → Test RIMES (Cross-Dataset)")
    print("="*70)
    
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "iam_to_iam_triplet_best.pth")))
    model = model.to(device)
    
    test_ds_rimes = TripletDataset(rimes_test, train=False, 
                                   triplets_per_writer=TRIPLETS_PER_WRITER, 
                                   target_size=TARGET_SIZE)
    
    metrics = evaluate_comprehensive(model, test_ds_rimes, device, 2000, "IAM→RIMES Test")
    plot_comprehensive_results(metrics, os.path.join(RESULTS_DIR, "iam_to_rimes_comprehensive.png"))
    
    results_summary.append({
        'experiment': 'IAM→RIMES',
        **{k: v for k, v in metrics.items() if not isinstance(v, (list, type(None)))}
    })
    
    del model
    torch.cuda.empty_cache()
    
    # ========== EXPERIMENT 3: Train RIMES → Test RIMES ==========
    print("\n" + "="*70)
    print("EXPERIMENT 3: Train RIMES → Test RIMES")
    print("="*70)
    
    train_ds = TripletDataset(rimes_train, train=True, 
                             triplets_per_writer=TRIPLETS_PER_WRITER, 
                             target_size=TARGET_SIZE)
    val_ds = TripletDataset(rimes_val, train=False, 
                           triplets_per_writer=TRIPLETS_PER_WRITER, 
                           target_size=TARGET_SIZE)
    test_ds = TripletDataset(rimes_test, train=False, 
                            triplets_per_writer=TRIPLETS_PER_WRITER, 
                            target_size=TARGET_SIZE)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    
    model = TripletMobileNetV3Small(embedding_dim=EMBEDDING_DIM)
    trainer = TripletTrainer(model, "rimes_to_rimes_triplet", device, MARGIN, RESULTS_DIR)
    
    history = trainer.train(train_loader, val_ds, EPOCHS, PATIENCE)
    plot_training_history_triplet(history, "RIMES→RIMES Triplet",
                                 os.path.join(RESULTS_DIR, "rimes_to_rimes_history.png"))
    
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "rimes_to_rimes_triplet_best.pth")))
    metrics = evaluate_comprehensive(model, test_ds, device, 2000, "RIMES→RIMES Test")
    plot_comprehensive_results(metrics, os.path.join(RESULTS_DIR, "rimes_to_rimes_comprehensive.png"))
    
    results_summary.append({
        'experiment': 'RIMES→RIMES',
        **{k: v for k, v in metrics.items() if not isinstance(v, (list, type(None)))}
    })
    
    trainer.cleanup()
    
    # ========== EXPERIMENT 4: Train RIMES → Test IAM ==========
    print("\n" + "="*70)
    print("EXPERIMENT 4: Train RIMES → Test IAM (Cross-Dataset)")
    print("="*70)
    
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "rimes_to_rimes_triplet_best.pth")))
    model = model.to(device)
    
    test_ds_iam = TripletDataset(iam_test, train=False, 
                                 triplets_per_writer=TRIPLETS_PER_WRITER, 
                                 target_size=TARGET_SIZE)
    
    metrics = evaluate_comprehensive(model, test_ds_iam, device, 2000, "RIMES→IAM Test")
    plot_comprehensive_results(metrics, os.path.join(RESULTS_DIR, "rimes_to_iam_comprehensive.png"))
    
    results_summary.append({
        'experiment': 'RIMES→IAM',
        **{k: v for k, v in metrics.items() if not isinstance(v, (list, type(None)))}
    })
    
    # ========== FINAL SUMMARY ==========
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "experiments_summary.csv"), index=False)
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(df_summary[['experiment', 'eer', 'auc', 'acc_far_0.1%', 
                      'acc_far_1.0%', 'd_prime']].to_string(index=False))
    print(f"\n✓ All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()