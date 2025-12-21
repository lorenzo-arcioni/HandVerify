# Notebooks Interattivi

Raccolta di Jupyter notebooks per esplorare, addestrare e valutare i modelli in modo interattivo.

## 📋 Indice

- [Notebooks Interattivi](#notebooks-interattivi)
  - [📋 Indice](#-indice)
  - [🚀 Setup Jupyter](#-setup-jupyter)
    - [Installazione](#installazione)
    - [Configurazione Kernel](#configurazione-kernel)
    - [Struttura Notebooks](#struttura-notebooks)
  - [📊 01: Data Exploration](#-01-data-exploration)
    - [Contenuti Notebook](#contenuti-notebook)
    - [Output Attesi](#output-attesi)
  - [🏋️ 02: Model Training](#️-02-model-training)
    - [Contenuti Notebook](#contenuti-notebook-1)
    - [Features Interattive](#features-interattive)
  - [📈 03: Evaluation \& Metrics](#-03-evaluation--metrics)
    - [Contenuti Notebook](#contenuti-notebook-2)
  - [🎯 04: Inference \& Deployment](#-04-inference--deployment)

## 🚀 Setup Jupyter

### Installazione

```bash
# Installa Jupyter
pip install jupyter jupyterlab ipywidgets

# Abilita widgets
jupyter nbextension enable --py widgetsnbextension

# Avvia Jupyter Lab
jupyter lab
```

### Configurazione Kernel

```bash
# Crea kernel con ambiente virtuale
python -m ipykernel install --user --name=handwriting-env --display-name "Handwriting Verification"

# Verifica kernel disponibili
jupyter kernelspec list
```

### Struttura Notebooks

```
notebooks/
├── 01_data_exploration.ipynb      # Esplorazione dataset
├── 02_model_training.ipynb        # Training interattivo
├── 03_evaluation.ipynb            # Analisi metriche
├── 04_inference.ipynb             # Predizioni su nuovi dati
├── utils/                         # Utility per notebooks
│   ├── __init__.py
│   ├── plotting.py               # Funzioni plot
│   └── visualization.py          # Visualizzazioni avanzate
└── data/                         # Dati di esempio
    └── samples/
```

## 📊 01: Data Exploration

**Obiettivo:** Esplorare e comprendere il dataset prima del training.

### Contenuti Notebook

```python
# ===== CELL 1: Setup =====
import sys
sys.path.append('..')  # Per importare src

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# Configurazione plot
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===== CELL 2: Carica Dataset Info =====
data_root = Path('../data/CVL_writers')

# Conta writer e immagini
writer_counts = {}
total_images = 0

for writer_dir in data_root.iterdir():
    if writer_dir.is_dir():
        images = list(writer_dir.glob('*.png'))
        writer_counts[writer_dir.name] = len(images)
        total_images += len(images)

print(f"Total Writers: {len(writer_counts)}")
print(f"Total Images: {total_images}")
print(f"Avg Images per Writer: {total_images / len(writer_counts):.2f}")

# ===== CELL 3: Distribuzione Immagini =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(list(writer_counts.values()), bins=20, edgecolor='black')
axes[0].set_xlabel('Number of Images')
axes[0].set_ylabel('Number of Writers')
axes[0].set_title('Distribution of Images per Writer')
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(list(writer_counts.values()))
axes[1].set_ylabel('Number of Images')
axes[1].set_title('Images per Writer - Box Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== CELL 4: Statistiche Dimensioni Immagini =====
def get_image_stats(data_root, num_samples=100):
    """Analizza dimensioni e statistiche immagini."""
    
    stats = {'width': [], 'height': [], 'aspect_ratio': [], 'size_kb': []}
    
    all_images = list(data_root.rglob('*.png'))[:num_samples]
    
    for img_path in all_images:
        img = Image.open(img_path)
        w, h = img.size
        
        stats['width'].append(w)
        stats['height'].append(h)
        stats['aspect_ratio'].append(w / h)
        stats['size_kb'].append(img_path.stat().st_size / 1024)
    
    return pd.DataFrame(stats)

stats_df = get_image_stats(data_root)

print("\nImage Statistics:")
print(stats_df.describe())

# ===== CELL 5: Visualizza Campioni =====
def plot_writer_samples(data_root, writer_id, num_samples=6):
    """Visualizza campioni di uno scrittore."""
    
    writer_dir = data_root / writer_id
    images = list(writer_dir.glob('*.png'))[:num_samples]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert('L')
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(img_path.name)
        axes[i].axis('off')
    
    plt.suptitle(f'Samples from {writer_id}', fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualizza primo writer
first_writer = sorted(writer_counts.keys())[0]
plot_writer_samples(data_root, first_writer)

# ===== CELL 6: Confronto Multi-Writer =====
def compare_writers(data_root, num_writers=4):
    """Confronta campioni da diversi writer."""
    
    writers = sorted(writer_counts.keys())[:num_writers]
    
    fig, axes = plt.subplots(num_writers, 3, figsize=(12, num_writers * 3))
    
    for i, writer in enumerate(writers):
        writer_dir = data_root / writer
        images = list(writer_dir.glob('*.png'))[:3]
        
        for j, img_path in enumerate(images):
            img = Image.open(img_path).convert('L')
            axes[i, j].imshow(img, cmap='gray')
            if j == 0:
                axes[i, j].set_ylabel(writer, fontsize=12)
            axes[i, j].axis('off')
    
    plt.suptitle('Multi-Writer Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

compare_writers(data_root)

# ===== CELL 7: Analisi Intensità Pixel =====
def analyze_pixel_intensity(data_root, num_samples=50):
    """Analizza distribuzione intensità pixel."""
    
    all_images = list(data_root.rglob('*.png'))[:num_samples]
    all_intensities = []
    
    for img_path in all_images:
        img = np.array(Image.open(img_path).convert('L'))
        all_intensities.extend(img.flatten())
    
    plt.figure(figsize=(12, 6))
    plt.hist(all_intensities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Mean Intensity: {np.mean(all_intensities):.2f}")
    print(f"Std Intensity: {np.std(all_intensities):.2f}")

analyze_pixel_intensity(data_root)

# ===== CELL 8: Data Augmentation Preview =====
from torchvision import transforms

# Definisci augmentation
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(448, scale=(0.9, 1.1)),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

def show_augmentations(img_path, num_augmentations=6):
    """Mostra effetto augmentation."""
    
    img = Image.open(img_path).convert('L')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i == 0:
            # Immagine originale
            ax.imshow(img, cmap='gray')
            ax.set_title('Original')
        else:
            # Augmentata
            aug_img = augmentation(img)
            ax.imshow(aug_img.squeeze(), cmap='gray')
            ax.set_title(f'Augmented {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test augmentation
sample_img = list(data_root.rglob('*.png'))[0]
show_augmentations(sample_img)

# ===== CELL 9: Split Analysis =====
from sklearn.model_selection import train_test_split

# Simula split train/test
writer_dirs = [str(d) for d in data_root.iterdir() if d.is_dir()]
train_dirs, test_dirs = train_test_split(writer_dirs, test_size=0.2, random_state=42)

print(f"\nDataset Split:")
print(f"Training Writers: {len(train_dirs)}")
print(f"Test Writers: {len(test_dirs)}")

# Conta immagini
train_images = sum([len(list(Path(d).glob('*.png'))) for d in train_dirs])
test_images = sum([len(list(Path(d).glob('*.png'))) for d in test_dirs])

print(f"Training Images: {train_images}")
print(f"Test Images: {test_images}")
```

### Output Attesi

- Statistiche dataset (writer, immagini)
- Visualizzazioni distribuzioni
- Campioni da vari writer
- Analisi dimensioni e intensità
- Preview augmentation

## 🏋️ 02: Model Training

**Obiettivo:** Training interattivo con monitoring in tempo reale.

### Contenuti Notebook

```python
# ===== CELL 1: Setup =====
import sys
sys.path.append('..')

import torch
from src.utils import set_seed, get_device, print_model_info
from src.models import get_model, list_models
from src.data import create_siamese_dataloaders
from src.training import BCETrainer

# Setup
set_seed(42)
device = get_device()

print("Available models:")
print(list_models())

# ===== CELL 2: Configurazione =====
CONFIG = {
    'model_name': 'resnet18',
    'batch_size': 16,
    'num_workers': 4,
    'test_size': 0.2,
    'samples_per_writer': 100,
    'target_size': 448,
    'epochs': 20,  # Ridotto per notebook
    'patience': 5,
}

print("Training Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ===== CELL 3: Carica Dati =====
print("\n📁 Loading data...")

train_loader, val_loader, train_ds, val_ds = create_siamese_dataloaders(
    data_root='../data/CVL_writers',
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
    test_size=CONFIG['test_size'],
    samples_per_writer=CONFIG['samples_per_writer'],
    target_size=CONFIG['target_size'],
    random_state=42
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ===== CELL 4: Visualizza Batch =====
# Visualizza un batch di training
img1, img2, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    # Immagine 1
    axes[0, i].imshow(img1[i].squeeze(), cmap='gray')
    axes[0, i].set_title(f'Image 1 - Label: {labels[i].item():.0f}')
    axes[0, i].axis('off')
    
    # Immagine 2
    axes[1, i].imshow(img2[i].squeeze(), cmap='gray')
    axes[1, i].set_title(f'Image 2')
    axes[1, i].axis('off')

plt.suptitle('Training Batch Sample', fontsize=16)
plt.tight_layout()
plt.show()

positive_ratio = labels.sum() / len(labels)
print(f"\nPositive ratio in batch: {positive_ratio:.2%}")

# ===== CELL 5: Inizializza Modello =====
print("\n🏗️ Initializing model...")

model = get_model(
    CONFIG['model_name'],
    in_channels=1,
    projection_dim=512
)

print_model_info(model, CONFIG['model_name'])

# ===== CELL 6: Setup Trainer =====
trainer = BCETrainer(
    model=model,
    model_name=f"{CONFIG['model_name']}_notebook",
    device=device,
    results_dir='../results/notebook_exp'
)

print("\n✅ Trainer initialized")
print(f"Optimizer: {trainer.optimizer.__class__.__name__}")
print(f"Learning Rate: {trainer.optimizer.param_groups[0]['lr']}")

# ===== CELL 7: Training =====
# Con progress tracking
from IPython.display import clear_output
import time

print("\n🚀 Starting training...\n")

history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=CONFIG['epochs'],
    patience=CONFIG['patience']
)

print("\n✅ Training completed!")

# ===== CELL 8: Plot Training History =====
history_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(history_df['epoch'], history_df['train_loss'], 
             marker='o', label='Train Loss', linewidth=2)
axes[0].plot(history_df['epoch'], history_df['val_loss'], 
             marker='s', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training History')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning Rate (se salvato)
if hasattr(trainer.optimizer, 'param_groups'):
    lr_values = [trainer.optimizer.param_groups[0]['lr']] * len(history_df)
    axes[1].plot(history_df['epoch'], lr_values, marker='o', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== CELL 9: Metriche Finali =====
print("\n" + "="*70)
print("FINAL METRICS")
print("="*70)
print(f"EER: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"d-prime: {metrics['d_prime']:.4f}")
print("="*70)

# ===== CELL 10: Salva Modello =====
# Opzionale: esporta per uso successivo
checkpoint_path = f"../results/notebook_exp/{CONFIG['model_name']}_notebook_best.pth"
print(f"\n✅ Model saved to: {checkpoint_path}")
```

### Features Interattive

**Progress Bar Real-time:**
```python
from tqdm.notebook import tqdm

# Sostituisce tqdm standard con versione notebook
# Già integrato nei trainer
```

**Live Plot Updates:**
```python
from IPython.display import clear_output

# Nel loop training
for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    # Update plot
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.show()
```

## 📈 03: Evaluation & Metrics

**Obiettivo:** Analisi approfondita delle metriche e performance.

### Contenuti Notebook

```python
# ===== CELL 1: Setup =====
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.evaluation import compute_verification_metrics, print_verification_results

# ===== CELL 2: Carica Metriche =====
# Da training precedente
metrics_path = '../results/notebook_exp/resnet18_notebook_final_metrics.csv'
metrics_df = pd.read_csv(metrics_path)

print("Loaded Metrics:")
print(metrics_df.head())

# ===== CELL 3: Metriche Principali =====
metrics = metrics_df.iloc[0]

print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)
print(f"EER: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Accuracy @ EER: {metrics['accuracy']:.4f}")
print(f"d-prime: {metrics['d_prime']:.4f}")
print("="*70)

# ===== CELL 4: ROC Curve =====
# Ricostruisci da dati salvati (se disponibili)
# Oppure ricalcola con modello

def plot_roc_curve_interactive(fpr, tpr, auc_score, eer_idx):
    """Plot ROC con punto EER evidenziato."""
    
    fig = plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC={auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # EER point
    plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=12,
             label=f'EER={fpr[eer_idx]:.4f}')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usa dati salvati o ricalcola
# plot_roc_curve_interactive(fpr, tpr, metrics['auc'], eer_idx)

# ===== CELL 5: Distance Distributions =====
# Se hai salvato genuine_dists e impostor_dists

def plot_distributions_interactive(genuine_dists, impostor_dists):
    """Plot distribuzioni genuine vs impostor."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    axes[0].hist(genuine_dists, bins=50, alpha=0.6, label='Genuine',
                 color='green', density=True, edgecolor='black')
    axes[0].hist(impostor_dists, bins=50, alpha=0.6, label='Impostor',
                 color='red', density=True, edgecolor='black')
    
    mu_g, mu_i = np.mean(genuine_dists), np.mean(impostor_dists)
    axes[0].axvline(mu_g, color='green', linestyle='--', linewidth=2)
    axes[0].axvline(mu_i, color='red', linestyle='--', linewidth=2)
    
    axes[0].set_xlabel('Distance')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distance Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data = [genuine_dists, impostor_dists]
    axes[1].boxplot(data, labels=['Genuine', 'Impostor'])
    axes[1].set_ylabel('Distance')
    axes[1].set_title('Distance Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# plot_distributions_interactive(genuine_dists, impostor_dists)

# ===== CELL 6: Error Rates Analysis =====
def plot_error_rates_interactive(fpr, fnr, thresholds, eer_idx):
    """Plot FAR e FRR vs threshold."""
    
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, fpr, 'r-', linewidth=2, label='FAR')
    plt.plot(thresholds, fnr, 'b-', linewidth=2, label='FRR')
    
    # EER point
    plt.plot(thresholds[eer_idx], fpr[eer_idx], 'go', markersize=12,
             label=f'EER={fpr[eer_idx]:.4f}')
    plt.axvline(thresholds[eer_idx], color='green', linestyle='--', alpha=0.5)
    
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('Error Rates vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# plot_error_rates_interactive(fpr, fnr, thresholds, eer_idx)

# ===== CELL 7: Operating Points Comparison =====
operating_points = {
    'EER': {
        'threshold': metrics['eer_threshold'],
        'accuracy': metrics['accuracy'],
        'far': metrics['eer'],
        'frr': metrics['eer']
    },
    'FAR=0.1%': {
        'threshold': metrics['threshold_far_0.001'],
        'accuracy': metrics['acc_far_0.001'],
        'far': 0.001,
        'frr': metrics['frr_far_0.001']
    },
    'FAR=1.0%': {
        'threshold': metrics['threshold_far_0.01'],
        'accuracy': metrics['acc_far_0.01'],
        'far': 0.01,
        'frr': metrics['frr_far_0.01']
    }
}

op_df = pd.DataFrame(operating_points).T

print("\nOperating Points Comparison:")
print(op_df)

# Visualizza
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].bar(op_df.index, op_df['accuracy'])
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy at Different Operating Points')
axes[0].tick_params(axis='x', rotation=45)

# FAR
axes[1].bar(op_df.index, op_df['far'])
axes[1].set_ylabel('FAR')
axes[1].set_title('False Accept Rate')
axes[1].set_yscale('log')
axes[1].tick_params(axis='x', rotation=45)

# FRR
axes[2].bar(op_df.index, op_df['frr'])
axes[2].set_ylabel('FRR')
axes[2].set_title('False Reject Rate')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ===== CELL 8: Model Comparison =====
# Confronta con altri modelli (se disponibili)
models_to_compare = [
    '../results/resnet18/resnet18_final_metrics.csv',
    '../results/efficientnet/efficientnet_final_metrics.csv',
    '../results/mobilenet/mobilenet_final_metrics.csv'
]

comparison_data = []

for model_path in models_to_compare:
    if Path(model_path).exists():
        df = pd.read_csv(model_path)
        model_name = Path(model_path).parent.name
        comparison_data.append({
            'Model': model_name,
            'EER': df['eer'].values[0],
            'AUC': df['auc'].values[0],
            'd-prime': df['d_prime'].values[0]
        })

if comparison_data:
    comp_df = pd.DataFrame(comparison_data)
    
    print("\nModel Comparison:")
    print(comp_df)
    
    # Plot confronto
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(comp_df['Model'], comp_df['EER']*100)
    axes[0].set_ylabel('EER (%)')
    axes[0].set_title('Equal Error Rate')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(comp_df['Model'], comp_df['AUC'])
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Area Under ROC')
    axes[1].set_ylim([0.9, 1.0])
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(comp_df['Model'], comp_df['d-prime'])
    axes[2].set_ylabel("d'")
    axes[2].set_title('Discriminability')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ===== CELL 9: Interactive Threshold Selector =====
from ipywidgets import interact, FloatSlider

def threshold_explorer(threshold):
    """Esplora metriche a vari threshold."""
    
    # Simula predizioni (sostituisci con dati reali)
    predictions = (y_scores >= threshold).astype(int)
    
    # Calcola metriche
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, zero_division=0)
    rec = recall_score(y_true, predictions, zero_division=0)
    
    print(f"Threshold: {threshold:.3f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Threshold={threshold:.3f})')
    plt.show()

# Widget interattivo
#interact(threshold_explorer, 
#         threshold=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.5))
```

## 🎯 04: Inference & Deployment

**Obiettivo:** Usare il modello addestrato per fare predizioni su nuovi dati.