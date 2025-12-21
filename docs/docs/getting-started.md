# Guida Introduttiva

Questa guida ti accompagnerà attraverso i primi passi per utilizzare il sistema di verifica della calligrafia.

## 🔧 Installazione

### Prerequisiti

Prima di iniziare, assicurati di avere installato:

- **Python 3.8 o superiore**
- **pip** (gestore pacchetti Python)
- **CUDA 11.8+** (opzionale, per accelerazione GPU)
- **Git**

### Verifica Versione Python

```bash
python --version
# Output: Python 3.8.x o superiore
```

### Step 1: Clone del Repository

```bash
git clone https://github.com/yourusername/handwriting-verification.git
cd handwriting-verification
```

### Step 2: Crea Ambiente Virtuale (Consigliato)

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Installa le Dipendenze

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dipendenze principali:**
- `torch>=2.0.0` - Framework deep learning
- `torchvision>=0.15.0` - Modelli e trasformazioni per visione
- `numpy>=1.24.0` - Calcoli numerici
- `pandas>=2.0.0` - Gestione dati
- `scikit-learn>=1.3.0` - Metriche e split dati
- `Pillow>=10.0.0` - Elaborazione immagini
- `tqdm>=4.65.0` - Progress bar
- `matplotlib>=3.7.0` - Visualizzazione
- `seaborn>=0.12.0` - Grafici statistici

### Step 4: Verifica Installazione

```python
# test_installation.py
import torch
import torchvision
from src.utils import get_device

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = get_device()
print(f"Using device: {device}")
```

Esegui:
```bash
python test_installation.py
```

Output atteso:
```
PyTorch version: 2.x.x
Torchvision version: 0.x.x
CUDA available: True
Using device: cuda
GPU: NVIDIA GeForce RTX 3090
```

## 📁 Preparazione Dataset

### Struttura Directory Richiesta

Il sistema si aspetta una struttura gerarchica con una cartella per scrittore:

```
data/
└── CVL_writers/
    ├── writer_001/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    ├── writer_002/
    │   ├── image_001.png
    │   └── ...
    └── writer_N/
        └── ...
```

### Esempio: CVL Database

```bash
# Crea la struttura
mkdir -p data/CVL_writers

# Organizza le immagini per scrittore
# Ogni scrittore deve avere almeno 2 immagini
```

**Requisiti immagini:**
- Formato: PNG, JPG, JPEG
- Scala di grigi (1 canale) o RGB (convertito automaticamente)
- Minimo 2 immagini per scrittore
- Risoluzione consigliata: >= 448x448 pixel

### Script di Organizzazione

```python
# organize_dataset.py
import os
import shutil
from pathlib import Path

def organize_cvl_dataset(source_dir, output_dir):
    """
    Organizza il dataset CVL nella struttura richiesta.
    
    Args:
        source_dir: Directory con tutte le immagini
        output_dir: Directory di output organizzata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in Path(source_dir).glob('*.png'):
        # Estrai writer_id dal nome file
        # Esempio: cvl-1-1.png -> writer_001
        parts = img_file.stem.split('-')
        writer_id = f"writer_{int(parts[1]):03d}"
        
        # Crea directory scrittore
        writer_dir = Path(output_dir) / writer_id
        writer_dir.mkdir(exist_ok=True)
        
        # Copia immagine
        shutil.copy(img_file, writer_dir / img_file.name)
        
    print(f"✓ Dataset organizzato in {output_dir}")

# Utilizzo
organize_cvl_dataset('raw_data/cvl', 'data/CVL_writers')
```

## 🎯 Primo Training

### Esempio 1: Training Base con Siamese Network

```python
# train_simple.py
from src.utils import set_seed, get_device, print_model_info
from src.models import get_model
from src.data import create_siamese_dataloaders
from src.training import BCETrainer

# 1. Setup
set_seed(42)
device = get_device()

# 2. Crea dataloaders
print("📁 Caricamento dati...")
train_loader, val_loader, train_ds, val_ds = create_siamese_dataloaders(
    data_root='data/CVL_writers',
    batch_size=16,
    num_workers=4,
    test_size=0.2,
    samples_per_writer=100,
    target_size=448,
    random_state=42
)

# 3. Inizializza modello
print("\n🏗️ Inizializzazione modello...")
model = get_model('resnet18', in_channels=1, projection_dim=512)
print_model_info(model, 'ResNet18-Siamese')

# 4. Crea trainer
trainer = BCETrainer(
    model=model,
    model_name='resnet18_siamese',
    device=device,
    results_dir='results/experiment_1'
)

# 5. Addestra
print("\n🚀 Inizio training...\n")
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=50,
    patience=7
)

# 6. Risultati
print("\n📊 Training completato!")
print(f"Best Validation Loss: {trainer.best_loss:.4f}")
print(f"Final EER: {metrics['eer']:.4f}")
print(f"Final AUC: {metrics['auc']:.4f}")
```

Esegui:
```bash
python train_simple.py
```

### Esempio 2: Training con Contrastive Learning

```python
# train_contrastive.py
from src.models.contrastive_backbones import ContrastiveMobileNetV3Small
from src.data import create_contrastive_dataloaders
from src.training import ContrastiveTrainer

# Setup
set_seed(42)
device = get_device()

# Dataloaders
train_loader, val_loader, train_ds, val_ds = create_contrastive_dataloaders(
    data_root='data/CVL_writers',
    batch_size=32,
    test_size=0.2
)

# Modello
model = ContrastiveMobileNetV3Small(in_channels=1, projection_dim=128)

# Trainer
trainer = ContrastiveTrainer(
    model=model,
    model_name='mobilenetv3_contrastive',
    device=device,
    margin=1.0,
    results_dir='results/contrastive_exp'
)

# Training
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=50
)
```

### Esempio 3: Training con Triplet Loss

```python
# train_triplet.py
from src.models.triplet_backbones import TripletResNet18
from src.data import create_triplet_dataloaders
from src.training import TripletTrainer

# Dataloaders
train_loader, val_loader, train_ds, val_ds = create_triplet_dataloaders(
    data_root='data/CVL_writers',
    batch_size=16,
    samples_per_writer=100
)

# Modello
model = TripletResNet18(in_channels=1, embedding_dim=128)

# Trainer con margin personalizzato
trainer = TripletTrainer(
    model=model,
    model_name='resnet18_triplet',
    device=device,
    margin=0.5
)

# Training
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=50
)
```

## 📊 Output del Training

Durante il training vedrai output simili a:

```
==============================================================
Training resnet18_siamese
==============================================================

📁 Loading data...
TRAIN: 80 writers, 8000 samples
VAL: 20 writers, 2000 samples

🏗️ Model: ResNet18-Siamese
Parameters: 11.69M (11,690,000)

🚀 Starting training...

Epoch 1/50 - Train Loss: 0.2456 | Val Loss: 0.1823
  ✓ Saved best model (Val Loss=0.1823)

Epoch 2/50 - Train Loss: 0.1678 | Val Loss: 0.1456
  ✓ Saved best model (Val Loss=0.1456)

...

==============================================================
FINAL COMPREHENSIVE VALIDATION
==============================================================

🔍 Computing verification metrics...
  Genuine pairs: 100%|████████████| 500/500 [00:15<00:00]
  Impostor pairs: 100%|████████████| 500/500 [00:15<00:00]

======================================================================
FINAL VALIDATION METRICS
======================================================================

🎯 PRIMARY BIOMETRIC METRICS:
  EER:              0.0245 (2.45%)
  AUC:              0.9912
  EER Threshold:    0.3456

📊 CLASSIFICATION METRICS (@ EER threshold):
  Accuracy:         0.9755
  Precision:        0.9782
  Recall:           0.9734
  F1-Score:         0.9758

📈 DISCRIMINABILITY:
  d-prime (d'):     3.2456
  Decidability:     4.5912

======================================================================

✓ Training completed! Best Val Loss=0.1234
```

## 🔍 Verifica Risultati

I risultati vengono salvati automaticamente in:

```
results/experiment_1/
├── resnet18_siamese_best.pth         # Checkpoint miglior modello
├── resnet18_siamese_final.pth        # Checkpoint finale
├── resnet18_siamese_history.csv      # Storia training
└── resnet18_siamese_final_metrics.csv # Metriche finali
```

### Visualizza Storia Training

```python
import pandas as pd
import matplotlib.pyplot as plt

# Carica storia
history = pd.read_csv('results/experiment_1/resnet18_siamese_history.csv')

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True)
plt.savefig('training_history.png')
plt.show()
```

### Carica Metriche Finali

```python
# Carica metriche
metrics = pd.read_csv('results/experiment_1/resnet18_siamese_final_metrics.csv')

print("Metriche Finali:")
print(f"EER: {metrics['eer'].values[0]:.4f}")
print(f"AUC: {metrics['auc'].values[0]:.4f}")
print(f"Accuracy: {metrics['accuracy'].values[0]:.4f}")
print(f"d-prime: {metrics['d_prime'].values[0]:.4f}")
```

## 🎓 Prossimi Passi

Ora che hai completato il primo training, puoi:

1. **Esplorare altri modelli**: Prova `efficientnet_b0`, `mobilenet_v3_large`, ecc.
2. **K-Fold Cross-Validation**: Vedi [Training Avanzato](training.md#k-fold-cross-validation)
3. **Hyperparameter Tuning**: Ottimizza batch size, learning rate, margin
4. **Valutazione Dettagliata**: Consulta [Evaluation](evaluation.md)
5. **Inference**: Usa il modello per predizioni su nuovi dati

## ❓ Risoluzione Problemi Comuni

### CUDA Out of Memory

```python
# Riduci batch size
batch_size=8  # invece di 16

# Oppure riduci target_size
target_size=224  # invece di 448
```

### Import Error

```bash
# Assicurati di essere nella directory root
cd handwriting-verification

# Reinstalla dipendenze
pip install -r requirements.txt --force-reinstall
```

### Dataset Non Trovato

```bash
# Verifica percorso
ls data/CVL_writers/

# Deve mostrare directory di scrittori
writer_001/ writer_002/ ...
```

### Modello Non Disponibile

```python
from src.models import list_models

# Vedi tutti i modelli disponibili
print(list_models())
# Output: ['resnet18', 'resnet34', 'efficientnet_b0', ...]
```

## 📚 Risorse Aggiuntive

- **[Struttura Progetto](project-structure.md)**: Organizzazione del codice
- **[Dataset](datasets.md)**: Gestione dati dettagliata
- **[Training Avanzato](training.md)**: Tecniche di training avanzate
- **[Notebooks](notebooks.md)**: Tutorial interattivi
- **[API Reference](../api/)**: Documentazione completa API

---

**Prossimo:** [Struttura del Progetto →](project-structure.md)