# Struttura del Progetto

Questa pagina descrive l'organizzazione del codice e la responsabilità di ciascun modulo.

## 📂 Struttura Directory

```
handwriting-verification/
│
├── src/                          # Codice sorgente principale
│   ├── __init__.py
│   ├── data/                     # Gestione dataset
│   │   ├── __init__.py
│   │   ├── base_dataset.py       # Classe base dataset
│   │   ├── siamese_dataset.py    # Dataset per Siamese
│   │   ├── contrastive_dataset.py # Dataset per Contrastive
│   │   ├── triplet_dataset.py    # Dataset per Triplet
│   │   ├── dataloader_factory.py # Factory per dataloader
│   │   └── transforms.py         # Trasformazioni immagini
│   │
│   ├── models/                   # Architetture modelli
│   │   ├── __init__.py
│   │   ├── base.py              # Classi base (Siamese, Triplet, Contrastive)
│   │   ├── bce_backbones.py     # Backbone per BCE loss
│   │   ├── contrastive_backbones.py # Backbone per Contrastive
│   │   ├── triplet_backbones.py # Backbone per Triplet
│   │   └── registry.py          # Registry modelli
│   │
│   ├── training/                 # Logica di training
│   │   ├── __init__.py
│   │   ├── base_trainer.py      # Trainer base
│   │   ├── trainer_bce.py       # Trainer BCE
│   │   ├── trainer_contrastive.py # Trainer Contrastive
│   │   ├── trainer_triplet.py   # Trainer Triplet
│   │   └── losses.py            # Funzioni di loss
│   │
│   ├── evaluation/               # Metriche e valutazione
│   │   ├── __init__.py
│   │   ├── metrics.py           # Calcolo metriche
│   │   └── visualization.py     # Grafici e plot
│   │
│   └── utils/                    # Utility generali
│       ├── __init__.py
│       └── helpers.py           # Funzioni helper
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_inference.ipynb
│
├── scripts/                      # Script di training
│   ├── train_siamese.py
│   ├── train_contrastive.py
│   ├── train_triplet.py
│   └── kfold_comparison.py
│
├── data/                         # Directory dati
│   └── CVL_writers/
│       ├── writer_001/
│       └── ...
│
├── results/                      # Risultati esperimenti
│   ├── experiment_1/
│   ├── experiment_2/
│   └── ...
│
├── docs/                         # Documentazione MkDocs
│   ├── index.md
│   ├── getting-started.md
│   └── ...
│
├── tests/                        # Test unitari
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
│
├── requirements.txt              # Dipendenze Python
├── mkdocs.yml                   # Configurazione MkDocs
├── README.md                    # README principale
└── LICENSE                      # Licenza

```

## 🏗️ Moduli Principali

### 1. `src/data/` - Gestione Dataset

#### `base_dataset.py`

Classe base astratta per tutti i dataset.

```python
class BaseWriterDataset(Dataset, ABC):
    """
    Fornisce funzionalità comuni:
    - Caricamento immagini da directory di scrittori
    - Applicazione trasformazioni
    - Gestione minimo immagini per scrittore
    """
```

**Responsabilità:**
- ✅ Caricamento struttura directory
- ✅ Gestione trasformazioni train/test
- ✅ Validazione dati (minimo immagini)
- ✅ Loading immagini con PIL

#### `siamese_dataset.py`

Dataset per Siamese Networks con coppie binarie.

```python
class SiameseDataset(BaseWriterDataset):
    """
    Genera coppie (img1, img2, label):
    - label=1: stessa persona (genuine)
    - label=0: persone diverse (impostor)
    """
```

**Output:**
```python
img1: torch.Tensor  # Shape: (1, 448, 448)
img2: torch.Tensor  # Shape: (1, 448, 448)
label: torch.Tensor # Shape: (1,), values: 0.0 o 1.0
```

#### `contrastive_dataset.py`

Dataset per Contrastive Learning (identico a Siamese ma ratio 50/50).

```python
class ContrastiveDataset(BaseWriterDataset):
    """
    Simile a SiameseDataset ma sempre 50% positive, 50% negative.
    """
```

#### `triplet_dataset.py`

Dataset per Triplet Learning.

```python
class TripletDataset(BaseWriterDataset):
    """
    Genera triple (anchor, positive, negative):
    - anchor: immagine di riferimento
    - positive: stessa persona di anchor
    - negative: persona diversa
    """
```

**Output:**
```python
anchor: torch.Tensor   # Shape: (1, 448, 448)
positive: torch.Tensor # Shape: (1, 448, 448)
negative: torch.Tensor # Shape: (1, 448, 448)
```

#### `dataloader_factory.py`

Factory pattern per creare dataloader in modo unificato.

```python
# Funzioni principali
create_dataloaders(dataset_class, ...)
create_kfold_dataloaders(dataset_class, ...)

# Wrapper per retrocompatibilità
create_siamese_dataloaders(...)
create_contrastive_dataloaders(...)
create_triplet_dataloaders(...)
```

**Vantaggi:**
- ✅ Interfaccia unificata per tutti i dataset
- ✅ Gestione automatica train/test split
- ✅ K-Fold cross-validation built-in
- ✅ Parametri consistenti

#### `transforms.py`

Trasformazioni per training e test.

```python
def get_train_transforms(target_size=448):
    """
    - RandomResizedCrop: augmentation spaziale
    - RandomRotation: rotazione ±15°
    - ToTensor: conversione a tensor
    """

def get_test_transforms(target_size=448):
    """
    - Resize: ridimensionamento fisso
    - ToTensor: conversione a tensor
    """
```

### 2. `src/models/` - Architetture Modelli

#### `base.py`

Classi base per le tre architetture principali.

```python
class BaseSiameseNetwork(nn.Module):
    """
    Architettura Siamese base:
    - encoder: backbone condiviso
    - fc: classificatore per similarità
    - forward(): calcola similarità 0-1
    - get_embedding(): estrae embedding normalizzato
    """

class BaseContrastiveNetwork(nn.Module):
    """
    Architettura Contrastive base:
    - encoder: backbone condiviso
    - projection: testa proiezione MLP
    - forward(): embedding L2-normalizzato
    """

class BaseTripletNetwork(nn.Module):
    """
    Architettura Triplet base:
    - encoder: backbone condiviso
    - fc: proiezione a embedding_dim
    - forward(): embedding L2-normalizzato
    """
```

#### `bce_backbones.py`

Implementazioni concrete per Siamese con BCE loss.

**Modelli disponibili:**
- `SiameseResNet18`, `SiameseResNet34`, `SiameseResNet50`
- `SiameseEfficientNetB0`, `SiameseEfficientNetB1`, `SiameseEfficientNetV2`
- `SiameseMobileNetV3Small`, `SiameseMobileNetV3Large`
- `SiameseDenseNet121`
- `SiameseRegNetY400MF`

**Esempio:**
```python
class SiameseResNet18(BaseSiameseNetwork):
    def __init__(self, in_channels=1, projection_dim=512):
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(in_channels, 64, ...)
        encoder = nn.Sequential(*list(resnet.children())[:-1])
        super().__init__(encoder, feature_dim=512, projection_dim=projection_dim)
```

#### `contrastive_backbones.py`

Implementazioni per Contrastive Learning.

**Modelli disponibili:**
- `ContrastiveMobileNetV3Small`, `ContrastiveMobileNetV3Large`
- `ContrastiveResNet18`, `ContrastiveResNet34`, `ContrastiveResNet50`
- `ContrastiveEfficientNetB0`
- `ContrastiveDenseNet121`

#### `triplet_backbones.py`

Implementazioni per Triplet Learning.

**Modelli disponibili:**
- `TripletMobileNetV3Small`, `TripletMobileNetV3Large`
- `TripletResNet18`, `TripletResNet34`
- `TripletEfficientNetB0`

#### `registry.py`

Sistema di registry per accesso unificato ai modelli.

```python
MODEL_REGISTRY = {
    'resnet18': SiameseResNet18,
    'efficientnet_b0': SiameseEfficientNetB0,
    # ...
}

# Utilizzo
model = get_model('resnet18', in_channels=1, projection_dim=512)
available = list_models()
```

### 3. `src/training/` - Logica di Training

#### `base_trainer.py`

Classe base astratta con logica comune.

**Funzionalità comuni:**
- ✅ Training loop con early stopping
- ✅ Validazione con metriche biometriche complete
- ✅ Salvataggio checkpoint (best/final)
- ✅ Storia training (CSV)
- ✅ K-Fold cross-validation
- ✅ Gestione scheduler

**Metodi astratti (da implementare):**
```python
@abstractmethod
def _setup_optimizer(self): ...

@abstractmethod
def train_epoch(self, train_loader): ...

@abstractmethod
def validate_loss(self, val_loader): ...

@abstractmethod
def _get_embeddings(self, img1, img2): ...
```

**Metodi concreti:**
```python
def train(self, train_loader, val_loader, val_dataset, epochs, patience):
    """Main training loop con early stopping e validazione completa"""

def validate_comprehensive(self, val_dataset, num_pairs=1000):
    """Validazione con tutte le metriche biometriche"""

def train_kfold(self, data_root, n_splits, ...):
    """K-Fold cross-validation"""
```

#### `trainer_bce.py`

Trainer per Siamese Networks con BCE loss.

```python
class BCETrainer(BaseTrainer):
    def __init__(self, model, model_name, device):
        self.criterion = BCELoss()
        self._setup_optimizer()
    
    def train_epoch(self, train_loader):
        """Forward pass + BCE loss + backward"""
    
    def validate_loss(self, val_loader):
        """Calcola BCE loss su validation set"""
    
    def _get_embeddings(self, img1, img2):
        """Estrae embeddings con return_embeddings=True"""
```

**Optimizer:**
- AdamW con learning rate adattivo (3e-5 o 5e-5)
- ReduceLROnPlateau scheduler

#### `trainer_contrastive.py`

Trainer per Contrastive Learning.

```python
class ContrastiveTrainer(BaseTrainer):
    def __init__(self, model, model_name, device, margin=1.0):
        self.criterion = ContrastiveLoss(margin=margin)
        self._setup_optimizer()
```

**Optimizer:**
- AdamW con lr=1e-4
- CosineAnnealingLR scheduler

#### `trainer_triplet.py`

Trainer per Triplet Learning.

```python
class TripletTrainer(BaseTrainer):
    def __init__(self, model, model_name, device, margin=0.5):
        self.criterion = TripletLoss(margin=margin)
        self._setup_optimizer()
    
    def train_epoch(self, train_loader):
        """Forward pass per anchor, positive, negative"""
```

**Optimizer:**
- AdamW con lr=1e-4
- ReduceLROnPlateau scheduler

#### `losses.py`

Funzioni di loss implementate.

```python
class BCELoss(nn.Module):
    """Binary Cross-Entropy per classificazione similarità"""

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss con margin:
    - Positive: minimizza distanza
    - Negative: massimizza distanza (fino a margin)
    """

class TripletLoss(nn.Module):
    """
    Triplet loss con margin:
    - Minimizza: distance(anchor, positive)
    - Massimizza: distance(anchor, negative)
    - Subject to: d(a,p) < d(a,n) - margin
    """

class CombinedLoss(nn.Module):
    """Combina BCE + Contrastive (non usata attualmente)"""
```

### 4. `src/evaluation/` - Metriche e Valutazione

#### `metrics.py`

Calcolo metriche biometriche complete.

```python
def compute_eer(fpr, tpr, thresholds):
    """Calcola Equal Error Rate dalla curva ROC"""

def compute_classification_metrics(y_true, y_scores, threshold):
    """Accuracy, Precision, Recall, F1 a threshold dato"""

def compute_verification_metrics(genuine_dists, impostor_dists):
    """
    Calcola TUTTE le metriche:
    - EER, AUC-ROC
    - Accuracy, Precision, Recall, F1 (@ EER threshold)
    - FAR/FRR a operating points specifici
    - d-prime e decidability
    - Statistiche distribuzioni
    - Curve ROC complete
    """

def print_verification_results(metrics, dataset_name):
    """Stampa formattata di tutte le metriche"""
```

**Metriche calcolate:**

| Metrica | Descrizione | Range |
|---------|-------------|-------|
| **EER** | Equal Error Rate | [0, 1] |
| **AUC** | Area Under ROC Curve | [0, 1] |
| **Accuracy** | Classificazione corretta | [0, 1] |
| **Precision** | True Positive / (TP + FP) | [0, 1] |
| **Recall** | True Positive / (TP + FN) | [0, 1] |
| **F1-Score** | Media armonica Prec/Rec | [0, 1] |
| **d-prime** | Indice discriminabilità | [-∞, +∞] |
| **FAR** | False Accept Rate | [0, 1] |
| **FRR** | False Reject Rate | [0, 1] |

### 5. `src/utils/` - Utility Generali

#### `helpers.py`

Funzioni utility generiche.

```python
def set_seed(seed=42):
    """Imposta seed per riproducibilità"""

def get_device():
    """Rileva CUDA o CPU automaticamente"""

def count_parameters(model):
    """Conta parametri trainabili"""

def format_number(num):
    """Formatta numeri grandi (K, M, B)"""

def print_model_info(model, model_name):
    """Stampa info modello"""

def load_checkpoint(model, checkpoint_path, device):
    """Carica checkpoint in modello"""

def save_checkpoint(model, save_path):
    """Salva checkpoint modello"""

def ensure_dir(directory):
    """Crea directory se non esiste"""
```

## 🔄 Flusso di Esecuzione

### Training Standard

```
1. Inizializzazione
   ├── set_seed(42)
   ├── device = get_device()
   └── model = get_model('resnet18')

2. Preparazione Dati
   ├── create_dataloaders()
   ├── train/test split (sklearn)
   └── Dataset istanziati

3. Setup Trainer
   ├── trainer = BCETrainer(model, ...)
   ├── optimizer setup
   └── scheduler setup

4. Training Loop
   ├── train_epoch()
   │   ├── forward pass
   │   ├── loss calculation
   │   └── backward + optimizer step
   ├── validate_loss()
   ├── scheduler step
   ├── save best checkpoint
   └── early stopping check

5. Validazione Finale
   ├── validate_comprehensive()
   ├── compute_verification_metrics()
   └── save final metrics

6. Salvataggio
   ├── checkpoint (.pth)
   ├── history (CSV)
   └── metrics (CSV)
```

### K-Fold Cross-Validation

```
1. Setup K-Fold
   └── KFold(n_splits=5, shuffle=True, random_state=42)

2. Per ogni Fold
   ├── create_kfold_dataloaders(fold=i)
   ├── reset model weights
   ├── reset optimizer
   ├── train(fold=i)
   └── save fold results

3. Aggregazione
   ├── combine fold histories
   ├── compute mean/std EER
   └── save aggregated results
```

## 📊 Output e Artefatti

### Directory `results/`

```
results/
└── experiment_name/
    ├── model_name_best.pth           # Miglior checkpoint (val loss)
    ├── model_name_final.pth          # Checkpoint finale
    ├── model_name_history.csv        # Storia training
    ├── model_name_final_metrics.csv  # Metriche finali
    │
    # Se K-Fold:
    ├── model_name_fold1_best.pth
    ├── model_name_fold1_final_metrics.csv
    ├── ...
    ├── model_name_fold5_best.pth
    ├── model_name_fold5_final_metrics.csv
    └── model_name_kfold_detailed.csv  # Tutte le fold combinate
```

### Formato CSV

**history.csv:**
```csv
epoch,train_loss,val_loss
1,0.2456,0.1823
2,0.1678,0.1456
...
```

**final_metrics.csv:**
```csv
eer,auc,accuracy,precision,recall,f1,d_prime,decidability,...
0.0245,0.9912,0.9755,0.9782,0.9734,0.9758,3.2456,4.5912,...
```

## 🧪 Testing

### Struttura Test

```
tests/
├── test_data.py          # Test dataset e dataloader
├── test_models.py        # Test architetture
├── test_training.py      # Test trainer
└── test_evaluation.py    # Test metriche
```

### Eseguire Test

```bash
# Tutti i test
pytest tests/

# Test specifico
pytest tests/test_data.py

# Con coverage
pytest --cov=src tests/
```

## 🎯 Best Practices

### 1. Organizzazione Esperimenti

```python
# Usa directory separate per esperimento
results_dir = f'results/exp_{experiment_name}_{timestamp}'
```

### 2. Naming Convention

```python
# Modelli
model_name = f'{backbone}_{learning_paradigm}_{custom_suffix}'
# Esempi: 'resnet18_siamese_v1', 'mobilenet_triplet_margin05'
```

### 3. Reproducibilità

```python
# Sempre all'inizio
set_seed(42)

# Documenta hyperparameters
config = {
    'model': 'resnet18',
    'batch_size': 16,
    'lr': 5e-5,
    'margin': 0.5,
    'seed': 42
}
```

### 4. Gestione Memoria

```python
# Cleanup dopo training
trainer.cleanup()
torch.cuda.empty_cache()
gc.collect()
```

## 📚 Risorse Aggiuntive

- **[Dataset](datasets.md)**: Gestione dati dettagliata
- **[Training](training.md)**: Guide training avanzate
- **[Evaluation](evaluation.md)**: Interpretazione metriche
- **[API Reference](../api/)**: Documentazione API completa

---

**Prossimo:** [Gestione Dataset →](datasets.md)