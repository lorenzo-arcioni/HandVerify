# Gestione Dataset

Guida completa alla preparazione, organizzazione e utilizzo dei dataset per la verifica della calligrafia.

## 📋 Indice

- [Struttura Richiesta](#struttura-richiesta)
- [Dataset Supportati](#dataset-supportati)
- [Preparazione Dati](#preparazione-dati)
- [Classi Dataset](#classi-dataset)
- [Dataloader Factory](#dataloader-factory)
- [Data Augmentation](#data-augmentation)
- [Best Practices](#best-practices)

## 🗂️ Struttura Richiesta

### Formato Standard

Il sistema richiede una struttura gerarchica con **una cartella per scrittore**:

```
data/
└── dataset_name/
    ├── writer_001/
    │   ├── sample_01.png
    │   ├── sample_02.png
    │   ├── sample_03.png
    │   └── ...
    ├── writer_002/
    │   ├── sample_01.png
    │   ├── sample_02.png
    │   └── ...
    └── writer_N/
        └── ...
```

### Requisiti

- ✅ **Minimo 2 immagini** per scrittore (per formare coppie)
- ✅ **Formati supportati**: PNG, JPG, JPEG
- ✅ **Canali**: Grayscale (1 canale) o RGB (convertito automaticamente)
- ✅ **Risoluzione minima consigliata**: 224x224 pixel
- ✅ **Risoluzione ottimale**: 448x448 pixel o superiore

## 📚 Dataset Supportati

### 1. CVL Database

**CVL (Computer Vision Lab) Database** - Dataset pubblico per writer identification.

**Caratteristiche:**
- 👥 **310 scrittori**
- 📄 **7 pagine per scrittore**
- 🖼️ **Immagini full-page**
- 🌍 **Lingua**: Inglese/Tedesco

**Download:**
```bash
# Link: https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/
wget https://cvl.tuwien.ac.at/wp-content/uploads/2014/01/cvl-database-1-1.zip
unzip cvl-database-1-1.zip -d data/cvl_raw
```

**Script di Organizzazione:**

```python
# scripts/organize_cvl.py
import os
import shutil
from pathlib import Path
from PIL import Image

def organize_cvl_database(source_dir, output_dir, crop_size=(448, 448)):
    """
    Organizza CVL database nella struttura richiesta.
    
    Args:
        source_dir: Directory con immagini CVL originali
        output_dir: Directory output organizzata
        crop_size: Dimensione crop per patch
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in Path(source_dir).glob('*.png'):
        # Nome file: cvl-1-1.png (writer_id-page_id.png)
        parts = img_path.stem.split('-')
        writer_id = f"writer_{int(parts[1]):03d}"
        page_id = int(parts[2])
        
        # Crea directory scrittore
        writer_dir = Path(output_dir) / writer_id
        writer_dir.mkdir(exist_ok=True)
        
        # Opzionale: crop in patch multiple
        if crop_size:
            img = Image.open(img_path).convert('L')
            width, height = img.size
            
            # Dividi in patch 2x2
            patch_w, patch_h = width // 2, height // 2
            patches = [
                img.crop((0, 0, patch_w, patch_h)),           # Top-left
                img.crop((patch_w, 0, width, patch_h)),       # Top-right
                img.crop((0, patch_h, patch_w, height)),      # Bottom-left
                img.crop((patch_w, patch_h, width, height))   # Bottom-right
            ]
            
            for i, patch in enumerate(patches):
                patch = patch.resize(crop_size, Image.LANCZOS)
                out_name = f"page{page_id:02d}_patch{i+1}.png"
                patch.save(writer_dir / out_name)
        else:
            # Copia intera immagine
            shutil.copy(img_path, writer_dir / f"page{page_id:02d}.png")
    
    print(f"✓ CVL dataset organizzato in {output_dir}")

# Utilizzo
organize_cvl_database(
    source_dir='data/cvl_raw',
    output_dir='data/CVL_writers',
    crop_size=(448, 448)
)
```

### 2. IAM Handwriting Database

**IAM** - Dataset per handwriting recognition e writer identification.

**Caratteristiche:**
- 👥 **657 scrittori**
- 📄 **1539 pagine**
- 🖼️ **Form-level e line-level images**

**Preparazione:**

```python
# scripts/organize_iam.py
import os
from pathlib import Path
from PIL import Image

def organize_iam_database(forms_dir, output_dir):
    """
    Organizza IAM forms nella struttura richiesta.
    
    Args:
        forms_dir: Directory con form IAM (forms/*.png)
        output_dir: Directory output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for form_path in Path(forms_dir).glob('*.png'):
        # Nome: a01-000u.png (writer_id-form_id.png)
        writer_id = form_path.stem.split('-')[0]
        writer_dir = Path(output_dir) / f"writer_{writer_id}"
        writer_dir.mkdir(exist_ok=True)
        
        # Copia form
        img = Image.open(form_path).convert('L')
        img = img.resize((448, 448), Image.LANCZOS)
        img.save(writer_dir / form_path.name)
    
    print(f"✓ IAM dataset organizzato in {output_dir}")

# Utilizzo
organize_iam_database(
    forms_dir='data/iam_raw/forms',
    output_dir='data/IAM_writers'
)
```

### 3. ICDAR 2013 Writer Identification

**ICDAR 2013** - Competition dataset per writer identification.

**Caratteristiche:**
- 👥 **250 scrittori (training)** + 250 (test)
- 📄 **Multiple pages per scrittore**

### 4. Dataset Personalizzato

Se hai un dataset custom, organizzalo seguendo lo stesso schema:

```python
# scripts/organize_custom.py
import os
import shutil
from pathlib import Path

def organize_custom_dataset(source_dir, output_dir, naming_function):
    """
    Organizza dataset personalizzato.
    
    Args:
        source_dir: Directory con immagini
        output_dir: Directory output
        naming_function: Funzione per estrarre writer_id da filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in Path(source_dir).rglob('*.png'):
        # Usa funzione custom per estrarre writer_id
        writer_id = naming_function(img_path.name)
        
        writer_dir = Path(output_dir) / writer_id
        writer_dir.mkdir(exist_ok=True)
        
        shutil.copy(img_path, writer_dir / img_path.name)
    
    print(f"✓ Dataset custom organizzato in {output_dir}")

# Esempio: filename pattern "writer123_sample01.png"
def extract_writer_id(filename):
    return filename.split('_')[0]  # "writer123"

organize_custom_dataset(
    source_dir='data/my_data_raw',
    output_dir='data/MyDataset_writers',
    naming_function=extract_writer_id
)
```

## 🔧 Classi Dataset

### BaseWriterDataset

Classe base astratta che fornisce funzionalità comuni.

```python
from src.data import BaseWriterDataset

class BaseWriterDataset(Dataset, ABC):
    """
    Classe base per tutti i dataset di verifica.
    
    Args:
        writer_dirs: Lista di directory (una per scrittore)
        train: True per training (con augmentation)
        samples_per_writer: Numero di sample da generare per scrittore
        target_size: Dimensione immagini (ridimensionamento)
        min_images_per_writer: Minimo immagini richieste per scrittore
    """
```

**Funzionalità fornite:**

- ✅ **Caricamento automatico** delle immagini da directory
- ✅ **Validazione** (scarta scrittori con < min_images)
- ✅ **Trasformazioni** train/test automatiche
- ✅ **Loading immagini** ottimizzato con PIL

**Metodi principali:**

```python
def _load_writer_images(self) -> dict:
    """Carica percorsi immagini per ogni scrittore"""

def _load_image(self, path: str) -> torch.Tensor:
    """Carica e trasforma singola immagine"""

@abstractmethod
def __getitem__(self, idx: int):
    """Da implementare nelle sottoclassi"""
```

### SiameseDataset

Dataset per Siamese Networks (coppie binarie).

```python
from src.data import SiameseDataset

dataset = SiameseDataset(
    writer_dirs=['data/CVL_writers/writer_001', ...],
    train=True,
    positive_ratio=0.5,      # 50% coppie positive
    samples_per_writer=100,
    target_size=448
)
```

**Output:**
```python
img1, img2, label = dataset[0]

# img1: torch.Tensor, shape (1, 448, 448)
# img2: torch.Tensor, shape (1, 448, 448)
# label: torch.Tensor, shape (1,), value 0.0 o 1.0
```

**Logica generazione coppie:**

```python
# Positive pair (label=1)
if random.random() < positive_ratio:
    writer = random.choice(writer_ids)
    img1, img2 = random.sample(writer_images[writer], 2)
    label = 1.0

# Negative pair (label=0)
else:
    writer1, writer2 = random.sample(writer_ids, 2)
    img1 = random.choice(writer_images[writer1])
    img2 = random.choice(writer_images[writer2])
    label = 0.0
```

### ContrastiveDataset

Dataset per Contrastive Learning (50/50 ratio fisso).

```python
from src.data import ContrastiveDataset

dataset = ContrastiveDataset(
    writer_dirs=writer_dirs,
    train=True,
    samples_per_writer=100,
    target_size=448
)
```

**Differenza con SiameseDataset:**
- Ratio fisso 50% positive / 50% negative
- Nessun parametro `positive_ratio`

### TripletDataset

Dataset per Triplet Learning (triple anchor-positive-negative).

```python
from src.data import TripletDataset

dataset = TripletDataset(
    writer_dirs=writer_dirs,
    train=True,
    samples_per_writer=100,
    target_size=448
)
```

**Output:**
```python
anchor, positive, negative = dataset[0]

# anchor: torch.Tensor, shape (1, 448, 448)
# positive: torch.Tensor, shape (1, 448, 448), stesso scrittore di anchor
# negative: torch.Tensor, shape (1, 448, 448), scrittore diverso
```

**Logica generazione triple:**

```python
# 1. Scegli scrittore per anchor
anchor_writer = random.choice(writer_ids)

# 2. Anchor e positive dallo stesso scrittore
anchor_img, positive_img = random.sample(
    writer_images[anchor_writer], 2
)

# 3. Negative da scrittore diverso
negative_writer = random.choice(
    [w for w in writer_ids if w != anchor_writer]
)
negative_img = random.choice(writer_images[negative_writer])
```

## 🏭 Dataloader Factory

### Funzioni Generiche

Le funzioni factory creano automaticamente i dataloader con split train/test.

#### `create_dataloaders()`

Funzione generica per qualsiasi dataset.

```python
from src.data import create_dataloaders, SiameseDataset

train_loader, test_loader, train_ds, test_ds = create_dataloaders(
    dataset_class=SiameseDataset,
    data_root='data/CVL_writers',
    batch_size=16,
    num_workers=4,
    test_size=0.2,           # 20% per test
    samples_per_writer=100,
    target_size=448,
    random_state=42,
    # Parametri specifici del dataset
    positive_ratio=0.5       # Solo per SiameseDataset
)
```

**Parametri:**

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `dataset_class` | Class | - | Classe dataset (SiameseDataset, etc.) |
| `data_root` | str | - | Path root con directory scrittori |
| `batch_size` | int | 16 | Dimensione batch |
| `num_workers` | int | 4 | Worker per caricamento parallelo |
| `test_size` | float | 0.2 | Frazione per test (0.0-1.0) |
| `samples_per_writer` | int | 100 | Sample da generare per scrittore |
| `target_size` | int | 448 | Dimensione immagini |
| `random_state` | int | 42 | Seed per split riproducibile |
| `**dataset_kwargs` | dict | - | Parametri aggiuntivi per dataset |

**Ritorna:**

```python
train_loader: DataLoader  # DataLoader training
test_loader: DataLoader   # DataLoader test
train_dataset: Dataset    # Dataset training
test_dataset: Dataset     # Dataset test
```

#### `create_kfold_dataloaders()`

Funzione per K-Fold cross-validation.

```python
from src.data import create_kfold_dataloaders, TripletDataset

train_loader, val_loader, train_ds, val_ds = create_kfold_dataloaders(
    dataset_class=TripletDataset,
    data_root='data/CVL_writers',
    n_splits=5,              # 5 fold
    current_fold=0,          # Fold corrente (0-4)
    batch_size=16,
    num_workers=4,
    samples_per_writer=100,
    target_size=448,
    random_state=42
)
```

**K-Fold Split:**

```
Fold 0: Train=[1,2,3,4], Val=[0]
Fold 1: Train=[0,2,3,4], Val=[1]
Fold 2: Train=[0,1,3,4], Val=[2]
Fold 3: Train=[0,1,2,4], Val=[3]
Fold 4: Train=[0,1,2,3], Val=[4]
```

### Funzioni Specifiche (Backward Compatibility)

Per comodità, sono disponibili wrapper specifici per tipo:

```python
# Siamese
from src.data import create_siamese_dataloaders
train_loader, test_loader, train_ds, test_ds = create_siamese_dataloaders(
    data_root='data/CVL_writers',
    batch_size=16,
    test_size=0.2
)

# Contrastive
from src.data import create_contrastive_dataloaders
train_loader, test_loader, train_ds, test_ds = create_contrastive_dataloaders(
    data_root='data/CVL_writers',
    batch_size=32
)

# Triplet
from src.data import create_triplet_dataloaders
train_loader, test_loader, train_ds, test_ds = create_triplet_dataloaders(
    data_root='data/CVL_writers',
    batch_size=16
)
```

## 🎨 Data Augmentation

### Trasformazioni Training

```python
from src.data.transforms import get_train_transforms

train_transforms = get_train_transforms(target_size=448)

# Composizione:
# 1. RandomResizedCrop(448, scale=(0.9, 1.1))
#    - Crop random con scala 90%-110%
#    - Simula variazioni di dimensione scrittura
#
# 2. RandomRotation(15)
#    - Rotazione casuale ±15 gradi
#    - Simula inclinazione carta
#
# 3. ToTensor()
#    - Converti a Tensor PyTorch
#    - Normalizza in [0, 1]
```

**Esempio di augmentation:**

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('data/CVL_writers/writer_001/sample_01.png')

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    augmented = train_transforms(img)
    ax.imshow(augmented.squeeze(), cmap='gray')
    ax.set_title(f'Augmentation {i+1}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('augmentation_examples.png')
```

### Trasformazioni Test

```python
from src.data.transforms import get_test_transforms

test_transforms = get_test_transforms(target_size=448)

# Composizione:
# 1. Resize((448, 448))
#    - Ridimensionamento fisso (no augmentation)
#
# 2. ToTensor()
#    - Converti a Tensor
```

### Custom Transforms

Per augmentation personalizzata:

```python
import torchvision.transforms as transforms

custom_transforms = transforms.Compose([
    # Augmentation spaziale
    transforms.RandomResizedCrop(448, scale=(0.85, 1.15)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    # Augmentation di intensità
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    
    # Blur occasionale
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    
    # To Tensor
    transforms.ToTensor(),
    
    # Normalizzazione
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

## 📊 Statistiche Dataset

### Script di Analisi

```python
# scripts/dataset_stats.py
import os
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(data_root):
    """Analizza statistiche dataset."""
    
    stats = {
        'writer_id': [],
        'num_images': [],
        'avg_size': []
    }
    
    for writer_dir in Path(data_root).iterdir():
        if not writer_dir.is_dir():
            continue
        
        images = list(writer_dir.glob('*.png'))
        num_images = len(images)
        
        if num_images > 0:
            # Dimensioni medie
            from PIL import Image
            sizes = [Image.open(img).size for img in images[:5]]
            avg_size = sum([w*h for w,h in sizes]) / len(sizes)
            
            stats['writer_id'].append(writer_dir.name)
            stats['num_images'].append(num_images)
            stats['avg_size'].append(avg_size)
    
    df = pd.DataFrame(stats)
    
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS: {data_root}")
    print(f"{'='*60}")
    print(f"Total writers: {len(df)}")
    print(f"Total images: {df['num_images'].sum()}")
    print(f"Avg images per writer: {df['num_images'].mean():.2f}")
    print(f"Min images per writer: {df['num_images'].min()}")
    print(f"Max images per writer: {df['num_images'].max()}")
    print(f"{'='*60}\n")
    
    # Plot distribuzione
    plt.figure(figsize=(10, 6))
    plt.hist(df['num_images'], bins=20, edgecolor='black')
    plt.xlabel('Number of Images')
    plt.ylabel('Number of Writers')
    plt.title('Distribution of Images per Writer')
    plt.grid(True, alpha=0.3)
    plt.savefig('dataset_distribution.png', dpi=150)
    
    return df

# Utilizzo
df = analyze_dataset('data/CVL_writers')
df.to_csv('dataset_stats.csv', index=False)
```

## ✅ Best Practices

### 1. Validazione Dati

Prima di iniziare il training, valida sempre il dataset:

```python
from src.data import create_siamese_dataloaders

try:
    train_loader, test_loader, train_ds, test_ds = create_siamese_dataloaders(
        data_root='data/CVL_writers',
        batch_size=16,
        test_size=0.2
    )
    
    # Test loading
    img1, img2, label = next(iter(train_loader))
    print(f"✓ Batch shape: img1={img1.shape}, img2={img2.shape}, label={label.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
```

### 2. Bilanciamento Classi

Per Siamese/Contrastive, verifica il bilanciamento:

```python
# Conta labels in un batch
labels_list = []
for img1, img2, labels in train_loader:
    labels_list.extend(labels.numpy())

positive_ratio = sum(labels_list) / len(labels_list)
print(f"Positive ratio: {positive_ratio:.2%}")  # Dovrebbe essere ~50%
```

### 3. Dimensione Batch Ottimale

Regole generali:

| GPU VRAM | Target Size | Batch Size | Architettura |
|----------|-------------|------------|--------------|
| 6GB | 224 | 32 | Lightweight (MobileNet) |
| 6GB | 448 | 8-16 | Medium (ResNet18/34) |
| 8GB | 448 | 16-32 | Medium |
| 12GB+ | 448 | 32-64 | Large (ResNet50, EfficientNet) |

### 4. Num Workers

```python
import multiprocessing

# Usa tutti i core disponibili (ma non più del necessario)
num_workers = min(4, multiprocessing.cpu_count())
```

### 5. Preprocessing Consistency

Assicurati che training e test usino la stessa pipeline (eccetto augmentation):

```python
# ✓ CORRETTO
train_ds = SiameseDataset(..., train=True)   # Con augmentation
test_ds = SiameseDataset(..., train=False)   # Senza augmentation

# ✗ SBAGLIATO
# Usare trasformazioni diverse
```

### 6. Memory Management

Per dataset grandi:

```python
# Pin memory per GPU transfer veloce
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    pin_memory=True  # ← Importante per GPU
)

# Cleanup dopo uso
del train_loader
torch.cuda.empty_cache()
```

## 🔍 Troubleshooting

### Problema: "Not enough images per writer"

```bash
# Controlla quante immagini ha ogni scrittore
python scripts/dataset_stats.py
```

**Soluzione:**
- Rimuovi scrittori con < 2 immagini
- Oppure riduci `min_images_per_writer=1` (non consigliato)

### Problema: "CUDA out of memory"

**Soluzioni:**
```python
# 1. Riduci batch size
batch_size = 8  # invece di 16

# 2. Riduci target_size
target_size = 224  # invece di 448

# 3. Usa gradient accumulation (vedi Training docs)
```

### Problema: "DataLoader slow"

**Soluzioni:**
```python
# 1. Aumenta num_workers
num_workers = 8  # invece di 4

# 2. Attiva pin_memory
pin_memory = True

# 3. Preprocesa e salva immagini
# (ridimensionale tutte le immagini offline)
```

## 📚 Risorse Aggiuntive

- **[Training](training.md)**: Uso dei dataloader nel training
- **[Project Structure](project-structure.md)**: Architettura codice dataset
- **[Notebooks](notebooks.md)**: Tutorial interattivi sulla gestione dati

---

**Prossimo:** [Training →](training.md)