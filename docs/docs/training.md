# Training

Guida completa all'addestramento dei modelli di verifica della calligrafia.

## 📋 Indice

- [Paradigmi di Apprendimento](#paradigmi-di-apprendimento)
- [Training Base](#training-base)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Early Stopping](#early-stopping)
- [Learning Rate Scheduling](#learning-rate-scheduling)
- [Best Practices](#best-practices)

## 🎓 Paradigmi di Apprendimento

Il sistema supporta tre approcci principali per la verifica della calligrafia.

### 1. Siamese Networks con BCE Loss

**Concetto:** Due reti identiche (pesi condivisi) che elaborano due immagini e predicono similarità tramite classificazione binaria.

```
         ┌─────────┐
  img1 → │ Encoder │ → feat1 ┐
         └─────────┘          ├→ concat → FC → sigmoid → similarity (0-1)
         ┌─────────┐          │
  img2 → │ Encoder │ → feat2 ┘
         └─────────┘
```

**Loss Function:**
```python
BCE_loss = -[y * log(p) + (1-y) * log(1-p)]
```
dove:
- `y` = label (1 per stesso scrittore, 0 per diverso)
- `p` = similarity prediction (0-1)

**Vantaggi:**
- ✅ Interpretabile (output come probabilità)
- ✅ Training stabile
- ✅ Buone performance generali

**Svantaggi:**
- ❌ Non ottimizza direttamente lo spazio metrico

**Quando usarlo:**
- Dataset bilanciati
- Quando serve output probabilistico
- Baseline per confronti

### 2. Contrastive Learning

**Concetto:** Apprendimento metrico che minimizza distanza per coppie simili e massimizza per coppie diverse.

```
         ┌─────────┐
  img1 → │ Encoder │ → emb1 ┐
         └─────────┘         ├→ distance → loss
         ┌─────────┐         │
  img2 → │ Encoder │ → emb2 ┘
         └─────────┘
```

**Loss Function:**
```python
L = y * d² + (1-y) * max(margin - d, 0)²
```
dove:
- `d` = distanza euclidea tra embedding
- `margin` = margine per coppie diverse
- `y` = 1 se simili, 0 se diversi

**Vantaggi:**
- ✅ Ottimizza direttamente lo spazio metrico
- ✅ Embedding compatti e discriminativi
- ✅ Buono per pochi esempi per classe

**Svantaggi:**
- ❌ Richiede tuning del margin
- ❌ Convergenza più lenta

**Quando usarlo:**
- Quando serve spazio embedding di qualità
- Few-shot learning
- Transfer learning

### 3. Triplet Learning

**Concetto:** Apprendimento con triple (anchor, positive, negative) che ottimizza distanze relative.

```
           ┌─────────┐
  anchor → │ Encoder │ → emb_a ┐
           └─────────┘          │
           ┌─────────┐          ├→ triplet_loss
positive → │ Encoder │ → emb_p ─┤
           └─────────┘          │
           ┌─────────┐          │
negative → │ Encoder │ → emb_n ┘
           └─────────┘
```

**Loss Function:**
```python
L = max(d(a,p) - d(a,n) + margin, 0)
```
dove:
- `d(a,p)` = distanza anchor-positive
- `d(a,n)` = distanza anchor-negative
- Obiettivo: `d(a,p) < d(a,n) - margin`

**Vantaggi:**
- ✅ Ottimizzazione diretta delle distanze relative
- ✅ Stato dell'arte per embedding learning
- ✅ Robusto a dataset sbilanciati

**Svantaggi:**
- ❌ Triplet selection critico
- ❌ Training più complesso
- ❌ Convergenza lenta senza hard mining

**Quando usarlo:**
- Massima qualità embedding
- Dataset grandi
- Verifica ad alte prestazioni

## 🚀 Training Base

### Training con BCE (Siamese)

```python
# scripts/train_siamese.py
from src.utils import set_seed, get_device, print_model_info
from src.models import get_model
from src.data import create_siamese_dataloaders
from src.training import BCETrainer

# ===== CONFIGURAZIONE =====
CONFIG = {
    'seed': 42,
    'data_root': 'data/CVL_writers',
    'model_name': 'resnet18',
    'batch_size': 16,
    'num_workers': 4,
    'test_size': 0.2,
    'samples_per_writer': 100,
    'target_size': 448,
    'epochs': 50,
    'patience': 7,
    'results_dir': 'results/siamese_exp1'
}

# ===== SETUP =====
set_seed(CONFIG['seed'])
device = get_device()

# ===== DATI =====
print("📁 Loading data...")
train_loader, val_loader, train_ds, val_ds = create_siamese_dataloaders(
    data_root=CONFIG['data_root'],
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
    test_size=CONFIG['test_size'],
    samples_per_writer=CONFIG['samples_per_writer'],
    target_size=CONFIG['target_size'],
    random_state=CONFIG['seed']
)

# ===== MODELLO =====
print("\n🏗️ Initializing model...")
model = get_model(
    CONFIG['model_name'],
    in_channels=1,
    projection_dim=512
)
print_model_info(model, CONFIG['model_name'])

# ===== TRAINER =====
trainer = BCETrainer(
    model=model,
    model_name=f"{CONFIG['model_name']}_siamese",
    device=device,
    results_dir=CONFIG['results_dir']
)

# ===== TRAINING =====
print("\n🚀 Starting training...\n")
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=CONFIG['epochs'],
    patience=CONFIG['patience']
)

# ===== RISULTATI =====
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Best Val Loss: {trainer.best_loss:.4f}")
print(f"Final EER: {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
print(f"Final AUC: {metrics['auc']:.4f}")
print(f"Final Accuracy: {metrics['accuracy']:.4f}")
print(f"d-prime: {metrics['d_prime']:.4f}")
print("="*70)

# Cleanup
trainer.cleanup()
```

### Training con Contrastive Loss

```python
# scripts/train_contrastive.py
from src.models.contrastive_backbones import ContrastiveMobileNetV3Small
from src.data import create_contrastive_dataloaders
from src.training import ContrastiveTrainer

# Setup
set_seed(42)
device = get_device()

# Dati
train_loader, val_loader, train_ds, val_ds = create_contrastive_dataloaders(
    data_root='data/CVL_writers',
    batch_size=32,  # Batch più grande per contrastive
    test_size=0.2
)

# Modello
model = ContrastiveMobileNetV3Small(
    in_channels=1,
    projection_dim=128  # Embedding più compatto
)

# Trainer
trainer = ContrastiveTrainer(
    model=model,
    model_name='mobilenetv3_contrastive',
    device=device,
    margin=1.0,  # Hyperparameter importante!
    results_dir='results/contrastive_exp1'
)

# Training
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=50,
    patience=7
)
```

**Tuning del Margin:**
```python
# Margin piccolo (0.5): più strict, embedding compatti
# Margin grande (2.0): più permissivo, embedding sparsi
# Default ottimale: 1.0
```

### Training con Triplet Loss

```python
# scripts/train_triplet.py
from src.models.triplet_backbones import TripletResNet18
from src.data import create_triplet_dataloaders
from src.training import TripletTrainer

# Setup
set_seed(42)
device = get_device()

# Dati
train_loader, val_loader, train_ds, val_ds = create_triplet_dataloaders(
    data_root='data/CVL_writers',
    batch_size=16,
    samples_per_writer=100
)

# Modello
model = TripletResNet18(
    in_channels=1,
    embedding_dim=128  # Dimensione embedding
)

# Trainer
trainer = TripletTrainer(
    model=model,
    model_name='resnet18_triplet',
    device=device,
    margin=0.5,  # Margin per triplet loss
    results_dir='results/triplet_exp1'
)

# Training
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=50,
    patience=7
)
```

## 🔄 K-Fold Cross-Validation

La K-Fold cross-validation fornisce una stima più robusta delle performance.

### Perché K-Fold?

- ✅ **Stima più affidabile**: Media su K fold invece di singolo split
- ✅ **Uso completo dei dati**: Ogni sample è usato per training e validation
- ✅ **Rilevamento overfitting**: Varianza alta tra fold indica overfitting
- ✅ **Pubblicazioni**: Standard per confronti scientifici

### K-Fold con BCE

```python
# scripts/kfold_siamese.py
from src.utils import set_seed, get_device
from src.models import get_model
from src.training import BCETrainer

# Setup
set_seed(42)
device = get_device()

# Modello
model = get_model('resnet18', in_channels=1, projection_dim=512)

# Trainer
trainer = BCETrainer(
    model=model,
    model_name='resnet18_siamese',
    device=device,
    results_dir='results/kfold_siamese'
)

# K-Fold Training
aggregated_results = trainer.train_kfold(
    data_root='data/CVL_writers',
    n_splits=5,              # 5 fold
    batch_size=16,
    num_workers=4,
    samples_per_writer=100,
    target_size=448,
    epochs=50,
    patience=7,
    random_state=42
)

# Risultati aggregati
print(f"\n{'='*70}")
print("K-FOLD RESULTS")
print(f"{'='*70}")
print(f"Mean EER: {aggregated_results['mean_eer']:.4f}")
print(f"Std EER: {aggregated_results['std_eer']:.4f}")
print(f"All Folds: {aggregated_results['all_folds_eer']}")
print(f"{'='*70}\n")
```

**Output:**
```
==================================================================
K-FOLD RESULTS
==================================================================
Mean EER: 0.0245
Std EER: 0.0032
All Folds: [0.0234, 0.0256, 0.0241, 0.0239, 0.0254]
==================================================================
```

### K-Fold con Contrastive

```python
from src.models.contrastive_backbones import ContrastiveResNet18
from src.training import ContrastiveTrainer

model = ContrastiveResNet18(in_channels=1, projection_dim=128)

trainer = ContrastiveTrainer(
    model=model,
    model_name='resnet18_contrastive',
    device=device,
    margin=1.0
)

aggregated = trainer.train_kfold(
    data_root='data/CVL_writers',
    n_splits=5,
    batch_size=32,
    epochs=50
)
```

### K-Fold con Triplet

```python
from src.models.triplet_backbones import TripletMobileNetV3Large
from src.training import TripletTrainer

model = TripletMobileNetV3Large(in_channels=1, embedding_dim=128)

trainer = TripletTrainer(
    model=model,
    model_name='mobilenet_triplet',
    device=device,
    margin=0.5
)

aggregated = trainer.train_kfold(
    data_root='data/CVL_writers',
    n_splits=5,
    batch_size=16,
    epochs=50
)
```

### Confronto Multi-Modello K-Fold

```python
# scripts/kfold_comparison.py
from src.utils import set_seed, get_device
from src.models import list_models, get_model
from src.training import BCETrainer
import pandas as pd

MODELS = ['resnet18', 'efficientnet_b0', 'mobilenet_v3_small']
RESULTS = []

for model_name in MODELS:
    print(f"\n{'#'*70}")
    print(f"# Training: {model_name}")
    print(f"{'#'*70}\n")
    
    # Setup
    set_seed(42)
    device = get_device()
    model = get_model(model_name, in_channels=1, projection_dim=512)
    
    # Trainer
    trainer = BCETrainer(
        model=model,
        model_name=model_name,
        device=device,
        results_dir=f'results/comparison_{model_name}'
    )
    
    # K-Fold
    agg = trainer.train_kfold(
        data_root='data/CVL_writers',
        n_splits=5,
        batch_size=16,
        epochs=50,
        patience=7
    )
    
    RESULTS.append({
        'model': model_name,
        'mean_eer': agg['mean_eer'],
        'std_eer': agg['std_eer'],
        'folds': agg['all_folds_eer']
    })
    
    # Cleanup
    trainer.cleanup()

# Salva risultati
df = pd.DataFrame(RESULTS)
df.to_csv('results/kfold_comparison.csv', index=False)

# Stampa tabella
print(f"\n{'='*70}")
print("COMPARISON RESULTS")
print(f"{'='*70}")
print(df[['model', 'mean_eer', 'std_eer']])
print(f"{'='*70}\n")
```

## 🎛️ Hyperparameter Tuning

### Parametri Principali

#### 1. Batch Size

**Impatto:**
- **Piccolo (8-16)**: Convergenza più rumorosa, generalizzazione migliore
- **Grande (32-64)**: Convergenza stabile, rischio overfitting

**Raccomandazioni:**
```python
# Lightweight models (MobileNet)
batch_size = 32  # o 64 con GPU grande

# Medium models (ResNet18/34)
batch_size = 16  # o 32

# Large models (ResNet50, EfficientNet)
batch_size = 8   # o 16 con GPU grande
```

#### 2. Learning Rate

**Impatto:**
- **Alto (1e-3)**: Convergenza veloce, instabilità
- **Basso (1e-5)**: Convergenza lenta, stabile

**Raccomandazioni:**
```python
# BCE Trainer
lr = 5e-5  # per modelli < 15M params
lr = 3e-5  # per modelli > 15M params

# Contrastive/Triplet Trainer
lr = 1e-4  # Default ottimale
```

**Grid Search:**
```python
for lr in [1e-5, 5e-5, 1e-4, 5e-4]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    # Train e valuta...
```

#### 3. Margin (Contrastive/Triplet)

**Contrastive Loss:**
```python
# margin = 0.5: embedding molto compatti
# margin = 1.0: bilanciato (default)
# margin = 2.0: embedding sparsi

for margin in [0.5, 1.0, 1.5, 2.0]:
    trainer = ContrastiveTrainer(..., margin=margin)
    # Train e valuta...
```

**Triplet Loss:**
```python
# margin = 0.2: strict separation
# margin = 0.5: bilanciato (default)
# margin = 1.0: relaxed separation

for margin in [0.2, 0.5, 1.0]:
    trainer = TripletTrainer(..., margin=margin)
    # Train e valuta...
```

#### 4. Projection Dimension

**BCE (Siamese):**
```python
# projection_dim = 256: compatto, più veloce
# projection_dim = 512: bilanciato (default)
# projection_dim = 1024: espressivo, più lento

model = get_model('resnet18', projection_dim=512)
```

**Contrastive/Triplet:**
```python
# embedding_dim = 64: molto compatto
# embedding_dim = 128: bilanciato (default)
# embedding_dim = 256: espressivo

model = ContrastiveResNet18(projection_dim=128)
```

### Grid Search Completo

```python
# scripts/grid_search.py
import itertools
from src.utils import set_seed, get_device
from src.models import get_model
from src.training import BCETrainer

# Definisci griglia
GRID = {
    'model': ['resnet18', 'efficientnet_b0'],
    'batch_size': [16, 32],
    'lr': [3e-5, 5e-5],
    'projection_dim': [256, 512]
}

# Genera tutte le combinazioni
combinations = list(itertools.product(*GRID.values()))

RESULTS = []

for i, (model_name, bs, lr, proj_dim) in enumerate(combinations):
    print(f"\n[{i+1}/{len(combinations)}] Training: {model_name}, bs={bs}, lr={lr}, proj={proj_dim}")
    
    set_seed(42)
    device = get_device()
    
    # Modello
    model = get_model(model_name, projection_dim=proj_dim)
    
    # Trainer con lr custom
    trainer = BCETrainer(model=model, model_name=model_name, device=device)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training rapido (poche epoche)
    history, metrics = trainer.train(
        train_loader=train_loader,  # Pre-caricato
        val_loader=val_loader,
        val_dataset=val_ds,
        epochs=20,  # Ridotto per grid search
        patience=5
    )
    
    RESULTS.append({
        'model': model_name,
        'batch_size': bs,
        'lr': lr,
        'projection_dim': proj_dim,
        'eer': metrics['eer'],
        'auc': metrics['auc']
    })
    
    trainer.cleanup()

# Salva e analizza
df = pd.DataFrame(RESULTS)
df.to_csv('results/grid_search_results.csv', index=False)

# Best configuration
best = df.loc[df['eer'].idxmin()]
print(f"\n{'='*70}")
print("BEST CONFIGURATION")
print(f"{'='*70}")
print(best)
print(f"{'='*70}\n")
```

## ⏱️ Early Stopping

Early stopping è implementato automaticamente in tutti i trainer.

### Funzionamento

```python
# Parametri
patience = 7  # Numero di epoche senza miglioramento
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        save_checkpoint(is_best=True)
    else:
        patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
```

### Configurazione

```python
# Pazienza bassa: training più veloce, rischio underfit
trainer.train(..., patience=3)

# Pazienza media: bilanciato (default)
trainer.train(..., patience=7)

# Pazienza alta: training più lungo, rischio overfit
trainer.train(..., patience=15)
```

**Raccomandazioni:**
- Dataset piccoli (<50 writers): `patience=5`
- Dataset medi (50-200 writers): `patience=7`
- Dataset grandi (>200 writers): `patience=10`

## 📉 Learning Rate Scheduling

### ReduceLROnPlateau (BCE, Triplet)

Riduce LR quando validation loss plateau.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimizza val_loss
    factor=0.5,      # LR = LR * 0.5
    patience=3,      # Dopo 3 epoche senza miglioramento
    min_lr=1e-7      # LR minimo
)

# Nel training loop
for epoch in range(epochs):
    train_loss = train_epoch()
    val_loss = validate()
    scheduler.step(val_loss)  # Aggiorna LR
```

**Output tipico:**
```
Epoch 10: LR = 5e-5
Epoch 15: LR = 2.5e-5  (ridotto dopo plateau)
Epoch 20: LR = 1.25e-5 (ridotto di nuovo)
```

### CosineAnnealingLR (Contrastive)

Riduce LR seguendo un coseno.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,        # Numero totale epoche
    eta_min=1e-6     # LR minimo
)

# Nel training loop
for epoch in range(epochs):
    train_loss = train_epoch()
    scheduler.step()  # Aggiorna LR (no val_loss)
```

**Curva LR:**
```
Epoch  0: LR = 1e-4
Epoch 12: LR = 7.5e-5
Epoch 25: LR = 5e-5
Epoch 37: LR = 2.5e-5
Epoch 50: LR = 1e-6
```

### Custom Scheduler

```python
# Warmup + Cosine Decay
def get_lr_schedule(optimizer, warmup_epochs=5, total_epochs=50):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Utilizzo
scheduler = get_lr_schedule(optimizer, warmup_epochs=5, total_epochs=50)
```

## ✅ Best Practices

### 1. Riproducibilità

```python
# SEMPRE all'inizio
from src.utils import set_seed
set_seed(42)

# Documenta configurazione
CONFIG = {
    'seed': 42,
    'model': 'resnet18',
    'batch_size': 16,
    'lr': 5e-5,
    'epochs': 50,
    'patience': 7
}

# Salva config
import json
with open('results/config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
```

### 2. Monitoring

```python
# Usa progress bar (già integrato nei trainer)
from tqdm import tqdm

# Log a file
import logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logging.info(f"Starting training: {model_name}")
```

### 3. Checkpoint Management

```python
# I trainer salvano automaticamente:
# - best_model.pth: miglior val_loss
# - final_model.pth: ultimo checkpoint

# Per caricare:
from src.utils import load_checkpoint
model = load_checkpoint(
    model,
    'results/exp1/resnet18_best.pth',
    device
)
```

### 4. Gradient Accumulation

Per batch size effettivo maggiore con memoria limitata:

```python
accumulation_steps = 4  # Batch effettivo = 16 * 4 = 64

for i, (img1, img2, labels) in enumerate(train_loader):
    outputs = model(img1, img2)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # Scala loss
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Mixed Precision Training

Per training più veloce con GPU Tensor Core:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for img1, img2, labels in train_loader:
    optimizer.zero_grad()
    
    with autocast():  # Mixed precision
        outputs = model(img1, img2)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 6. Data Parallel (Multi-GPU)

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)
```

## 📊 Interpretazione Output

### Training Progress

```
Epoch 1/50 - Train Loss: 0.2456 | Val Loss: 0.1823
  ✓ Saved best model (Val Loss=0.1823)

Epoch 2/50 - Train Loss: 0.1678 | Val Loss: 0.1456
  ✓ Saved best model (Val Loss=0.1456)

Epoch 3/50 - Train Loss: 0.1234 | Val Loss: 0.1389
  ✓ Saved best model (Val Loss=0.1389)

...

Epoch 15/50 - Train Loss: 0.0456 | Val Loss: 0.1401
  Patience: 3/7
```

**Cosa guardare:**
- **Train Loss decresce**: ✅ Modello sta imparando
- **Val Loss decresce**: ✅ Generalizzazione buona
- **Train Loss << Val Loss**: ⚠️ Possibile overfitting
- **Entrambi plateau**: ⚠️ Convergenza o LR troppo basso

### Final Metrics

```
======================================================================
FINAL VALIDATION METRICS
======================================================================

🎯 PRIMARY BIOMETRIC METRICS:
  EER:              0.0245 (2.45%)     ← Metrica principale
  AUC:              0.9912              ← Qualità classificazione
  EER Threshold:    0.3456              ← Threshold operativo

📊 CLASSIFICATION METRICS (@ EER threshold):
  Accuracy:         0.9755              ← % classificazione corretta
  Precision:        0.9782              ← % genuine corretti
  Recall:           0.9734              ← % genuine trovati
  F1-Score:         0.9758              ← Media armonica

📈 DISCRIMINABILITY:
  d-prime (d'):     3.2456              ← Separabilità distribuzioni
  Decidability:     4.5912              ← d' * sqrt(2)
```

**Target:**
- EER < 5%: ✅ Buono
- EER < 2%: ✅✅ Eccellente
- AUC > 0.95: ✅ Buono
- AUC > 0.99: ✅✅ Eccellente

## 📚 Risorse Aggiuntive

- **[Evaluation](evaluation.md)**: Interpretazione metriche dettagliata
- **[Datasets](datasets.md)**: Preparazione dati per training
- **[Project Structure](project-structure.md)**: Architettura trainer
- **[Notebooks](notebooks.md)**: Tutorial interattivi

---

**Prossimo:** [Evaluation →](evaluation.md)