# Sistema di Verifica della Calligrafia

## Panoramica

Benvenuto nella documentazione del **Sistema di Verifica della Calligrafia**! Questo progetto implementa un framework completo per la verifica dello scrittore utilizzando tecniche di deep learning. Il sistema è progettato per determinare se due campioni di scrittura a mano provengono dalla stessa persona, un compito critico nell'analisi forense dei documenti, verifica delle firme e autenticazione biometrica.

## 🎯 Caratteristiche Principali

- **Paradigmi di Apprendimento Multipli**: Supporto per Siamese Networks (BCE), Contrastive Learning e Triplet Learning
- **Architettura Flessibile**: Oltre 10 backbone CNN pre-addestrate (ResNet, EfficientNet, MobileNet, DenseNet, RegNet)
- **Valutazione Completa**: Metriche biometriche inclusi EER, AUC, FAR/FRR e d-prime
- **Production-Ready**: K-Fold cross-validation, early stopping e gestione checkpoint
- **Design Modulare**: Separazione netta tra dati, modelli, training e valutazione

## 🚀 Quick Start

### Installazione

```bash
# Clona il repository
git clone https://github.com/yourusername/handwriting-verification.git
cd handwriting-verification

# Installa le dipendenze
pip install -r requirements.txt
```

### Utilizzo Base

```python
from src.models import get_model
from src.training import BCETrainer
from src.data import create_siamese_dataloaders
from src.utils import get_device, set_seed

# Setup
set_seed(42)
device = get_device()

# Crea i dataloader
train_loader, val_loader, train_ds, val_ds = create_siamese_dataloaders(
    data_root='data/CVL_writers',
    batch_size=16,
    test_size=0.2
)

# Inizializza il modello
model = get_model('resnet18', in_channels=1, projection_dim=512)

# Crea il trainer
trainer = BCETrainer(
    model=model,
    model_name='resnet18_siamese',
    device=device
)

# Addestra
history, metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    val_dataset=val_ds,
    epochs=50,
    patience=7
)
```

## 📊 Risultati

Il sistema raggiunge prestazioni state-of-the-art su diversi dataset di riferimento:

| Modello | Dataset | EER | AUC |
|---------|---------|-----|-----|
| ResNet18 | CVL | 0.0245 | 0.9912 |
| EfficientNet-B0 | CVL | 0.0198 | 0.9935 |
| MobileNetV3 | CVL | 0.0312 | 0.9876 |

Per risultati dettagliati, consulta la sezione [Risultati](results.md).

## 📚 Struttura della Documentazione

- **[Getting Started](getting-started.md)**: Guida passo-passo per iniziare
- **[Struttura del Progetto](project-structure.md)**: Organizzazione del codice
- **[Dataset](datasets.md)**: Preparazione e gestione dei dati
- **[Training](training.md)**: Addestramento e ottimizzazione
- **[Evaluation](evaluation.md)**: Metriche e valutazione
- **[Notebooks](notebooks.md)**: Tutorial interattivi
- **[Risultati](results.md)**: Benchmark e analisi

## 🔬 Metodologia

Il sistema implementa tre approcci principali:

### 1. Siamese Networks (BCE Loss)
Reti neurali siamesi che apprendono una funzione di similarità attraverso classificazione binaria.

### 2. Contrastive Learning
Apprendimento metrico che spinge le coppie simili ad essere vicine nello spazio di embedding.

### 3. Triplet Learning
Apprendimento con triple (anchor, positive, negative) per ottimizzare direttamente la distanza relativa.

## 🛠️ Requisiti di Sistema

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (consigliato per GPU)
- 8GB RAM minimo (16GB consigliato)
- 4GB VRAM GPU (per training)

## 📖 Citazione

Se utilizzi questo progetto nella tua ricerca, per favore cita:

```bibtex
@software{handwriting_verification_2025,
  title={Handwriting Verification System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/handwriting-verification}
}
```

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file [LICENSE](../LICENSE) per dettagli.

## 🤝 Contributi

I contributi sono benvenuti! Per favore leggi la [guida ai contributi](../CONTRIBUTING.md) prima di iniziare.

## 📧 Contatti

Per domande o supporto:
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/handwriting-verification/issues)
- Discussioni: [GitHub Discussions](https://github.com/yourusername/handwriting-verification/discussions)

---

**Nota**: Questa documentazione è in continuo aggiornamento. Ultima revisione: Dicembre 2024.