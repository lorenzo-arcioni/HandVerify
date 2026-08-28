# %% [markdown]
# # Final Validation — Best Model on New Dataset
#
# Questo notebook:
# 1. Scarica il checkpoint (.pth) del modello migliore da un link GitHub (raw).
# 2. Ricostruisce l'architettura esatta con cui è stato addestrato.
# 3. Costruisce un validation set ESAUSTIVO sul nuovo dataset: TUTTE le
#    coppie genuine + TUTTE le coppie impostor, senza sotto-campionare per
#    rispettare un positive_ratio (a differenza del training/validation
#    "standard" usato negli esperimenti precedenti).
# 4. Calcola le metriche complete (EER, AUC, FAR/FRR, d-prime, ecc.) con
#    la stessa pipeline (`BaseTrainer.validate_comprehensive`) usata negli
#    esperimenti originali, per restare comparabile con le tabelle già
#    prodotte in `biometric_results_analysis.py`.
#
# ⚠️ Compila i placeholder nella cella CONFIG prima di eseguire.

# %%
import sys
sys.path.append('/kaggle/working/HandVerify')  # <-- placeholder: path del repo

import os
import urllib.request

import torch
import numpy as np

from src.models import get_model
from src.training import BCETrainer, ContrastiveTrainer, TripletTrainer
from src.data import create_dataloaders, SiameseDataset, ContrastiveDataset, TripletDataset
from src.evaluation import print_results
from src.utils import set_seed, get_device, ensure_dir

# %% [markdown]
# ## CONFIG — da compilare in base al modello scelto (vedi analisi cross-dataset)

# %%
RANDOM_STATE = 42

# --- Checkpoint del modello migliore (caricato a mano su GitHub) ---
# Esempio: "https://raw.githubusercontent.com/<user>/<repo>/main/checkpoints/resnet34_contrastive_best.pth"
CHECKPOINT_URL = "<PLACEHOLDER: raw.githubusercontent.com URL del .pth>"
CHECKPOINT_LOCAL_PATH = "/kaggle/working/best_model.pth"

# --- Architettura / loss del modello scelto (deve combaciare col training originale) ---
MODEL_TYPE = "<PLACEHOLDER: 'bce' | 'contrastive' | 'triplet'>"
MODEL_NAME = "<PLACEHOLDER: es. 'resnet34', 'mobilenet_v3_small', ...>"

# Iperparametri architetturali usati in fase di training del checkpoint
# (devono essere IDENTICI, altrimenti load_state_dict fallisce o è inconsistente)
MODEL_HPARAMS = {
    'in_channels': 1,
    'freeze_backbone_layers': 3,   # <-- placeholder: quello usato nel training originale
    'dropout': 0.2,                # <-- placeholder
    # BCE usa 'projection_dim', Contrastive/Triplet usano 'embedding_dim':
    # get_model si occupa già di mappare il nome giusto.
    'embedding_dim': 32,           # <-- placeholder (o projection_dim, vedi sopra)
}

# --- Dataset target su cui validare (quello NUOVO, non ancora deciso) ---
TARGET_DATASET_PATH = "<PLACEHOLDER: path del nuovo dataset con sottocartelle per writer>"
TARGET_DATASET_NAME = "<PLACEHOLDER: nome breve, es. 'new_dataset'>"

TARGET_SIZE = 448
NUM_WORKERS = 4
RESULTS_DIR = f"/kaggle/working/final_validation_{TARGET_DATASET_NAME}"

# Mappa loss -> classe Dataset e classe Trainer
DATASET_CLASS_BY_TYPE = {
    'bce': SiameseDataset,
    'contrastive': ContrastiveDataset,
    'triplet': TripletDataset,
}
TRAINER_CLASS_BY_TYPE = {
    'bce': BCETrainer,
    'contrastive': ContrastiveTrainer,
    'triplet': TripletTrainer,
}

# %%
set_seed(RANDOM_STATE)
device = get_device()
ensure_dir(RESULTS_DIR)

# %% [markdown]
# ## 1. Download del checkpoint da GitHub

# %%
if not os.path.exists(CHECKPOINT_LOCAL_PATH):
    print(f"⬇️  Downloading checkpoint from:\n  {CHECKPOINT_URL}")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_LOCAL_PATH)
    print(f"✓ Saved to {CHECKPOINT_LOCAL_PATH}")
else:
    print(f"✓ Checkpoint già presente in {CHECKPOINT_LOCAL_PATH}")

# %% [markdown]
# ## 2. Ricostruzione dell'architettura + caricamento pesi

# %%
model = get_model(
    MODEL_NAME,
    model_type=MODEL_TYPE,
    **MODEL_HPARAMS
)

state_dict = torch.load(CHECKPOINT_LOCAL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print(f"✓ Loaded {MODEL_TYPE.upper()} / {MODEL_NAME} from checkpoint")

# %% [markdown]
# ## 3. Dataset di validazione ESAUSTIVO (tutte le genuine + tutte le impostor)
#
# Trucco: le classi dataset del progetto (`BaseWriterDataset` e sottoclassi)
# usano sempre TUTTE le coppie genuine, ma campionano un sottoinsieme di
# impostor per rispettare `positive_ratio`. Per ottenere il protocollo
# "esaustivo" (nessun campionamento) basta passare un `positive_ratio`
# estremamente piccolo: il numero di impostor "necessari" per quel ratio
# supera sempre la dimensione del pool disponibile, quindi il codice
# ripiega automaticamente sull'uso di TUTTE le impostor pairs
# (vedi `_resample_negatives` in `base_dataset.py`).
#
# Per TripletDataset la stessa logica si applica a `_create_validation_pairs`.

# %%
EXHAUSTIVE_POSITIVE_RATIO = 1e-9  # forza l'uso di tutte le impostor pairs

dataset_class = DATASET_CLASS_BY_TYPE[MODEL_TYPE]

_, val_loader, _, val_dataset = create_dataloaders(
    dataset_class,
    data_root=TARGET_DATASET_PATH,
    batch_size=16,
    num_workers=NUM_WORKERS,
    test_size=1.0,  # tutto il dataset target va in validazione, niente training
    target_size=TARGET_SIZE,
    random_state=RANDOM_STATE,
    positive_ratio=EXHAUSTIVE_POSITIVE_RATIO,
    resample_negatives_every_n_epochs=0,  # nessun resampling: è già "tutto"
)

n_genuine = len(val_dataset.all_genuine_pairs)
if hasattr(val_dataset, 'validation_pairs'):
    n_total = len(val_dataset.validation_pairs)
else:
    n_total = len(val_dataset.samples)
n_impostor = n_total - n_genuine

print(f"\n📊 Protocollo di validazione esaustivo su '{TARGET_DATASET_NAME}':")
print(f"  Genuine pairs:  {n_genuine}")
print(f"  Impostor pairs: {n_impostor}")
print(f"  Totale:         {n_total}")

# %% [markdown]
# ## 4. Validazione completa (riusa `BaseTrainer.validate_comprehensive`)
#
# Istanziamo il Trainer solo per riusare la logica di valutazione già
# scritta e testata (compute_metrics, gestione BCE vs embeddings). Non
# chiamiamo mai `.train()`.

# %%
trainer_class = TRAINER_CLASS_BY_TYPE[MODEL_TYPE]

trainer_kwargs = dict(
    model=model,
    model_name=f"{MODEL_NAME}_{MODEL_TYPE}_{TARGET_DATASET_NAME}_exhaustive",
    device=device,
    results_dir=RESULTS_DIR,
)
if MODEL_TYPE == 'contrastive':
    trainer_kwargs['margin'] = 1.0   # placeholder: stesso margin del training originale
elif MODEL_TYPE == 'triplet':
    trainer_kwargs['margin'] = 0.5   # placeholder: stesso margin del training originale

trainer = trainer_class(**trainer_kwargs)

# %%
metrics = trainer.validate_comprehensive(val_dataset)
print_results(metrics, dataset_name=f"{TARGET_DATASET_NAME} (exhaustive protocol)")

# %% [markdown]
# ## 5. Salvataggio risultati

# %%
import pandas as pd

metrics_to_save = {}
for key, value in metrics.items():
    if isinstance(value, np.ndarray):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        metrics_to_save[key] = str(value.tolist())
    else:
        metrics_to_save[key] = value

out_path = os.path.join(RESULTS_DIR, f"{TARGET_DATASET_NAME}_exhaustive_metrics.csv")
pd.DataFrame([metrics_to_save]).to_csv(out_path, index=False)
print(f"\n✓ Metriche salvate in {out_path}")
