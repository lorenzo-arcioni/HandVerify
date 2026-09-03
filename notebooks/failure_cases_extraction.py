# %% [markdown]
# # Failure Case Extraction — coppie di immagini dove il modello ha sbagliato
#
# Ricostruisce ESATTAMENTE lo split (train/val/test) usato nel training
# originale (stesso `random_state`, stessa funzione di split), ricarica il
# checkpoint scelto, e per ogni coppia del TEST set (mai vista in training
# né early stopping) confronta lo score con la soglia EER già salvata nel
# CSV delle metriche finali. Salva:
#   - un CSV con tutte le coppie in errore (path, label, score, tipo errore)
#   - una griglia PNG delle N coppie più "gravi" (score più lontano dalla soglia)
#
# Configurazione tramite un JSON: modifica CONFIG_JSON nella cella sotto
# (o carica da file con CONFIG_PATH) per scegliere backbone/loss/split.

# %%
import sys
sys.path.append('/home/lorenzo/Documenti/GitHub/HandVerify')

import os
import json
import ast
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.models import get_model
from src.data import create_dataloaders, create_cross_dataset_dataloaders
from src.data import SiameseDataset, ContrastiveDataset, TripletDataset
from src.utils import set_seed, get_device

# %% [markdown]
# ## CONFIG
#
# Modifica questo blocco (o punta `CONFIG_PATH` a un file .json con la
# stessa struttura) per scegliere la configurazione da analizzare.
#
# - `backbone`: uno tra efficientnet_b0/b1, mobilenet_v3_small/large, resnet18/34
# - `loss_type`: 'bce' | 'contrastive' | 'triplet'
# - `split`: 'iam_to_iam' | 'iam_to_rimes' | 'rimes_to_iam' | 'rimes_to_rimes'
# - `top_n_per_type`: quante coppie (per tipo di errore) mostrare nella griglia PNG
# - `results_root` / `dataset_root`: path base, coerenti col training originale

# %%
CONFIG_PATH = None  # es. "./failure_config.json" — se valorizzato, sovrascrive CONFIG_JSON

CONFIG_JSON = """
{
    "backbone": "resnet18",
    "loss_type": "contrastive",
    "split": "rimes_to_iam",

    "dataset_root": "/home/lorenzo/Documenti/GitHub/HandVerify/datasets/processed-handwritten",
    "results_root": "/home/lorenzo/Documenti/GitHub/HandVerify/results",

    "random_state": 42,
    "target_size": 448,
    "batch_size": 16,

    "val_size": 0.1,
    "test_size": 0.2,

    "model_hparams": {
        "in_channels": 1,
        "freeze_backbone_layers": 3,
        "dropout": 0.2,
        "embedding_dim": 32
    },

    "margin": {
        "contrastive": 1.0,
        "triplet": 0.5
    },

    "top_n_per_type": 12,
    "output_dir": "./failure_analysis"
}
"""

if CONFIG_PATH is not None and os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = json.load(f)
else:
    CFG = json.loads(CONFIG_JSON)

print(json.dumps(CFG, indent=2))

# %% [markdown]
# ## Costanti derivate dalla config

# %%
BACKBONE = CFG["backbone"]
LOSS_TYPE = CFG["loss_type"]
SPLIT = CFG["split"]  # es. "rimes_to_iam"
TRAIN_DS, TEST_DS = SPLIT.split("_to_")

RANDOM_STATE = CFG["random_state"]
TARGET_SIZE = CFG["target_size"]

DATASET_ROOT = Path(CFG["dataset_root"])
IAM_PATH = DATASET_ROOT / "iam_processed"
RIMES_PATH = DATASET_ROOT / "rimes_processed"

DATASET_PATH_BY_NAME = {"iam": IAM_PATH, "rimes": RIMES_PATH}
TRAIN_PATH = DATASET_PATH_BY_NAME[TRAIN_DS]
TEST_PATH_ROOT = DATASET_PATH_BY_NAME[TEST_DS]

EXP_NAME = f"{BACKBONE}_{LOSS_TYPE}_{SPLIT}"
CHECKPOINT_PATH = (
    Path(CFG["results_root"]) / LOSS_TYPE / f"{LOSS_TYPE}_experiments"
    / EXP_NAME / f"{EXP_NAME}_best.pth"
)
METRICS_CSV_PATH = (
    Path(CFG["results_root"]) / LOSS_TYPE / f"{LOSS_TYPE}_experiments"
    / EXP_NAME / f"{EXP_NAME}_final_metrics.csv"
)

OUTPUT_DIR = Path(CFG["output_dir"]) / EXP_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert CHECKPOINT_PATH.exists(), f"Checkpoint non trovato: {CHECKPOINT_PATH}"
assert METRICS_CSV_PATH.exists(), f"Metrics CSV non trovato: {METRICS_CSV_PATH}"

print(f"Esperimento:   {EXP_NAME}")
print(f"Checkpoint:    {CHECKPOINT_PATH}")
print(f"Metrics CSV:   {METRICS_CSV_PATH}")
print(f"Output dir:    {OUTPUT_DIR}")

set_seed(RANDOM_STATE)
device = get_device()

# %% [markdown]
# ## Ricostruzione ESATTA dello split test
#
# Stessa funzione, stessi parametri, stesso `random_state` del training
# originale: `create_dataloaders` per same-dataset (train_size implicito
# 1 - val_size - test_size, split 3 vie sui writer), `create_cross_dataset_dataloaders`
# per cross-dataset (tutto il source in training, target diviso val/test).
#
# NB: per il caso same-dataset lo split è a 3 vie (train/val/test) fatto
# UNA VOLTA con `train_test_split(test_size=val_size+test_size)` poi
# suddiviso ulteriormente — esattamente come in `dataloader_factory.py`,
# quindi basta richiamare la stessa funzione con gli stessi argomenti per
# ottenere bit-per-bit lo stesso test set.

# %%
DATASET_CLASS_BY_TYPE = {
    "bce": SiameseDataset,
    "contrastive": ContrastiveDataset,
    "triplet": TripletDataset,
}
dataset_class = DATASET_CLASS_BY_TYPE[LOSS_TYPE]

is_same_dataset = (TRAIN_DS == TEST_DS)

if is_same_dataset:
    print(f"📊 Same-dataset split: {TRAIN_DS} (val_size={CFG['val_size']}, test_size={CFG['test_size']})")
    _, _, test_loader, _, _, test_dataset = create_dataloaders(
        dataset_class,
        data_root=str(TRAIN_PATH),
        batch_size=CFG["batch_size"],
        num_workers=0,
        val_size=CFG["val_size"],
        test_size=CFG["test_size"],
        target_size=TARGET_SIZE,
        random_state=RANDOM_STATE,
        # positive_ratio/resample non influenzano la composizione del test
        # set (fisso, tutte le genuine + subset impostor coerente col
        # validate_comprehensive originale), ma vanno passati per costruire
        # correttamente l'oggetto dataset.
        positive_ratio=0.5,
        resample_negatives_every_n_epochs=0,
    )
else:
    print(f"📊 Cross-dataset split: {TRAIN_DS} → {TEST_DS} (val_size={CFG['val_size']} sul target)")
    # create_cross_dataset_dataloaders: val_size = frazione di target_root
    # riservata a VAL; qui vogliamo il TEST (il resto, 1 - val_size).
    # La funzione ritorna già (train, val, test) coerentemente.
    _, _, test_loader, _, _, test_dataset = create_cross_dataset_dataloaders(
        dataset_class,
        train_root=str(TRAIN_PATH),
        target_root=str(TEST_PATH_ROOT),
        batch_size=CFG["batch_size"],
        num_workers=0,
        val_size=CFG["val_size"],
        target_size=TARGET_SIZE,
        random_state=RANDOM_STATE,
        positive_ratio=0.5,
        resample_negatives_every_n_epochs=0,
    )

n_pairs = len(test_dataset.validation_pairs) if hasattr(test_dataset, "validation_pairs") \
    else len(test_dataset.samples)
print(f"\n✓ Test set ricostruito: {n_pairs} coppie")

# %% [markdown]
# ## Verifica di coerenza contro Table 1 / il log originale
#
# Confronta il conteggio di coppie appena ricostruito con quello atteso
# (se lo conosci per questo split): se non coincide, lo split NON è
# identico a quello del training originale (path dataset diverso, ordine
# file diverso, o parametro random_state/val_size/test_size diverso) e i
# risultati sottostanti NON sono affidabili.

# %%
genuine_count = sum(1 for *_ , label in
                     (test_dataset.validation_pairs if hasattr(test_dataset, "validation_pairs")
                      else test_dataset.samples) if label == 1.0)
impostor_count = n_pairs - genuine_count
print(f"Genuine: {genuine_count}  |  Impostor: {impostor_count}")
print("Confronta questi numeri con Table 1 / il log originale per questo split "
      "prima di fidarti dei risultati sotto.")

# %% [markdown]
# ## Caricamento modello dal checkpoint

# %%
model_hparams = dict(CFG["model_hparams"])
model = get_model(BACKBONE, model_type=LOSS_TYPE, **model_hparams)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device).eval()

print(f"✓ Caricato {LOSS_TYPE}/{BACKBONE} da {CHECKPOINT_PATH.name}")

use_embeddings = LOSS_TYPE in ("contrastive", "triplet")
margin = CFG["margin"].get(LOSS_TYPE) if use_embeddings else None
if use_embeddings:
    print(f"  (score = cosine similarity tra embedding; margin di training = {margin}, "
          f"non usato in valutazione ma riportato per riferimento)")

# %% [markdown]
# ## Soglia EER — riletta dal CSV già prodotto per Table 4
#
# Non ricalcolata: usiamo esattamente lo stesso valore già pubblicato nel
# paper per questa configurazione, per garantire coerenza tra le coppie
# marcate come errore qui e i numeri (EER, accuracy@EER, ecc.) già riportati.

# %%
metrics_df = pd.read_csv(METRICS_CSV_PATH)
eer_threshold = float(metrics_df["eer_threshold"].values[0])
eer_reported = float(metrics_df["eer"].values[0])
auc_reported = float(metrics_df["auc"].values[0])

print(f"EER threshold (da CSV): {eer_threshold:.4f}")
print(f"EER riportato:          {eer_reported:.4f} ({eer_reported*100:.2f}%)")
print(f"AUC riportato:          {auc_reported:.4f}")

# %% [markdown]
# ## Calcolo score su tutte le coppie del test set + identificazione errori
#
# Stessa logica di `BaseTrainer.validate_comprehensive`: BCE usa l'output
# diretto del modello (similarity in [0,1]); Contrastive/Triplet usano la
# cosine similarity tra embedding L2-normalizzati.

# %%
from src.data.transforms import get_test_transforms
transform = get_test_transforms(TARGET_SIZE)

def load_img(path):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0).to(device)

@torch.no_grad()
def score_pair(img1_path, img2_path):
    img1 = load_img(img1_path)
    img2 = load_img(img2_path)
    if use_embeddings:
        emb1 = model(img1)
        emb2 = model(img2)
        return F.cosine_similarity(emb1, emb2).item()
    else:
        return model(img1, img2).item()

records = []
n_total = len(test_dataset.validation_pairs) if hasattr(test_dataset, "validation_pairs") \
    else len(test_dataset.samples)

from tqdm import tqdm
for idx in tqdm(range(n_total), desc="Scoring test pairs"):
    img1_path, img2_path, label = test_dataset.get_validation_pair(idx)
    score = score_pair(img1_path, img2_path)

    predicted_genuine = score >= eer_threshold
    is_genuine = (label == 1.0)

    if is_genuine and not predicted_genuine:
        error_type = "false_reject"
    elif (not is_genuine) and predicted_genuine:
        error_type = "false_accept"
    else:
        error_type = None  # correttamente classificato

    records.append({
        "img1_path": img1_path,
        "img2_path": img2_path,
        "label": "genuine" if is_genuine else "impostor",
        "score": score,
        "threshold": eer_threshold,
        "margin_from_threshold": score - eer_threshold,
        "error_type": error_type,
    })

results_df = pd.DataFrame(records)
n_errors = results_df["error_type"].notna().sum()
print(f"\n✓ {n_total} coppie valutate, {n_errors} errori "
      f"({n_errors/n_total*100:.2f}%, atteso ≈ EER={eer_reported*100:.2f}% dato che "
      f"si valuta esattamente alla soglia EER)")

# %% [markdown]
# ## Salvataggio CSV completo

# %%
csv_path = OUTPUT_DIR / f"{EXP_NAME}_pair_scores.csv"
results_df.to_csv(csv_path, index=False)
print(f"✓ Salvato CSV completo: {csv_path}")

errors_df = results_df[results_df["error_type"].notna()].copy()
errors_csv_path = OUTPUT_DIR / f"{EXP_NAME}_failure_pairs.csv"
errors_df.to_csv(errors_csv_path, index=False)
print(f"✓ Salvato CSV solo errori: {errors_csv_path} ({len(errors_df)} righe)")

# %% [markdown]
# ## Selezione dei fallimenti più "gravi"
#
# - False accept più gravi: impostor con score più alto (più lontano
#   sopra la soglia, il modello "ci ha creduto di più").
# - False reject più gravi: genuine con score più basso (più lontano
#   sotto la soglia).

# %%
TOP_N = CFG["top_n_per_type"]

worst_false_accepts = (
    errors_df[errors_df["error_type"] == "false_accept"]
    .sort_values("score", ascending=False)
    .head(TOP_N)
)
worst_false_rejects = (
    errors_df[errors_df["error_type"] == "false_reject"]
    .sort_values("score", ascending=True)
    .head(TOP_N)
)

print(f"False accepts trovati: {(errors_df['error_type']=='false_accept').sum()} "
      f"(mostrando i {len(worst_false_accepts)} più gravi)")
print(f"False rejects trovati: {(errors_df['error_type']=='false_reject').sum()} "
      f"(mostrando i {len(worst_false_rejects)} più gravi)")

worst_false_accepts.to_csv(OUTPUT_DIR / f"{EXP_NAME}_worst_false_accepts.csv", index=False)
worst_false_rejects.to_csv(OUTPUT_DIR / f"{EXP_NAME}_worst_false_rejects.csv", index=False)

# %% [markdown]
# ## Griglia visuale delle coppie in errore (immagini affiancate)

# %%
def plot_pair_grid(df, title, save_path, max_rows=12):
    df = df.head(max_rows)
    n = len(df)
    if n == 0:
        print(f"  (nessuna coppia per '{title}', salto il plot)")
        return

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (_, row) in enumerate(df.iterrows()):
        img1 = Image.open(row["img1_path"]).convert("L")
        img2 = Image.open(row["img2_path"]).convert("L")

        axes[i, 0].imshow(img1, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title(Path(row["img1_path"]).name, fontsize=7)

        axes[i, 1].imshow(img2, cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title(Path(row["img2_path"]).name, fontsize=7)

        fig.text(0.5, axes[i, 0].get_position().y1 + 0.002,
                  f"label={row['label']}  score={row['score']:.4f}  "
                  f"(threshold={row['threshold']:.4f}, Δ={row['margin_from_threshold']:+.4f})",
                  ha="center", fontsize=8)

    fig.suptitle(f"{title}\n{EXP_NAME}", fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Salvato: {save_path}")


print("Genero griglia false accepts...")
plot_pair_grid(
    worst_false_accepts,
    f"Top False Accepts (impostor pairs classificate come genuine) — {SPLIT}",
    OUTPUT_DIR / f"{EXP_NAME}_false_accepts_grid.png",
)

print("Genero griglia false rejects...")
plot_pair_grid(
    worst_false_rejects,
    f"Top False Rejects (genuine pairs classificate come impostor) — {SPLIT}",
    OUTPUT_DIR / f"{EXP_NAME}_false_rejects_grid.png",
)

# %% [markdown]
# ## Riepilogo finale

# %%
print(f"\n{'='*70}")
print(f"RIEPILOGO — {EXP_NAME}")
print(f"{'='*70}")
print(f"Test set:              {n_total} coppie ({genuine_count} genuine, {impostor_count} impostor)")
print(f"Errori totali:         {n_errors} ({n_errors/n_total*100:.2f}%)")
print(f"  di cui false accept: {(errors_df['error_type']=='false_accept').sum()}")
print(f"  di cui false reject: {(errors_df['error_type']=='false_reject').sum()}")
print(f"\nOutput salvati in: {OUTPUT_DIR}/")
print(f"  - {EXP_NAME}_pair_scores.csv        (tutte le coppie con score)")
print(f"  - {EXP_NAME}_failure_pairs.csv      (solo errori)")
print(f"  - {EXP_NAME}_worst_false_accepts.csv")
print(f"  - {EXP_NAME}_worst_false_rejects.csv")
print(f"  - {EXP_NAME}_false_accepts_grid.png")
print(f"  - {EXP_NAME}_false_rejects_grid.png")
print(f"{'='*70}\n")