# %% [markdown]
# # Grad-CAM: pesi vecchi vs pesi nuovi sulle STESSE coppie
#
# Funziona con: resnet18/34/50, efficientnet_b0/b1, mobilenet_v3_small/large
# e con i tre tipi di loss (bce | contrastive | triplet).
#
# Per ogni rete si ispezionano due livelli (vedi `get_cam_layers`):
#   - "final"    : ultima feature map prima del global pooling (stride 32, 14x14 a 448 px)
#   - "stride16" : fine dell'ultimo stage a stride 16 (28x28 a 448 px), piu' fine ma
#                  meno legato alla decisione: da usare come vista di dettaglio.
#
# Come leggere le mappe (versione con segno di gradcam.py):
#   rosso = regione che AUMENTA la somiglianza tra le due scritture
#   blu   = regione che la DIMINUISCE
# Ogni mappa e' normalizzata sul proprio massimo: si confronta DOVE guarda il
# modello, non quanto intensamente.

# %%
import sys
sys.path.append('/home/lorenzo/Documenti/GitHub/HandVerify')

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.models import get_model
from src.data.transforms import get_test_transforms
from src.interpretability import compute_pairwise_gradcam, overlay_cam_on_image
from src.utils import get_device

device = get_device()

# %% [markdown]
# ## CONFIG

# %%
# resnet18 | resnet34 | efficientnet_b0 | efficientnet_b1 | mobilenet_v3_small | mobilenet_v3_large
BACKBONE = "resnet18"
LOSS_TYPE = "contrastive"      # bce | contrastive | triplet
TARGET_SIZE = 448
FREEZE_LAYERS = 3
DROPOUT = 0.2

IMAGES_ROOT = Path("/home/lorenzo/Documenti/GitHub/HandVerify/datasets/processed-handwritten/iam_processed")
OUTPUT_DIR = Path("./gradcam_compare") / f"{BACKBONE}_{LOSS_TYPE}"

RESULTS = Path("/home/lorenzo/Documenti/GitHub/HandVerify/results")
RESULTS2 = Path("/home/lorenzo/Documenti/GitHub/HandVerify/results2")
EXP = f"{BACKBONE}_{LOSS_TYPE}_iam_to_iam"
# adatta se la struttura delle cartelle per bce/triplet e' diversa da quella contrastive
EXP_SUBDIR = Path(LOSS_TYPE) / f"{LOSS_TYPE}_experiments" / EXP

# "threshold": un float, oppure il path di un *_final_metrics.csv da cui leggere
# eer_threshold. "embedding_dim" (per bce: projection_dim) puo' differire tra i due modelli.
# ATTENZIONE: soglie ed embedding_dim qui sotto erano quelli del run mobilenet_v3_small:
# aggiornali per il backbone che stai analizzando.
MODELS = {
    "vecchi": {
        "checkpoint": RESULTS / EXP_SUBDIR / f"{EXP}_best.pth",
        "threshold": 0.6469,
        "embedding_dim": 32,
    },
    "nuovi": {
        "checkpoint": RESULTS2 / EXP_SUBDIR / f"{EXP}_best.pth",
        "threshold": 0.535,
        "embedding_dim": 32,
    },
}

# Livelli da ispezionare, tag del modello per il confronto tra livelli
LEVELS = ["stride16", "final"]
MAIN_LEVEL = "final"
COMPARE_TAG = "nuovi"

# Coppie dei failure case (erano quelle del run mobilenet: con un altro backbone
# gli errori possono essere altri, eventualmente sostituiscile).
PAIRS = {
    "false_accepts": [
        ("k03-117_full.png", "r02-065_full.png", "impostor"),
        ("l01-065_full.png", "r02-065_full.png", "impostor"),
    ],
    "false_rejects": [
        ("g01-004_full.png", "g01-008_full.png", "genuine"),
        ("h06-079_full.png", "h06-085_full.png", "genuine"),
    ],
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Livelli da ispezionare per ogni backbone
#
# Regola: si aggancia l'OUTPUT dell'intero stage/blocco (dopo BN, attivazione e
# skip connection), mai un Conv2d interno (che sarebbe pre-BN e pre-skip).

# %%
# indice (in encoder[0] = `features`) dell'ultimo blocco a stride 16
_STRIDE16_IDX = {
    "efficientnet_b0": 5,
    "efficientnet_b1": 5,
    "mobilenet_v3_small": 8,
    "mobilenet_v3_large": 12,
}


def get_cam_layers(model, backbone):
    """Ritorna {livello: (modulo, etichetta)} per il backbone dato."""
    body = model.encoder[0]
    if backbone.startswith("resnet"):
        _stem, _l1, _l2, l3, l4, _pool = list(body.children())
        return {
            "stride16": (l3, "layer3"),
            "final": (l4, "layer4"),
        }
    if backbone in _STRIDE16_IDX:
        i16 = _STRIDE16_IDX[backbone]
        last = len(body) - 1
        return {
            "stride16": (body[i16], f"features[{i16}]"),
            "final": (body[last], f"features[{last}]"),
        }
    raise ValueError(f"Backbone non supportato in get_cam_layers: {backbone}")


def probe_layers(model, layers, size):
    """Forward finto per verificare canali/risoluzione/stride dei layer scelti."""
    shapes, handles = {}, []
    for lvl, (mod, _) in layers.items():
        handles.append(mod.register_forward_hook(
            lambda m, i, o, lvl=lvl: shapes.__setitem__(lvl, tuple(o.shape))
        ))
    with torch.no_grad():
        model.encoder(torch.zeros(1, 1, size, size, device=device))
    for h in handles:
        h.remove()

    expected = {"stride16": size // 16, "final": size // 32}
    for lvl, (_, label) in layers.items():
        c, h, w = shapes[lvl][1:]
        flag = "OK" if h == expected[lvl] else f"ATTENZIONE: atteso {expected[lvl]}x{expected[lvl]}"
        print(f"  {lvl:9s} {label:14s} -> {c:4d} canali, {h}x{w}  {flag}")

# %% [markdown]
# ## Localizza le immagini per nome file

# %%
index = {}
for p in IMAGES_ROOT.rglob("*.png"):
    index.setdefault(p.name, []).append(p)


def resolve(name):
    matches = index.get(name, [])
    if len(matches) != 1:
        raise RuntimeError(
            f"'{name}': trovati {len(matches)} file in {IMAGES_ROOT}: {matches}. "
            f"Se >1, sostituisci il nome in PAIRS con il path completo."
        )
    return matches[0]


resolved = {
    group: [(resolve(a), resolve(b), label) for a, b, label in pairs]
    for group, pairs in PAIRS.items()
}
for group, pairs in resolved.items():
    for a, b, _ in pairs:
        print(f"{group}: {a}  |  {b}")

# %% [markdown]
# ## Caricamento modelli e utilita'

# %%
transform = get_test_transforms(TARGET_SIZE)


def load_tensor(path):
    img = Image.open(path).convert("L")
    # requires_grad: il backward funziona anche se i layer a monte del target
    # sono congelati (altrimenti lo score non avrebbe grad_fn).
    return transform(img).unsqueeze(0).to(device).requires_grad_(True)


def load_gray_uint8(path):
    # stesso resize (stretch) del transform di test, cosi' la CAM combacia
    return np.array(Image.open(path).convert("L").resize((TARGET_SIZE, TARGET_SIZE)))


def resolve_threshold(t):
    if isinstance(t, (int, float)):
        return float(t)
    return float(pd.read_csv(t)["eer_threshold"].values[0])


def load_model(cfg):
    model = get_model(
        BACKBONE, model_type=LOSS_TYPE,
        in_channels=1, freeze_backbone_layers=FREEZE_LAYERS,
        dropout=DROPOUT, embedding_dim=cfg["embedding_dim"],   # get_model lo mappa su projection_dim per bce
    )
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def decision_score(model, t1, t2):
    """Score usato per la soglia EER (indipendente da cosa restituisce gradcam.py).
    bce: probabilita' del classificatore; contrastive/triplet: cosine similarity."""
    if LOSS_TYPE == "bce":
        return float(model(t1, t2).squeeze())
    return float(F.cosine_similarity(model(t1), model(t2)))


loaded, cam_layers = {}, {}
for tag, cfg in MODELS.items():
    model = load_model(cfg)
    loaded[tag] = (model, resolve_threshold(cfg["threshold"]))
    cam_layers[tag] = get_cam_layers(model, BACKBONE)
    print(f"{tag}: soglia EER = {loaded[tag][1]:.4f}")
    probe_layers(model, cam_layers[tag], TARGET_SIZE)

# %% [markdown]
# ## Controllo rapido sulla CAM
# Se `min < 0` la CAM conserva il segno (versione nuova di gradcam.py).
# Se `min == 0` stai ancora usando la normalizzazione senza segno.

# %%
p1, p2, _ = resolved[next(iter(resolved))][0]
model, _ = loaded[COMPARE_TAG]
c1, c2, s = compute_pairwise_gradcam(
    model, LOSS_TYPE, load_tensor(p1), load_tensor(p2), device,
    target_layer=cam_layers[COMPARE_TAG]["final"][0],
)
print(f"cam1 range [{c1.min():.3f}, {c1.max():.3f}]  shape {c1.shape}")
print(f"cam2 range [{c2.min():.3f}, {c2.max():.3f}]  shape {c2.shape}")

# %% [markdown]
# ## Griglie Grad-CAM
# Una sola funzione per tutti i confronti: ogni "colonna" e' una coppia
# (modello, livello) e occupa due riquadri (img1, img2).
# Titolo: livello, esito della coppia (score vs soglia, corretto/errore).

# %%
FRAME_COLORS = {"vecchi": "red", "nuovi": "green"}


def plot_grid(group, pairs, columns, title, fname):
    n = len(pairs)
    fig, axes = plt.subplots(
        n, 2 * len(columns),
        figsize=(4.2 * 2 * len(columns), 4.8 * n), squeeze=False,
    )

    for i, (p1, p2, label) in enumerate(pairs):
        t1, t2 = load_tensor(p1), load_tensor(p2)
        g1, g2 = load_gray_uint8(p1), load_gray_uint8(p2)

        for j, (tag, level) in enumerate(columns):
            model, thr = loaded[tag]
            layer, layer_label = cam_layers[tag][level]
            cam1, cam2, _ = compute_pairwise_gradcam(
                model, LOSS_TYPE, t1, t2, device, target_layer=layer
            )
            score = decision_score(model, t1, t2)
            ok = (score >= thr) == (label == "genuine")
            color = FRAME_COLORS.get(tag, "black")

            for k, (g, cam, p) in enumerate([(g1, cam1, p1), (g2, cam2, p2)]):
                ax = axes[i, 2 * j + k]
                ax.imshow(cv2.cvtColor(overlay_cam_on_image(g, cam), cv2.COLOR_BGR2RGB))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(color)
                    spine.set_linewidth(5)
                if k == 0:
                    head = (f"{tag} | {level} ({layer_label})\n"
                            f"{label} | score={score:.3f} (thr {thr:.3f}) {'OK' if ok else 'ERRORE'}\n")
                else:
                    head = "\n\n"
                ax.set_title(head + p.name, fontsize=8, color=color)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out = OUTPUT_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Salvato: {out}")

# %% [markdown]
# ### 1) Vecchi vs nuovi, allo stesso livello

# %%
for level in LEVELS:
    for group, pairs in resolved.items():
        plot_grid(
            group, pairs,
            columns=[(tag, level) for tag in loaded],
            title=f"Grad-CAM {group} | {BACKBONE} {LOSS_TYPE} | livello {level} | {' vs '.join(loaded)}",
            fname=f"gradcam_compare_{group}_{level}.png",
        )

# %% [markdown]
# ### 2) Stesso modello, livelli diversi
# Serve a capire quanto la spiegazione cambia con la risoluzione: se "final" e
# "stride16" indicano regioni molto diverse, non fidarti di una sola vista.

# %%
for group, pairs in resolved.items():
    plot_grid(
        group, pairs,
        columns=[(COMPARE_TAG, level) for level in LEVELS],
        title=f"Grad-CAM per livello | {group} | {BACKBONE} {LOSS_TYPE} | modello {COMPARE_TAG}",
        fname=f"gradcam_levels_{group}_{COMPARE_TAG}.png",
    )

# %% [markdown]
# ## Tabella riassuntiva degli score (senza CAM)

# %%
rows = []
for group, pairs in resolved.items():
    for p1, p2, label in pairs:
        t1, t2 = load_tensor(p1), load_tensor(p2)
        for tag, (model, thr) in loaded.items():
            score = decision_score(model, t1, t2)
            rows.append({
                "group": group, "img1": p1.name, "img2": p2.name, "label": label,
                "model": tag, "score": score, "threshold": thr,
                "correct": (score >= thr) == (label == "genuine"),
            })

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "gradcam_compare_scores.csv", index=False)
summary