# %% [markdown]
# # Diagnosi val loss `nan` con AMP — v2 (replica fedele)
#
# **Regole per replicare davvero il tuo run**
#
# 1. Riavvia il kernel ed esegui le celle **dall'alto verso il basso, senza eseguire altro in mezzo**.
#    Qualsiasi cella che consumi RNG (creare un modello, iterare un dataloader in train mode...)
#    prima della cella di cattura cambia l'inizializzazione e la traiettoria.
# 2. La cella di cattura ripete ESATTAMENTE la sequenza del tuo script:
#    `set_seed -> get_device -> get_model -> ContrastiveTrainer -> create_contrastive_dataloaders -> trainer.train(...)`
#    con la stessa CONFIG. L'unica differenza e' una sottoclasse del trainer il cui `validate_loss`
#    fa lo stesso calcolo ma **registra i batch non finiti** e poi interrompe il training.
# 3. La cattura salva batch cattivi + pesi di quell'epoca. Tutta l'analisi (parte B) lavora su quello:
#    nessun nuovo modello, nessuna nuova inizializzazione.
#
# Costo atteso se la NaN esce all'epoca 1 (come nel tuo ultimo log): ~2 min di train + ~2-4 min di val.
# Anche con seed identici la riproduzione bit-a-bit non e' garantita al 100% (kernel CUDA, worker dei
# dataloader): se la NaN non compare entro `CAPTURE_MAX_EPOCHS`, lo script si ferma da solo.

# %%
import sys
sys.path.append('/kaggle/working/HandVerify')

import os, copy, gc, time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from src.models import get_model
from src.training import ContrastiveTrainer
from src.data import create_contrastive_dataloaders
from src.utils import set_seed, get_device, ensure_dir

# %% [markdown]
# ## CONFIG (copiata dal tuo script)

# %%
RANDOM_STATE  = 42
EMBEDDING_DIM = 32

IAM_PATH = "/kaggle/input/processed-handwritten/processed-handwritten/iam_processed"

CONFIG = {
    'batch_size': 16,
    'num_workers': 4,
    'epochs': 50,
    'patience': 5,
    'target_size': 448,
    'val_size': 0.1,
    'test_size': 0.2,
    'positive_ratio': 0.5,
    'margin': 1.0,
    'resample_negatives_every_n_epochs': 1,
    'max_genuine_pairs': 2_000,
    'use_amp': True,
    'frozen_layers': 0,
    'dropout': 0.2
}

MODEL_NAME = 'efficientnet_b1'
DATA_PATH = IAM_PATH                      # esperimento iam_to_iam (stesso dataset -> split a 3 vie)
EXP_NAME = f"{MODEL_NAME}_contrastive_iam_to_iam"

BASE_OUTPUT_DIR = "/kaggle/working/diag_nan"
RESULTS_DIR = f"{BASE_OUTPUT_DIR}/{EXP_NAME}"

# --- opzioni della diagnostica (non influenzano il training) ---
CAPTURE_MAX_EPOCHS = 5          # se non esce nessuna NaN entro N epoche, si ferma
SCAN_ALL_AFTER_FIRST = True     # dopo il primo batch NaN continua la val per contarli tutti (+~2 min)
MAX_KEEP = 3                    # quanti batch cattivi conservare (tensori)
FP16_MAX = 65504.0

# %% [markdown]
# ## A1) Trainer con cattura (definizione di classe: non consuma RNG)

# %%
class NanCaptured(Exception):
    pass


class NoNanFound(Exception):
    pass


class DiagContrastiveTrainer(ContrastiveTrainer):
    """Identico a ContrastiveTrainer. Cambia SOLO validate_loss: stesso identico calcolo
    (autocast, embedding, criterion, media), ma tiene traccia dei batch non finiti."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_calls = 0
        self.capture = None

    @torch.no_grad()
    def validate_loss(self, val_loader):
        self.val_calls += 1
        self.model.eval()
        val_loss = 0.0
        bad_idx, kept = [], []

        for i, (img1, img2, labels) in enumerate(tqdm(val_loader, desc=f"Validation (ep {self.val_calls})")):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.float().to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                emb1 = self.model(img1)
                emb2 = self.model(img2)
            loss = self.criterion(emb1.float(), emb2.float(), labels)
            val_loss += loss.item()

            emb_bad = (~torch.isfinite(emb1).all()) or (~torch.isfinite(emb2).all())
            if bool(emb_bad) or not torch.isfinite(loss).item():
                bad_idx.append(i)
                if len(kept) < MAX_KEEP:
                    kept.append(dict(idx=i, a=img1.cpu(), b=img2.cpu(), y=labels.cpu()))
                if not SCAN_ALL_AFTER_FIRST:
                    break

        if bad_idx:
            self.capture = dict(
                epoch=self.val_calls,
                bad_idx=bad_idx,
                n_val_batches=len(val_loader),
                batches=kept,
                state={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            )
            torch.save(self.capture, f"{RESULTS_DIR}/capture.pt")
            raise NanCaptured(f"NaN catturata all'epoca {self.val_calls}: "
                              f"{len(bad_idx)} batch non finiti (primo: {bad_idx[0]})")

        if self.val_calls >= CAPTURE_MAX_EPOCHS:
            raise NoNanFound(f"Nessuna NaN nelle prime {CAPTURE_MAX_EPOCHS} epoche.")

        return val_loss / len(val_loader)

# %% [markdown]
# ## A2) Replica fedele — **eseguire questa cella per prima dopo A1, senza altro in mezzo**

# %%
set_seed(RANDOM_STATE)
device = get_device()
ensure_dir(BASE_OUTPUT_DIR)
ensure_dir(RESULTS_DIR)

model = get_model(
    MODEL_NAME,
    model_type='contrastive',
    in_channels=1,
    freeze_backbone_layers=CONFIG['frozen_layers'],
    projection_dim=EMBEDDING_DIM,
    dropout=CONFIG['dropout']
)

trainer = DiagContrastiveTrainer(
    model=model,
    model_name=EXP_NAME,
    device=device,
    margin=CONFIG['margin'],
    results_dir=RESULTS_DIR,
    use_amp=CONFIG['use_amp']
)

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
    create_contrastive_dataloaders(
        data_root=DATA_PATH,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        val_size=CONFIG['val_size'], test_size=CONFIG['test_size'],
        target_size=CONFIG['target_size'],
        random_state=RANDOM_STATE,
        max_genuine_pairs=CONFIG['max_genuine_pairs'],
        positive_ratio=CONFIG['positive_ratio'],
        resample_negatives_every_n_epochs=CONFIG['resample_negatives_every_n_epochs']
    )

print('\n')
print(f"Train pairs: {len(train_dataset)}")
print(f"Val pairs: {len(val_dataset)}")

try:
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        fold=None
    )
except NanCaptured as e:
    print("\n>>>", e)
except NoNanFound as e:
    print("\n>>>", e, "Riprova o passa alla cattura nel training vero (sezione B9).")

# %% [markdown]
# # Parte B — analisi sulla NaN catturata

# %%
assert trainer.capture is not None, "Nessuna NaN catturata: niente da analizzare (vedi sezione B9)."

cap = trainer.capture
bad_batches = cap["batches"]
model = trainer.model.eval()              # stessi pesi dell'epoca della NaN (nessun altro update dopo la cattura)
criterion = trainer.criterion
BS = CONFIG['batch_size']

print(f"Epoca: {cap['epoch']} | batch non finiti: {len(cap['bad_idx'])} / {cap['n_val_batches']}")
print("Indici batch non finiti:", cap["bad_idx"][:40])
print("-> pochi batch (1-5) = outlier isolati; molti = problema sistemico dei pesi/BN.")


# ---- utility --------------------------------------------------------------
def non_finite_rows(t):
    return ~torch.isfinite(t).reshape(t.shape[0], -1).all(dim=1)


def batch_paths(idx, k=None):
    s = idx * BS
    items = val_dataset.samples[s:s + BS]
    return items if k is None else items[k]


def trace_forward(m, x, amp):
    """Hook su tutti i moduli foglia in ordine di esecuzione."""
    recs, handles = [], []

    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                o = o.detach()
                fin = torch.isfinite(o)
                recs.append(dict(
                    name=name, type=type(mod).__name__, dtype=str(o.dtype).replace("torch.", ""),
                    finite=bool(fin.all()), n_bad=int((~fin).sum()),
                    absmax=float(o[fin].abs().max()) if fin.any() else float("nan"),
                ))
        return hook

    for n, mod in m.named_modules():
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(mk(n)))
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
            out = m(x)
    finally:
        for h in handles:
            h.remove()
    return pd.DataFrame(recs), out


@torch.no_grad()
def prenorm_stats(m, x, amp):
    with torch.autocast(device_type=device.type, enabled=amp):
        f = m.encoder(x)
        f = f.view(f.size(0), -1)
        p = m.projection(f)
    fin_f, fin_p = torch.isfinite(f), torch.isfinite(p)
    return dict(
        feat_dtype=str(f.dtype).replace("torch.", ""),
        feat_absmax=float(f[fin_f].abs().max()) if fin_f.any() else float("nan"),
        feat_nonfinite=int((~fin_f).sum()),
        proj_absmax=float(p[fin_p].abs().max()) if fin_p.any() else float("nan"),
        proj_nonfinite=int((~fin_p).sum()),
    )

# %% [markdown]
# ## B1) Quali campioni, e sono finiti in fp32?
# `fp32_finite=True` -> il campione e' sano in fp32 e rompe SOLO in fp16 (overflow da AMP).
# `fp32_finite=False` -> il problema c'e' anche in fp32 (dati o pesi).

# %%
rows = []
for bb in bad_batches:
    for side in ("a", "b"):
        x = bb[side].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type):
                e_amp = model(x)
            e_f32 = model(x)
        for k in torch.nonzero(non_finite_rows(e_amp)).flatten().tolist():
            t = x[k, 0].detach().cpu()
            rows.append(dict(
                batch=bb["idx"], side=side, k=k,
                path=batch_paths(bb["idx"], k)[0 if side == "a" else 1],
                fp32_finite=bool(torch.isfinite(e_f32[k]).all()),
                img_min=float(t.min()), img_max=float(t.max()), img_std=float(t.std()),
                ink_frac=float((t < 0.5).float().mean()),
            ))
culprits = pd.DataFrame(rows)
print(f"Campioni con embedding non finito (AMP): {len(culprits)}")
culprits

# %%
if len(culprits):
    c = culprits.iloc[0]
    bb = next(b for b in bad_batches if b["idx"] == c["batch"])
    other = "b" if c["side"] == "a" else "a"
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(bb[c["side"]][c["k"], 0].numpy(), cmap="gray"); ax[0].set_title("campione cattivo")
    ax[1].imshow(bb[other][c["k"], 0].numpy(), cmap="gray"); ax[1].set_title("altra immagine della coppia")
    for a_ in ax:
        a_.axis("off")
    plt.show()
    print(c.to_string())

# %% [markdown]
# ## B2) Primo modulo che produce valori non finiti
# Confronto layer per layer AMP vs fp32 sullo stesso campione.

# %%
if len(culprits):
    x1 = bb[c["side"]][c["k"]:c["k"] + 1].to(device)      # un solo campione (BN in eval => indipendente dal batch)
    tr_amp, _ = trace_forward(model, x1, amp=True)
    tr_f32, _ = trace_forward(model, x1, amp=False)
    cmp = tr_amp.merge(tr_f32[["name", "absmax", "finite"]], on="name", suffixes=("_amp", "_f32"))

    print("prenorm AMP :", prenorm_stats(model, x1, True))
    print("prenorm fp32:", prenorm_stats(model, x1, False))

    first = cmp.index[~cmp["finite_amp"]]
    if len(first):
        i0 = first[0]
        print(f"\nPrimo modulo non finito (AMP): #{i0} {cmp.loc[i0, 'name']} ({cmp.loc[i0, 'type']})")
        print("Finestra attorno al punto di rottura (absmax_f32 = valore reale in fp32; limite fp16 = 65504):")
        display_cols = ["name", "type", "dtype", "finite_amp", "absmax_amp", "absmax_f32"]
        print(cmp.iloc[max(0, i0 - 6): i0 + 4][display_cols].to_string())
    else:
        print("\nNessun modulo foglia non finito: il nan nasce in un'operazione non-modulo (somma residua, F.normalize).")

# %% [markdown]
# ## B3) Il campione e' un outlier rispetto a un batch normale?
# Rapporto tra massimo per layer del campione cattivo e di un batch regolare (fp32).

# %%
if len(culprits):
    reg_batch = next(b for b in val_loader)[0].to(device)
    tr_reg, _ = trace_forward(model, reg_batch, amp=False)
    ratio = cmp[["name", "type", "absmax_f32"]].merge(tr_reg[["name", "absmax"]].rename(columns={"absmax": "absmax_regular"}), on="name")
    ratio["ratio"] = ratio.absmax_f32 / ratio.absmax_regular.clip(lower=1e-6)
    print(f"Max attivazione fp32 del campione cattivo: {ratio.absmax_f32.max():.1f} "
          f"({100*ratio.absmax_f32.max()/FP16_MAX:.2f}% del limite fp16)")
    print(f"Max attivazione fp32 batch regolare      : {ratio.absmax_regular.max():.1f}")
    ratio.sort_values("ratio", ascending=False).head(10)

# %% [markdown]
# ## B4) BatchNorm della testa (BN1d) e statistiche running
# Un `running_var` molto piccolo amplifica in eval qualunque scostamento rispetto al train
# (dove la BN usa le statistiche del batch e c'e' dropout attivo).

# %%
rows = []
for n, mod in model.named_modules():
    if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
        rows.append(dict(name=n, kind=type(mod).__name__, var_min=float(mod.running_var.min()),
                         var_max=float(mod.running_var.max()), mean_absmax=float(mod.running_mean.abs().max()),
                         gamma_absmax=float(mod.weight.abs().max())))
bn_df = pd.DataFrame(rows)
print(f"BN totali: {len(bn_df)} | running_var min globale = {bn_df.var_min.min():.3e}")
print("\nBN della testa (projection):")
print(bn_df[bn_df.name.str.startswith("projection")].to_string(index=False))
print("\nPeggiori 8 per var_min (dove 1/sqrt(var) e' piu' grande):")
bn_df.sort_values("var_min").head(8)

# %% [markdown]
# ## B5) Stesso batch in train() vs eval(), AMP on/off (su una copia)

# %%
if len(culprits):
    rows = []
    for mode in ("eval", "train"):
        mcopy = copy.deepcopy(model)
        mcopy.train() if mode == "train" else mcopy.eval()
        for amp in (True, False):
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                e1, e2 = mcopy(bb["a"].to(device)), mcopy(bb["b"].to(device))
            rows.append(dict(bn_mode=mode, amp=amp,
                             n_bad_emb=int(non_finite_rows(e1).sum() + non_finite_rows(e2).sum())))
        del mcopy
    print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## B6) Test delle correzioni

# %%
def embed(m, x, mode):
    if mode == "fp32":
        return m(x)
    if mode == "amp":
        with torch.autocast(device_type=device.type):
            return m(x)
    if mode == "amp_head_fp32":
        with torch.autocast(device_type=device.type):
            f = m.encoder(x)
        f = f.float().view(f.size(0), -1)
        with torch.autocast(device_type=device.type, enabled=False):
            return F.normalize(m.projection(f), p=2, dim=1)
    if mode == "amp_fallback":
        with torch.autocast(device_type=device.type):
            e = m(x)
        return e if torch.isfinite(e).all() else m(x)
    raise ValueError(mode)


test_batches = [(b["a"], b["b"], b["y"]) for b in bad_batches]
for i, (a, b, y) in enumerate(val_loader):
    if i >= 40:
        break
    test_batches.append((a, b, y))

model.eval()
ref = []
with torch.no_grad():
    for a, b, y in test_batches:
        ref.append((embed(model, a.to(device), "fp32"), embed(model, b.to(device), "fp32")))

rows = []
for mode in ["fp32", "amp", "amp_head_fp32", "amp_fallback"]:
    n_bad, losses, cos_min = 0, [], []
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for (a, b, y), (r1, r2) in zip(test_batches, ref):
            e1, e2 = embed(model, a.to(device), mode), embed(model, b.to(device), mode)
            loss = criterion(e1.float(), e2.float(), y.float().to(device))
            if not torch.isfinite(loss):
                n_bad += 1
                continue
            losses.append(loss.item())
            cos_min.append(float(F.cosine_similarity(e1.float(), r1).min()))
    torch.cuda.synchronize()
    rows.append(dict(mode=mode, batches=len(test_batches), n_nonfinite_batches=n_bad,
                     mean_loss_finite=np.mean(losses) if losses else float("nan"),
                     min_cos_vs_fp32=min(cos_min) if cos_min else float("nan"),
                     seconds=round(time.time() - t0, 1)))
variants = pd.DataFrame(rows).set_index("mode")
variants

# %% [markdown]
# ## B7) (Opzionale) Sanita' di tutte le immagini di val
# Cerca immagini vuote/costanti/illeggibili. ~1-2 min.

# %%
SCAN_IMAGES = True
if SCAN_IMAGES:
    paths = sorted({p for s in val_dataset.samples for p in s[:2]})
    rows = []
    for p in tqdm(paths, desc="immagini val"):
        try:
            t = val_dataset._load_image(p)
            rows.append(dict(path=p, finite=bool(torch.isfinite(t).all()), std=float(t.std()),
                             ink_frac=float((t < 0.5).float().mean()), error=None))
        except Exception as e:
            rows.append(dict(path=p, error=repr(e)))
    img_df = pd.DataFrame(rows)
    anomalous = img_df[(img_df.error.notna()) | (~img_df.finite.fillna(True)) |
                       (img_df["std"] < 1e-6) | (img_df.ink_frac < 0.001) | (img_df.ink_frac > 0.6)]
    print(f"Immagini: {len(img_df)} | anomale: {len(anomalous)}")
    if len(culprits):
        print("Il campione cattivo e' tra le anomale:", c["path"] in set(anomalous.path))
    display_df = anomalous.head(15)
    display_df

# %% [markdown]
# ## B8) Verdetto automatico

# %%
def verdict():
    out = []
    if not len(culprits):
        return ["Nessun embedding non finito nei batch catturati (la loss era non finita per altro motivo): controlla B2/B5."]
    n_f32_bad = int((~culprits.fp32_finite).sum())
    c0 = culprits.iloc[0]

    out.append(f"NaN all'epoca {cap['epoch']}: {len(cap['bad_idx'])}/{cap['n_val_batches']} batch non finiti, "
               f"{len(culprits)} campioni cattivi nei batch conservati.")
    if n_f32_bad:
        out.append(f"[CAUSA] {n_f32_bad} campioni sono non finiti ANCHE in fp32: non e' solo AMP (dati o pesi). Vedi B2/B4/B7.")
    else:
        out.append("[CAUSA] tutti i campioni cattivi sono finiti in fp32: e' overflow/instabilita' fp16 in eval.")

    if c0.img_std < 1e-6 or c0.ink_frac < 0.001 or c0.ink_frac > 0.6:
        out.append(f"[DATI] il campione cattivo e' atipico (std={c0.img_std:.4f}, inchiostro={100*c0.ink_frac:.2f}%): "
                   f"probabile immagine fuori distribuzione che gonfia le attivazioni.")

    if 'first' in globals() and len(first):
        out.append(f"[DOVE] il primo non finito e' in '{cmp.loc[first[0], 'name']}' ({cmp.loc[first[0], 'type']}).")

    v = variants
    if v.loc["amp", "n_nonfinite_batches"] > 0:
        if v.loc["amp_head_fp32", "n_nonfinite_batches"] == 0:
            out.append("[FIX] projection+normalize in fp32 ('amp_head_fp32') elimina i nan: l'overflow e' nella testa.")
        else:
            out.append("[FIX] 'amp_head_fp32' NON basta: l'overflow e' gia' nell'encoder.")
        if v.loc["amp_fallback", "n_nonfinite_batches"] == 0:
            out.append("[FIX] 'amp_fallback' (ricalcolo fp32 solo sui batch non finiti) elimina i nan.")
        if v.loc["fp32", "n_nonfinite_batches"] == 0:
            out.append("[FIX] la val in fp32 e' sempre finita: soluzione sicura (costo: val piu' lenta).")
    return out


for m_ in verdict():
    print(m_)

# %% [markdown]
# ## B9) Fix nel training vero
#
# **`validate_loss` robusto** (in `ContrastiveTrainer`, serve `import os` in cima al file). Alla prima
# non-finitezza salva batch e pesi, ricalcola il batch in fp32 e non avvelena la media:
#
# ```python
# @torch.no_grad()
# def validate_loss(self, val_loader):
#     self.model.eval()
#     total, n_ok, n_fallback = 0.0, 0, 0
#     batch_path = os.path.join(self.results_dir, "first_bad_val_batch.pt")
#     state_path = os.path.join(self.results_dir, "state_at_first_nan.pth")
#
#     for bi, (img1, img2, labels) in enumerate(tqdm(val_loader, desc="Validation")):
#         img1, img2 = img1.to(self.device), img2.to(self.device)
#         labels = labels.float().to(self.device)
#
#         with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
#             emb1, emb2 = self.model(img1), self.model(img2)
#
#         if not (torch.isfinite(emb1).all() and torch.isfinite(emb2).all()):
#             n_fallback += 1
#             if not os.path.exists(batch_path):
#                 torch.save({"idx": bi, "a": img1.cpu(), "b": img2.cpu(), "y": labels.cpu()}, batch_path)
#                 torch.save(self.model.state_dict(), state_path)
#             emb1, emb2 = self.model(img1), self.model(img2)      # ricalcolo fp32
#
#         loss = self.criterion(emb1.float(), emb2.float(), labels)
#         if torch.isfinite(loss):
#             total += loss.item(); n_ok += 1
#
#     if n_fallback:
#         print(f"  ⚠ {n_fallback} batch di validazione ricalcolati in fp32")
#     return total / max(n_ok, 1)
# ```
#
# **Guard in `BaseTrainer.train`** (una val loss non finita non deve consumare patience ne' far
# scendere il learning rate: `ReduceLROnPlateau.step(nan)` conta come "nessun miglioramento"):
#
# ```python
# import math
# ...
# val_loss = self.validate_loss(val_loader)
# if not math.isfinite(val_loss):
#     print("  ⚠ val loss non finita: epoca ignorata per patience e scheduler")
#     if hasattr(train_dataset, 'on_epoch_end'):
#         train_dataset.on_epoch_end(epoch)
#     continue
# ```