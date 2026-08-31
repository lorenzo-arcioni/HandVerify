# %% [markdown]
# # Training finale per la demo: ResNet18 su IAM+RIMES mescolati
# ## Loss configurabile (bce / contrastive / triplet), impostor stratificati,
# ## genuine cross-dataset impossibili per costruzione
#
# Per cambiare loss basta modificare `CONFIG['loss_type']` qui sotto: tutto
# il resto (dataset, trainer, parametri del modello) si adatta da solo.
#
# Riepilogo delle protezioni gia' discusse:
#
# 1. **Genuine pair cross-dataset (bug)**: i writer_id vengono prefissati
#    con la sorgente (`iam__042`, `rimes__042`), cosi' una directory `IAM/042`
#    e una `RIMES/042` non possono mai essere fuse nello stesso writer.
# 2. **Impostor sbilanciati verso RIMES**: per BCE/Contrastive il pool di
#    impostor e' diviso in 3 strati (iam-iam / rimes-rimes / cross) pesati
#    esplicitamente. Per Triplet (dove ogni negativo e' scelto per un
#    singolo anchor, non per una coppia di writer arbitraria) si usano 2
#    strati dal punto di vista dell'anchor: "stesso dataset dell'anchor,
#    altro writer" vs "dataset diverso dall'anchor".

# %%
import sys
sys.path.append('/kaggle/working/HandVerify')  # <-- adattare al path del repo

import os
import random
from pathlib import Path
from itertools import combinations
from datetime import datetime

import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data.base_dataset import BaseWriterDataset
from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.dataloader_factory import _make_loader
from src.models import get_model
from src.training import BCETrainer, ContrastiveTrainer, TripletTrainer
from src.utils import set_seed, get_device, ensure_dir

# %% [markdown]
# ## Configurazione
#
# `loss_type`: 'bce' | 'contrastive' | 'triplet'. Tutto cio' che dipende
# dalla loss (classe dataset, classe trainer, se serve `margin`, come si
# chiama la dimensione dell'embedding) viene risolto piu' sotto in base a
# questo unico valore.

# %%
RANDOM_STATE = 42

IAM_PATH = "/kaggle/input/processed-handwritten/processed-handwritten/iam_processed"
RIMES_PATH = "/kaggle/input/processed-handwritten/processed-handwritten/rimes_processed"

CONFIG = {
    'loss_type': 'contrastive',     # <-- cambia qui: 'bce' | 'contrastive' | 'triplet'
    'backbone': 'resnet18',
    'embedding_dim': 128,           # per BCE viene rimappato in automatico su projection_dim
    'freeze_backbone_layers': 3,
    'dropout': 0.4,
    'margin': 0.5,                  # usato solo da contrastive/triplet, ignorato per bce
    'batch_size': 16,
    'num_workers': 4,
    'target_size': 448,
    'val_size': 0.10,
    'positive_ratio': 0.5,
    'resample_negatives_every_n_epochs': 1,
    'epochs': 50,
    'patience': 7,

    # Usati da BCE e Contrastive: pesi relativi (normalizzati in automatico)
    # dei 3 strati di impostor pair.
    'pair_strata_ratios': {'iam-iam': 1 / 3, 'rimes-rimes': 1 / 3, 'cross': 1 / 3},

    # Usati solo da Triplet: pesi relativi dei negativi scelti per ogni
    # anchor, dal punto di vista dell'anchor stesso (non ha senso "rimes-rimes"
    # se l'anchor e' IAM: o il negativo e' un altro writer IAM, o e' un
    # writer RIMES qualsiasi).
    'triplet_negative_strata_ratios': {'same_dataset': 0.5, 'cross_dataset': 0.5},
}

assert CONFIG['loss_type'] in ('bce', 'contrastive', 'triplet')

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME = f"{CONFIG['backbone']}_{CONFIG['loss_type']}_mixed_iam_rimes_stratified"
RESULTS_DIR = f"/kaggle/working/final_model/{EXP_NAME}"

print(f"Esperimento: {EXP_NAME}")
print(f"Loss: {CONFIG['loss_type']}")
print(f"Output: {RESULTS_DIR}")
print(f"Timestamp: {TIMESTAMP}")

set_seed(RANDOM_STATE)
device = get_device()
ensure_dir(RESULTS_DIR)

# %% [markdown]
# ## Funzioni condivise: caricamento writer "safe" + strati di impostor
#
# Usate sia dal dataset per BCE/Contrastive sia (in parte) da quello per
# Triplet, per non duplicare tre volte la stessa logica di prefissaggio e
# stratificazione.

# %%
def load_writer_images_prefixed(writer_sources):
    """
    writer_sources: lista di (writer_dir_path, source_label).
    Ritorna (writer_images, writer_source), con writer_id sempre prefissato
    dalla sorgente (es. 'iam__042'): una collisione di nome tra directory
    IAM e RIMES non puo' mai fondere due writer diversi nello stesso id.
    """
    writer_images, writer_source = {}, {}
    seen_ids = set()

    for writer_dir, source in writer_sources:
        writer_id = f"{source}__{Path(writer_dir).name}"
        assert writer_id not in seen_ids, (
            f"writer_id duplicato: '{writer_id}' ({writer_dir}). Non dovrebbe "
            f"poter succedere: controllare la lista writer_sources."
        )
        seen_ids.add(writer_id)

        writer_images[writer_id] = [
            os.path.join(writer_dir, f)
            for f in os.listdir(writer_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        writer_source[writer_id] = source

    return writer_images, writer_source


def build_pair_strata(writer_ids, writer_images, writer_source):
    """Tutte le impostor pairs (img1, img2) divise in 3 strati per sorgente."""
    strata = {"iam-iam": [], "rimes-rimes": [], "cross": []}
    for w1, w2 in combinations(writer_ids, 2):
        s1, s2 = writer_source[w1], writer_source[w2]
        key = ("iam-iam" if s1 == s2 == "iam" else
               "rimes-rimes" if s1 == s2 == "rimes" else "cross")
        for img1 in writer_images[w1]:
            for img2 in writer_images[w2]:
                strata[key].append((img1, img2))
    return strata


def sample_from_strata(strata, ratios, n_total, rng, label=""):
    """Campiona n_total coppie da `strata` secondo `ratios` (normalizzati)."""
    total_w = sum(ratios.values())
    ratios = {k: v / total_w for k, v in ratios.items()}
    sampled = []
    for key, w in ratios.items():
        pool = strata.get(key, [])
        n_target = int(round(n_total * w))
        n_take = min(n_target, len(pool))
        if n_take < n_target:
            print(f"  [attenzione]{(' ' + label) if label else ''} strato '{key}': "
                  f"richieste {n_target}, disponibili {len(pool):,}. Uso tutte quelle disponibili.")
        sampled.extend(rng.sample(pool, n_take))
    return sampled

# %% [markdown]
# ## Dataset per BCE / Contrastive: impostor stratificati (3 strati)
#
# Sottoclasse di `BaseWriterDataset` (la base di `SiameseDataset` e
# `ContrastiveDataset`, identiche a parte il nome): stessa interfaccia,
# solo `_load_writer_images`/`_generate_all_impostor_pairs`/
# `_resample_negatives` sono sovrascritti per usare writer_id prefissati e
# campionamento stratificato invece che uniforme sul pool combinato.

# %%
class StratifiedPairDataset(BaseWriterDataset):
    """Dataset di coppie (img1, img2, label) valido sia per BCE che per
    Contrastive: le due loss condividono esattamente lo stesso formato di
    input, cambia solo la loss/trainer usati a valle."""

    def __init__(self, writer_sources, train=True, target_size=448,
                 positive_ratio=0.5, resample_negatives_every_n_epochs=1,
                 strata_ratios=None, random_seed=42):
        self._writer_sources_input = writer_sources
        self.strata_ratios = strata_ratios or {"iam-iam": 1/3, "rimes-rimes": 1/3, "cross": 1/3}

        print(f"\n{'='*60}")
        print(f"Initializing {'TRAIN' if train else 'VAL'} StratifiedPairDataset")
        print(f"{'='*60}")
        print(f"  Writers: {len(writer_sources)}  |  Strata ratios: {self.strata_ratios}")

        super().__init__(
            writer_dirs=[d for d, _ in writer_sources],
            train=train, target_size=target_size,
            positive_ratio=positive_ratio,
            resample_negatives_every_n_epochs=resample_negatives_every_n_epochs,
            random_seed=random_seed,
        )

    def _load_writer_images(self):
        writer_images, self.writer_source = load_writer_images_prefixed(self._writer_sources_input)
        return writer_images

    def _generate_all_impostor_pairs(self):
        self._impostor_strata = build_pair_strata(self.writer_ids, self.writer_images, self.writer_source)
        for key, pairs in self._impostor_strata.items():
            print(f"  Impostor pool [{key}]: {len(pairs):,} coppie")
        return [p for pairs in self._impostor_strata.values() for p in pairs]

    def _resample_negatives(self, epoch: int = 0):
        rng = random.Random(self.random_seed + epoch)
        self.current_impostor_pairs = sample_from_strata(
            self._impostor_strata, self.strata_ratios, self.num_impostors_needed, rng)
        print(f"  Sampled {len(self.current_impostor_pairs):,} impostors (stratificati)")
        self._create_current_samples()

    def __getitem__(self, idx: int):
        img1_path, img2_path, label = self.samples[idx]
        return self._load_image(img1_path), self._load_image(img2_path), \
               torch.tensor(label, dtype=torch.float32)

    def get_validation_pair(self, idx: int):
        return self.samples[idx]

# %% [markdown]
# ## Dataset per Triplet: negativi stratificati per anchor (2 strati)

# %%
class StratifiedTripletDataset:
    """Equivalente stratificato di TripletDataset. Non eredita da
    BaseWriterDataset perche' TripletDataset originale non lo fa: la logica
    di campionamento e' intrinsecamente diversa (un negativo per anchor,
    non una lista di coppie etichettate)."""

    def __init__(self, writer_sources, train=True, target_size=448,
                 positive_ratio=0.5, resample_negatives_every_n_epochs=1,
                 negative_strata_ratios=None,
                 validation_strata_ratios=None, random_seed=42):
        self.writer_sources_input = writer_sources
        self.train = train
        self.target_size = target_size
        self.positive_ratio = positive_ratio
        self.resample_negatives_every_n_epochs = resample_negatives_every_n_epochs
        self.random_seed = random_seed
        self.negative_strata_ratios = negative_strata_ratios or \
            {"same_dataset": 0.5, "cross_dataset": 0.5}
        self.validation_strata_ratios = validation_strata_ratios or \
            {"iam-iam": 1/3, "rimes-rimes": 1/3, "cross": 1/3}

        self.transform = get_train_transforms(target_size) if train else get_test_transforms(target_size)

        print(f"\n{'='*60}")
        print(f"Initializing {'TRAIN' if train else 'VAL'} StratifiedTripletDataset")
        print(f"{'='*60}")
        print(f"  Writers: {len(writer_sources)}  |  Negative strata: {self.negative_strata_ratios}")

        self.writer_images, self.writer_source = load_writer_images_prefixed(writer_sources)
        self.writer_ids = list(self.writer_images.keys())

        self.all_genuine_pairs = self._generate_all_genuine_pairs()
        print(f"  Generated {len(self.all_genuine_pairs)} genuine pairs (base for triplets)")

        self._build_negative_pools()
        self._resample_triplets()
        self._create_validation_pairs()

    def _load_image(self, path):
        return self.transform(Image.open(path).convert("L"))

    def _generate_all_genuine_pairs(self):
        pairs = []
        for wid, imgs in self.writer_images.items():
            if len(imgs) >= 2:
                for img1, img2 in combinations(imgs, 2):
                    pairs.append((wid, img1, img2))
        return pairs

    def _build_negative_pools(self):
        """Per ogni writer: pool 'same_dataset' (altri writer, stessa
        sorgente) e 'cross_dataset' (qualunque writer, sorgente diversa)."""
        images_by_source = {}
        for wid, imgs in self.writer_images.items():
            images_by_source.setdefault(self.writer_source[wid], []).extend(imgs)

        self.negative_pool = {}
        for wid in self.writer_ids:
            src = self.writer_source[wid]
            own = set(self.writer_images[wid])
            same_dataset = [img for img in images_by_source.get(src, []) if img not in own]
            cross_dataset = [img for other_src, imgs in images_by_source.items()
                             if other_src != src for img in imgs]
            self.negative_pool[wid] = {"same_dataset": same_dataset, "cross_dataset": cross_dataset}

    def _resample_triplets(self, epoch: int = 0):
        rng = random.Random(self.random_seed + epoch)
        ratios = self.negative_strata_ratios
        keys, weights = list(ratios.keys()), list(ratios.values())

        self.triplets = []
        for wid, anchor_path, positive_path in self.all_genuine_pairs:
            chosen = rng.choices(keys, weights=weights, k=1)[0]
            pool = self.negative_pool[wid][chosen]
            if not pool:  # fallback se lo strato scelto e' vuoto per questo writer
                other = "cross_dataset" if chosen == "same_dataset" else "same_dataset"
                pool = self.negative_pool[wid][other]
            negative_path = rng.choice(pool)
            self.triplets.append((anchor_path, positive_path, negative_path))

        random.shuffle(self.triplets)
        print(f"  Generated {len(self.triplets)} triplets (strata: {ratios})")

    def _create_validation_pairs(self):
        genuine_samples = [(img1, img2, 1.0) for _, img1, img2 in self.all_genuine_pairs]
        strata = build_pair_strata(self.writer_ids, self.writer_images, self.writer_source)
        n_impostors_needed = int(len(genuine_samples) * (1 - self.positive_ratio) / self.positive_ratio)
        rng = random.Random(self.random_seed)
        impostor_samples = [(a, b, 0.0) for a, b in
                            sample_from_strata(strata, self.validation_strata_ratios,
                                                n_impostors_needed, rng, label="validation")]

        self.validation_pairs = genuine_samples + impostor_samples
        random.shuffle(self.validation_pairs)
        print(f"  Created {len(self.validation_pairs)} validation pairs "
              f"({len(genuine_samples)} genuine + {len(impostor_samples)} impostor, stratificati)")

    def on_epoch_end(self, epoch: int):
        if self.resample_negatives_every_n_epochs and \
           (epoch + 1) % self.resample_negatives_every_n_epochs == 0:
            print("\n🔄 Resampling triplet negatives for next epoch...")
            self._resample_triplets(epoch + 1)

    def __getitem__(self, idx: int):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        return self._load_image(anchor_path), self._load_image(positive_path), \
               self._load_image(negative_path)

    def __len__(self):
        return len(self.triplets)

    def get_validation_pair(self, idx: int):
        return self.validation_pairs[idx]

# %% [markdown]
# ## Raccolta writer IAM + RIMES, controllo collisioni, split 90/10 stratificato

# %%
def list_writer_sources(root_path, source_label):
    return [
        (os.path.join(root_path, d), source_label)
        for d in sorted(os.listdir(root_path))
        if os.path.isdir(os.path.join(root_path, d))
    ]

iam_writer_sources = list_writer_sources(IAM_PATH, "iam")
rimes_writer_sources = list_writer_sources(RIMES_PATH, "rimes")
all_writer_sources = iam_writer_sources + rimes_writer_sources

print(f"Writer IAM: {len(iam_writer_sources)}  |  RIMES: {len(rimes_writer_sources)}  "
      f"|  totale: {len(all_writer_sources)}")

name_collisions = ({Path(d).name for d, _ in iam_writer_sources} &
                    {Path(d).name for d, _ in rimes_writer_sources})
print(f"Nomi directory in comune IAM/RIMES: {len(name_collisions)} "
      f"(innocuo grazie al prefisso di sorgente nel writer_id)")

sources_only = [s for _, s in all_writer_sources]
train_writer_sources, val_writer_sources = train_test_split(
    all_writer_sources, test_size=CONFIG['val_size'],
    random_state=RANDOM_STATE, stratify=sources_only,
)
print(f"Train writers: {len(train_writer_sources)}  |  Val writers: {len(val_writer_sources)}")

# %% [markdown]
# ## Selezione di dataset/model_type/trainer in base a `CONFIG['loss_type']`

# %%
LOSS_TYPE = CONFIG['loss_type']

if LOSS_TYPE in ('bce', 'contrastive'):
    DatasetClass = StratifiedPairDataset
    dataset_kwargs = dict(strata_ratios=CONFIG['pair_strata_ratios'])
else:  # triplet
    DatasetClass = StratifiedTripletDataset
    dataset_kwargs = dict(negative_strata_ratios=CONFIG['triplet_negative_strata_ratios'])

TrainerClass = {'bce': BCETrainer, 'contrastive': ContrastiveTrainer, 'triplet': TripletTrainer}[LOSS_TYPE]
trainer_kwargs = {'margin': CONFIG['margin']} if LOSS_TYPE in ('contrastive', 'triplet') else {}

common_kwargs = dict(
    target_size=CONFIG['target_size'],
    positive_ratio=CONFIG['positive_ratio'],
    resample_negatives_every_n_epochs=CONFIG['resample_negatives_every_n_epochs'],
    random_seed=RANDOM_STATE,
)

train_dataset = DatasetClass(train_writer_sources, train=True, **common_kwargs, **dataset_kwargs)
val_dataset = DatasetClass(val_writer_sources, train=False, **common_kwargs, **dataset_kwargs)

# %% [markdown]
# ### Verifica finale: zero genuine pair cross-dataset

# %%
def assert_no_cross_dataset_genuine(dataset, name):
    bad = []
    for writer_id, img1, img2 in dataset.all_genuine_pairs:
        img1_is_iam = "iam_processed" in img1.replace("\\", "/")
        img2_is_iam = "iam_processed" in img2.replace("\\", "/")
        if img1_is_iam != img2_is_iam:
            bad.append((writer_id, img1, img2))
    print(f"[{name}] Genuine pairs: {len(dataset.all_genuine_pairs):,}  |  cross-dataset: {len(bad)}")
    assert not bad, f"Trovate genuine pairs cross-dataset in {name}! Esempio: {bad[0]}"

assert_no_cross_dataset_genuine(train_dataset, "train")
assert_no_cross_dataset_genuine(val_dataset, "val")

# %% [markdown]
# ## DataLoader
#
# Per Triplet uso `DataLoader` diretto (stesso pattern di
# `create_triplet_kfold_dataloaders` nel repo); per BCE/Contrastive riuso
# `_make_loader` per restare coerente col resto del progetto.

# %%
if LOSS_TYPE == 'triplet':
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                               num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'], pin_memory=True)
else:
    train_loader = _make_loader(train_dataset, CONFIG['batch_size'], shuffle=True,
                                 num_workers=CONFIG['num_workers'], random_state=RANDOM_STATE, drop_last=True)
    val_loader = _make_loader(val_dataset, CONFIG['batch_size'], shuffle=False,
                               num_workers=CONFIG['num_workers'], random_state=RANDOM_STATE)

print(f"Train samples: {len(train_dataset):,}  |  Val samples: {len(val_dataset):,}")

# %% [markdown]
# ## Modello + Trainer

# %%
model = get_model(
    CONFIG['backbone'], model_type=LOSS_TYPE, in_channels=1,
    embedding_dim=CONFIG['embedding_dim'],   # per BCE viene rimappato su projection_dim
    freeze_backbone_layers=CONFIG['freeze_backbone_layers'],
    dropout=CONFIG['dropout'],
)

trainer = TrainerClass(
    model=model, model_name=EXP_NAME, device=device,
    results_dir=RESULTS_DIR, **trainer_kwargs,
)

# %% [markdown]
# ## Training
#
# Nessun `test_dataset`: la validazione finale (soglie per la demo) usa il
# 10% di `val_dataset`, stratificato IAM/RIMES fin dallo split.

# %%
history, final_metrics = trainer.train(
    train_loader=train_loader, val_loader=val_loader, val_dataset=val_dataset,
    test_dataset=None, epochs=CONFIG['epochs'], patience=CONFIG['patience'], fold=None,
)

trainer.cleanup()

# %% [markdown]
# ## Riepilogo per l'uso nella demo

# %%
best_ckpt = os.path.join(RESULTS_DIR, f"{EXP_NAME}_best.pth")
metrics_csv = os.path.join(RESULTS_DIR, f"{EXP_NAME}_final_metrics.csv")

print(f"\n{'='*70}\nMODELLO PRONTO PER LA DEMO ({LOSS_TYPE})\n{'='*70}")
print(f"Checkpoint:      {best_ckpt}")
print(f"Metriche/soglie: {metrics_csv}")
print(f"EER:             {final_metrics['eer']:.4f}")
print(f"EER threshold:   {final_metrics['eer_threshold']:.4f}")
print(f"AUC:             {final_metrics['auc']:.4f}")
print(f"""
Copia ENTRAMBI i file ('{EXP_NAME}_best.pth' e '{EXP_NAME}_final_metrics.csv')
accanto alla demo e lancia:

    python webcam_demo.py --checkpoint {EXP_NAME}_best.pth
""")
