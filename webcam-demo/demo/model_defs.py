"""
Definizione stand-alone delle reti di HandVerify.

Replica esattamente `src/models/base.py` (BaseTripletNetwork,
BaseContrastiveNetwork, BaseSiameseNetwork) + i vari `*_backbones.py` del
repo lorenzo-arcioni/HandVerify, ma senza scaricare i pesi ImageNet (qui
arrivano tutti dal checkpoint .pth) e senza dipendenze dal repo.

Backbone supportati (stessa lista per tutte e tre le tipologie):
    efficientnet_b0, efficientnet_b1, efficientnet_v2_s,
    resnet18, resnet34, resnet50,
    mobilenet_v3_small, mobilenet_v3_large,
    densenet121,
    regnet_y_400mf

Tipologie supportate: triplet, contrastive, siamese.

Sia il backbone che la tipologia vengono dedotti dal nome del file del
checkpoint (convenzione del repo: "<backbone>_<tipologia>_<da>_to_<a>_..pth",
es. "efficientnet_b1_triplet_iam_to_iam_best.pth" o
"resnet18_contrastive_rimes_to_iam_best.pth"). Se il nome non aiuta, si
prova ogni combinazione finche' lo state_dict combacia esattamente. Si
puo' sempre forzare con arch=... e/o net_type=...

IMPORTANTE sulla rete Siamese: a differenza di Triplet e Contrastive, non
e' pensata per il confronto via cosine similarity tra due embedding. Il
repo la addestra come classificatore binario su coppie: incapsula le due
immagini nello stesso encoder, concatena le feature e le passa in un MLP
con uscita Sigmoid che stima direttamente P(stesso scrivente). Per un
confronto "fedele" con un checkpoint siamese va usato
`SiameseNetwork.predict_pair(imgA, imgB)`, non `get_embedding` + cosine
similarity (che qui e' comunque offerto come fallback/approssimazione,
vedi il metodo `get_embedding`, ma non e' il modo in cui la rete e' stata
addestrata).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBEDDING_DIM = 32
PROJECTION_DIM = 512

# nome -> (costruttore torchvision, dimensione delle feature in uscita)
BACKBONES = {
    "efficientnet_b0":    (models.efficientnet_b0, 1280),
    "efficientnet_b1":    (models.efficientnet_b1, 1280),
    "efficientnet_v2_s":  (models.efficientnet_v2_s, 1280),
    "resnet18":           (models.resnet18, 512),
    "resnet34":           (models.resnet34, 512),
    "resnet50":           (models.resnet50, 2048),
    "mobilenet_v3_small": (models.mobilenet_v3_small, 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, 960),
    "densenet121":        (models.densenet121, 1024),
    "regnet_y_400mf":     (models.regnet_y_400mf, 440),
}

NET_TYPES = ("triplet", "contrastive", "siamese")

_RESNET_PREFIX = "resnet"
_DENSENET_PREFIX = "densenet"
_REGNET_PREFIX = "regnet"
# tutto il resto (efficientnet*, mobilenet*) condivide lo schema features[0][0]


# --------------------------------------------------------------------------
# Encoder (condiviso dalle tre tipologie)
# --------------------------------------------------------------------------

def _to_grayscale(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Ricrea il conv con in_channels diversi (nessun weights=None da trasferire)."""
    if conv.in_channels == in_channels:
        return conv
    return nn.Conv2d(in_channels, conv.out_channels,
                     kernel_size=conv.kernel_size, stride=conv.stride,
                     padding=conv.padding, dilation=conv.dilation,
                     groups=conv.groups, bias=conv.bias is not None,
                     padding_mode=conv.padding_mode)


def _build_encoder(arch: str, in_channels: int) -> nn.Sequential:
    """Costruisce l'encoder con la stessa struttura del repo (chiavi identiche)."""
    ctor, _ = BACKBONES[arch]
    net = ctor(weights=None)

    if arch.startswith(_RESNET_PREFIX):
        net.conv1 = _to_grayscale(net.conv1, in_channels)
        return nn.Sequential(
            nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
                net.avgpool
            ),
            nn.Flatten()
        )

    if arch.startswith(_DENSENET_PREFIX):
        net.features.conv0 = _to_grayscale(net.features.conv0, in_channels)
        return nn.Sequential(
            net.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    if arch.startswith(_REGNET_PREFIX):
        net.stem[0] = _to_grayscale(net.stem[0], in_channels)
        return nn.Sequential(
            *list(net.children())[:-1],   # scarta il classifier (fc)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    # efficientnet* e mobilenet* condividono la stessa forma
    net.features[0][0] = _to_grayscale(net.features[0][0], in_channels)
    return nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())


class _FreezeMixin:
    """
    Replica _freeze_encoder_layers del repo. Non influisce sui pesi salvati
    (requires_grad non e' nello state_dict), quindi per il solo caricamento
    di un checkpoint in inferenza e' innocuo lasciare freeze_backbone_layers=0
    (default) e non chiamarlo mai.
    """

    def _freeze_encoder_layers(self, num_layers: int):
        frozen_count = 0
        for child in self.encoder[0].children():
            if frozen_count < num_layers:
                for param in child.parameters():
                    param.requires_grad = False
                frozen_count += 1
            else:
                break
        frozen_params = sum(p.numel() for p in self.encoder[0].parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder[0].parameters())
        print("Total encoder layers:", len(self.encoder[0]))
        print("  frozen {} encoder layers".format(frozen_count))
        print("  frozen parameters: {:,} / {:,} ({:.1f}%)".format(
            frozen_params, total_params,
            100 * frozen_params / total_params if total_params else 0))


# --------------------------------------------------------------------------
# Triplet
# --------------------------------------------------------------------------

class TripletNetwork(nn.Module, _FreezeMixin):
    """Encoder condiviso + testa 'fc', output L2-normalizzato a 128-D."""

    def __init__(self, arch: str = "efficientnet_b1", in_channels: int = 1,
                 embedding_dim: int = EMBEDDING_DIM,
                 freeze_backbone_layers: int = 0, dropout: float = 0.5):
        super().__init__()
        _require_arch(arch)
        self.arch = arch
        self.net_type = "triplet"
        self.encoder = _build_encoder(arch, in_channels)
        self.feature_dim = BACKBONES[arch][1]
        self.embedding_dim = embedding_dim
        if freeze_backbone_layers > 0:
            self._freeze_encoder_layers(freeze_backbone_layers)
        self.fc = _build_mlp_head(self.feature_dim, embedding_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    get_embedding = forward


# --------------------------------------------------------------------------
# Contrastive
# --------------------------------------------------------------------------

class ContrastiveNetwork(nn.Module, _FreezeMixin):
    """Encoder condiviso + testa 'projection' (NON 'fc'!), output L2-normalizzato."""

    def __init__(self, arch: str = "efficientnet_b1", in_channels: int = 1,
                 embedding_dim: int = EMBEDDING_DIM,
                 freeze_backbone_layers: int = 0, dropout: float = 0.5):
        super().__init__()
        _require_arch(arch)
        self.arch = arch
        self.net_type = "contrastive"
        self.encoder = _build_encoder(arch, in_channels)
        self.feature_dim = BACKBONES[arch][1]
        self.embedding_dim = embedding_dim
        if freeze_backbone_layers > 0:
            self._freeze_encoder_layers(freeze_backbone_layers)
        self.projection = _build_mlp_head(self.feature_dim, embedding_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _build_mlp_head(feature_dim: int, embedding_dim: int, dropout: float) -> nn.Sequential:
    """Testa condivisa da Triplet e Contrastive (stessa forma, nomi di attributo diversi)."""
    hidden1, hidden2, hidden3 = 1024, 512, 256
    return nn.Sequential(
        nn.Linear(feature_dim, hidden1),
        nn.BatchNorm1d(hidden1),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),

        nn.Linear(hidden1, hidden2),
        nn.BatchNorm1d(hidden2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),

        nn.Linear(hidden2, hidden3),
        nn.BatchNorm1d(hidden3),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout * 0.8),

        nn.Linear(hidden3, embedding_dim)
    )


# --------------------------------------------------------------------------
# Siamese
# --------------------------------------------------------------------------

class SiameseNetwork(nn.Module, _FreezeMixin):
    """
    Encoder condiviso + classificatore 'fc' su coppie concatenate.
    ATTENZIONE: non e' un embedding a cosine similarity, vedi docstring
    del modulo. forward(img1, img2) restituisce una probabilita' [0,1]
    (dopo Sigmoid) di "stesso scrivente".
    """

    def __init__(self, arch: str = "efficientnet_b1", in_channels: int = 1,
                 projection_dim: int = PROJECTION_DIM,
                 freeze_backbone_layers: int = 0, dropout: float = 0.5):
        super().__init__()
        _require_arch(arch)
        self.arch = arch
        self.net_type = "siamese"
        self.encoder = _build_encoder(arch, in_channels)
        self.feature_dim = BACKBONES[arch][1]
        self.projection_dim = projection_dim
        if freeze_backbone_layers > 0:
            self._freeze_encoder_layers(freeze_backbone_layers)
        self.fc = self._build_fc_block(self.feature_dim * 2, projection_dim, dropout)

    @staticmethod
    def _build_fc_block(input_dim: int, projection_dim: int, dropout: float) -> nn.Sequential:
        layer1 = projection_dim
        layer2 = projection_dim // 2
        layer3 = projection_dim // 4
        layer4 = projection_dim // 8
        layer5 = projection_dim // 16

        return nn.Sequential(
            nn.Linear(input_dim, layer1),
            nn.BatchNorm1d(layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(layer1, layer2),
            nn.BatchNorm1d(layer2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(layer2, layer3),
            nn.BatchNorm1d(layer3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.9),

            nn.Linear(layer3, layer4),
            nn.BatchNorm1d(layer4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),

            nn.Linear(layer4, layer5),
            nn.BatchNorm1d(layer5),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),

            nn.Linear(layer5, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor,
               return_embeddings: bool = False):
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        if return_embeddings:
            feat1 = F.normalize(feat1, p=2, dim=1)
            feat2 = F.normalize(feat2, p=2, dim=1)
            return feat1, feat2
        combined = torch.cat([feat1, feat2], dim=1)
        return self.fc(combined)

    def predict_pair(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Modo corretto/fedele al training: probabilita' [0,1] via classificatore."""
        with torch.no_grad():
            return self.forward(img1, img2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback SOLO per compatibilita' di interfaccia con Triplet/Contrastive
        (feature grezze dell'encoder, L2-normalizzate). Il classificatore 'fc'
        NON viene usato qui: la rete non e' stata addestrata per essere
        confrontata cosi', quindi i punteggi di cosine similarity ottenuti
        da questo embedding sono solo un'APPROSSIMAZIONE, non equivalenti a
        predict_pair().
        """
        return F.normalize(self.forward_one(x), p=2, dim=1)


def _require_arch(arch: str):
    if arch not in BACKBONES:
        raise ValueError("Backbone '{}' non supportato. Disponibili: {}".format(
            arch, ", ".join(sorted(BACKBONES))))


_NET_CLASSES = {
    "triplet": TripletNetwork,
    "contrastive": ContrastiveNetwork,
    "siamese": SiameseNetwork,
}


def build_network(arch: str, net_type: str, in_channels: int = 1):
    if net_type not in _NET_CLASSES:
        raise ValueError("Tipologia '{}' non supportata. Disponibili: {}".format(
            net_type, ", ".join(NET_TYPES)))
    return _NET_CLASSES[net_type](arch=arch, in_channels=in_channels)


# --------------------------------------------------------------------------
# Riconoscimento dal nome del file + caricamento
# --------------------------------------------------------------------------

def guess_arch(checkpoint_path: str):
    """Ricava il backbone dal nome del file, es. 'resnet18_contrastive_...' -> 'resnet18'."""
    name = os.path.basename(checkpoint_path).lower()
    for arch in sorted(BACKBONES, key=len, reverse=True):
        if arch in name:
            return arch
    return None


def guess_net_type(checkpoint_path: str):
    """Ricava la tipologia dal nome del file, es. '..._contrastive_...' -> 'contrastive'."""
    name = os.path.basename(checkpoint_path).lower()
    for net_type in NET_TYPES:
        if net_type in name:
            return net_type
    return None


def load_model(checkpoint_path: str, device: torch.device = None,
               arch: str = None, net_type: str = None, verbose: bool = True):
    """
    Costruisce la rete giusta (backbone + tipologia) e ci carica dentro lo
    state_dict del checkpoint.

    arch=None / net_type=None -> dedotti dal nome del file; se il nome non
    aiuta, o se il primo tentativo non combacia, si prova ogni combinazione
    (tipologia dedotta per prima, poi le altre) finche' una fa match esatto.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]

    if net_type is not None:
        if verbose:
            print("[model_defs] Tipologia forzata da --net-type: {}".format(net_type))
        type_candidates = [net_type]
    else:
        guess_t = guess_net_type(checkpoint_path)
        if verbose:
            print("[model_defs] Tipologia dedotta dal nome del file: {}".format(
                guess_t if guess_t else "non riconosciuta, provo tutte"))
        type_candidates = ([guess_t] if guess_t else []) + \
                          [t for t in NET_TYPES if t != guess_t]

    if arch is not None:
        if verbose:
            print("[model_defs] Backbone forzato da --arch: {}".format(arch))
        arch_candidates = [arch]
    else:
        guess_a = guess_arch(checkpoint_path)
        if verbose:
            print("[model_defs] Backbone dedotto dal nome del file: {}".format(
                guess_a if guess_a else "non riconosciuto, provo tutti"))
        arch_candidates = ([guess_a] if guess_a else []) + \
                          [a for a in sorted(BACKBONES) if a != guess_a]

    errors = []
    for t in type_candidates:
        for a in arch_candidates:
            model = build_network(a, t)
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                errors.append("  {}/{}: {}".format(t, a, str(exc).split("\n")[0]))
                continue
            if verbose:
                print("[model_defs] Checkpoint caricato correttamente: "
                     "tipologia={}, backbone={}".format(t, a))
            model.to(device).eval()
            return model

    raise RuntimeError(
        "Nessuna combinazione tipologia/backbone combacia con {}.\n"
        "Tentativi:\n{}".format(os.path.basename(checkpoint_path), "\n".join(errors))
    )