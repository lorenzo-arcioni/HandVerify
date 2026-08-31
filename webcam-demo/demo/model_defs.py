"""
Definizione stand-alone delle reti di HandVerify (solo Triplet e Contrastive:
la Siamese e' stata tolta perche' non e' usata da questa demo e comunque non
si presta al confronto via cosine similarity, vedi il repo originale).

Backbone supportati:
    efficientnet_b0, efficientnet_b1, efficientnet_v2_s,
    resnet18, resnet34, resnet50,
    mobilenet_v3_small, mobilenet_v3_large,
    densenet121, regnet_y_400mf

Sia il backbone che la tipologia (triplet/contrastive) vengono dedotti dal
nome del file del checkpoint (es. "resnet18_contrastive_mixed_iam_rimes_..pth"
-> resnet18 + contrastive). Se il nome non aiuta, si prova ogni combinazione
finche' lo state_dict combacia esattamente. Si puo' sempre forzare con
arch=... / net_type=....
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBEDDING_DIM = 32

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

NET_TYPES = ("triplet", "contrastive")

_RESNET_PREFIX = "resnet"
_DENSENET_PREFIX = "densenet"
_REGNET_PREFIX = "regnet"


def _to_grayscale(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    return nn.Conv2d(in_channels, conv.out_channels,
                     kernel_size=conv.kernel_size, stride=conv.stride,
                     padding=conv.padding, dilation=conv.dilation,
                     groups=conv.groups, bias=conv.bias is not None,
                     padding_mode=conv.padding_mode)


def _build_encoder(arch: str, in_channels: int) -> nn.Sequential:
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
        return nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    if arch.startswith(_REGNET_PREFIX):
        net.stem[0] = _to_grayscale(net.stem[0], in_channels)
        return nn.Sequential(*list(net.children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten())

    # efficientnet* e mobilenet* condividono la stessa forma
    net.features[0][0] = _to_grayscale(net.features[0][0], in_channels)
    return nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())


def _build_mlp_head(feature_dim: int, embedding_dim: int, dropout: float) -> nn.Sequential:
    hidden1, hidden2, hidden3 = 1024, 512, 256
    return nn.Sequential(
        nn.Linear(feature_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(hidden2, hidden3), nn.BatchNorm1d(hidden3), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.8),
        nn.Linear(hidden3, embedding_dim)
    )


class TripletNetwork(nn.Module):
    """Encoder condiviso + testa 'fc', output L2-normalizzato."""

    def __init__(self, arch="efficientnet_b1", in_channels=1, embedding_dim=EMBEDDING_DIM, dropout=0.5):
        super().__init__()
        _require_arch(arch)
        self.arch = arch
        self.net_type = "triplet"
        self.encoder = _build_encoder(arch, in_channels)
        self.fc = _build_mlp_head(BACKBONES[arch][1], embedding_dim, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class ContrastiveNetwork(nn.Module):
    """Encoder condiviso + testa 'projection' (non 'fc'), output L2-normalizzato."""

    def __init__(self, arch="efficientnet_b1", in_channels=1, embedding_dim=EMBEDDING_DIM, dropout=0.5):
        super().__init__()
        _require_arch(arch)
        self.arch = arch
        self.net_type = "contrastive"
        self.encoder = _build_encoder(arch, in_channels)
        self.projection = _build_mlp_head(BACKBONES[arch][1], embedding_dim, dropout)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)


def _require_arch(arch):
    if arch not in BACKBONES:
        raise ValueError("Backbone '{}' non supportato. Disponibili: {}".format(
            arch, ", ".join(sorted(BACKBONES))))


_NET_CLASSES = {"triplet": TripletNetwork, "contrastive": ContrastiveNetwork}


def build_network(arch, net_type, in_channels=1):
    if net_type not in _NET_CLASSES:
        raise ValueError("Tipologia '{}' non supportata. Disponibili: {}".format(
            net_type, ", ".join(NET_TYPES)))
    return _NET_CLASSES[net_type](arch=arch, in_channels=in_channels)


def guess_arch(checkpoint_path):
    name = os.path.basename(checkpoint_path).lower()
    for arch in sorted(BACKBONES, key=len, reverse=True):
        if arch in name:
            return arch
    return None


def guess_net_type(checkpoint_path):
    name = os.path.basename(checkpoint_path).lower()
    for net_type in NET_TYPES:
        if net_type in name:
            return net_type
    return None


def load_model(checkpoint_path, device=None, arch=None, net_type=None, verbose=True):
    """
    Costruisce la rete giusta (backbone + tipologia) e ci carica lo state_dict.
    arch/net_type=None -> dedotti dal nome del file; se il primo tentativo
    non combacia si prova ogni combinazione finche' una fa match esatto.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]

    if net_type is not None:
        type_candidates = [net_type]
    else:
        guess_t = guess_net_type(checkpoint_path)
        type_candidates = ([guess_t] if guess_t else []) + [t for t in NET_TYPES if t != guess_t]

    if arch is not None:
        arch_candidates = [arch]
    else:
        guess_a = guess_arch(checkpoint_path)
        arch_candidates = ([guess_a] if guess_a else []) + [a for a in sorted(BACKBONES) if a != guess_a]

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
                print("[model_defs] Checkpoint caricato: tipologia={}, backbone={}".format(t, a))
            model.to(device).eval()
            return model

    raise RuntimeError(
        "Nessuna combinazione tipologia/backbone combacia con {}.\nTentativi:\n{}".format(
            os.path.basename(checkpoint_path), "\n".join(errors)))