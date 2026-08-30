"""
Wrapper di inferenza: carica il checkpoint triplet e confronta due scritture.

Lo score e' la cosine similarity fra i due embedding L2-normalizzati, esattamente
come in `BaseTrainer._evaluate_verification` del repo HandVerify; di conseguenza
le soglie salvate nei CSV degli esperimenti sono direttamente riutilizzabili.

Cambiare i pesi = passare un altro file .pth (opzione --checkpoint di tutti gli
script). L'architettura viene riconosciuta dal nome del file e le soglie vengono
lette dal CSV delle metriche di quell'esperimento, se lo si trova accanto al
checkpoint o dentro triplet_experiments/.
"""

import csv
import os
import sys

import torch
import torch.nn.functional as F

from model_defs import load_model
from preprocess import preprocess_roi, to_tensor

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Soglie di riferimento: esperimento efficientnet_b1_triplet_iam_to_iam
# (AUC 0.956, EER 0.115). Usate quando il CSV delle metriche non si trova.
DEFAULT_THRESHOLDS = {
    "eer": 0.7912201881408691,
    "far1": 0.9312598705291748,
    "far01": 0.95,   # <-- placeholder, aggiornalo con un valore di riferimento reale
}

# alias di comodo, usato da chi importa solo le chiavi disponibili
THRESHOLDS = dict(DEFAULT_THRESHOLDS)

DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "efficientnet_b1_triplet_iam_to_iam_best.pth"
)


def find_metrics_csv(checkpoint_path):
    """
    Cerca il CSV delle metriche dell'esperimento a cui appartiene il checkpoint.
    I file seguono la convenzione '<esperimento>_best.pth' / '<esperimento>_final.pth'
    e '<esperimento>_final_metrics.csv'.
    """
    name = os.path.basename(checkpoint_path)
    stem = os.path.splitext(name)[0]
    for suffix in ("_best", "_final"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                     stem + "_final_metrics.csv"),
        os.path.join(PROJECT_ROOT, "triplet_experiments", stem,
                     stem + "_final_metrics.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def thresholds_for_checkpoint(checkpoint_path):
    path = find_metrics_csv(checkpoint_path)
    if path is None:
        return dict(DEFAULT_THRESHOLDS), "soglie di riferimento IAM (CSV non trovato)"

    try:
        row = next(csv.DictReader(open(path, encoding="utf-8")))
        thresholds = {"eer": float(row["eer_threshold"])}

        # FAR target = 1%
        far1 = row.get("threshold_at_far_0.01")
        thresholds["far1"] = float(far1) if far1 not in (None, "", "None") else thresholds["eer"]

        # FAR target = 0.1%  <-- NUOVO
        far01 = row.get("threshold_at_far_0.001")
        thresholds["far01"] = float(far01) if far01 not in (None, "", "None") else thresholds["eer"]

        return thresholds, os.path.basename(path)
    except Exception as exc:
        return dict(DEFAULT_THRESHOLDS), "soglie IAM ({} non leggibile: {})".format(
            os.path.basename(path), exc)


class HandwritingVerifier:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None, mode="scan",
                 detect=True, arch=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.model = load_model(checkpoint, self.device, arch=arch)
        self.arch = self.model.arch
        self.mode = mode
        self.detect = detect
        self.thresholds, self.threshold_source = thresholds_for_checkpoint(checkpoint)

    def describe(self):
        return "{}  |  backbone: {}  |  soglie da: {}".format(
            os.path.basename(self.checkpoint), self.arch, self.threshold_source)

    def resolve_threshold(self, threshold):
        """Accetta una chiave ('eer', 'far1', ...) oppure un valore numerico."""
        if isinstance(threshold, (int, float)):
            return float(threshold)
        return self.thresholds[threshold]

    def prepare(self, roi_bgr):
        """ROI webcam -> (immagine 448 preprocessata, stage intermedi)."""
        return preprocess_roi(roi_bgr, mode=self.mode, detect=self.detect,
                              return_stages=True)

    @torch.no_grad()
    def embed(self, img_448):
        x = to_tensor(img_448).to(self.device)
        return self.model(x)

    @torch.no_grad()
    def similarity(self, img_a_448, img_b_448):
        """Cosine similarity in [-1, 1] fra le due scritture."""
        emb = self.model(torch.cat([
            to_tensor(img_a_448), to_tensor(img_b_448)
        ]).to(self.device))
        return F.cosine_similarity(emb[0:1], emb[1:2]).item()

    def verify(self, img_a_448, img_b_448, threshold="eer"):
        score = self.similarity(img_a_448, img_b_448)
        thr = self.resolve_threshold(threshold)
        return {
            "score": score,
            "threshold": thr,
            "same_writer": score >= thr,
            "margin": score - thr,
        }
