"""
Wrapper di inferenza: carica il checkpoint e confronta due scritture.

Lo score e' la cosine similarity fra i due embedding L2-normalizzati.
Le soglie vengono lette dal CSV delle metriche dell'esperimento se lo si
trova accanto al checkpoint o dentro triplet_experiments/, altrimenti si
usano dei valori di riferimento di default.
"""

import csv
import os
import sys

import torch
import torch.nn.functional as F

from model_defs import load_model
from preprocess import preprocess_manual_roi, to_tensor

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Soglie di riferimento: esperimento efficientnet_b1_triplet_iam_to_iam
# (AUC 0.956, EER 0.115). Usate solo come fallback quando il CSV delle
# metriche del checkpoint scelto non si trova.
#   eer   -> eer_threshold               (Equal Error Rate)
#   far1  -> threshold_at_far_0.01       (soglia per FAR = 1%)
#   far01 -> threshold_at_far_0.001      (soglia per FAR = 0.1%)
DEFAULT_THRESHOLDS = {
    "eer": 0.7912201881408691,
    "far1": 0.9312598705291748,
    "far01": 0.9312598705291748,  # nessun riferimento reale: fallback = far1
}

DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "efficientnet_b1_triplet_iam_to_iam_best.pth"
)


def find_metrics_csv(checkpoint_path):
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

        far1 = row.get("threshold_at_far_0.01")   # soglia per FAR = 1%
        thresholds["far1"] = float(far1) if far1 not in (None, "", "None") else thresholds["eer"]

        far01 = row.get("threshold_at_far_0.001")  # soglia per FAR = 0.1%
        thresholds["far01"] = float(far01) if far01 not in (None, "", "None") else thresholds["far1"]

        return thresholds, os.path.basename(path)
    except Exception as exc:
        return dict(DEFAULT_THRESHOLDS), "soglie IAM ({} non leggibile: {})".format(
            os.path.basename(path), exc)


class HandwritingVerifier:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None, arch=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.model = load_model(checkpoint, self.device, arch=arch)
        self.arch = self.model.arch
        self.thresholds, self.threshold_source = thresholds_for_checkpoint(checkpoint)

    def describe(self):
        return "{}  |  backbone: {}  |  soglie da: {}".format(
            os.path.basename(self.checkpoint), self.arch, self.threshold_source)

    def resolve_threshold(self, threshold):
        if isinstance(threshold, (int, float)):
            return float(threshold)
        return self.thresholds[threshold]

    def prepare(self, roi_bgr):
        """ROI disegnata a mano -> immagine 448x448 preprocessata (None se vuota)."""
        return preprocess_manual_roi(roi_bgr)

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