"""
Analisi per occlusione e validazione quantitativa delle mappe di importanza
(Grad-CAM incluso) per reti di verifica pairwise (bce | contrastive | triplet).

Convenzione di segno (la stessa di gradcam.py):
    delta = score_originale - score_con_regione_occlusa
    delta > 0  (rosso): la regione SOSTIENE la somiglianza (togliendola lo score scende)
    delta < 0  (blu)  : la regione la ABBASSA
"""

from typing import Dict, List, Tuple

from tqdm.auto import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Score in batch
# --------------------------------------------------------------------------
@torch.no_grad()
def _score_batch(model, model_type: str, x: torch.Tensor, other: torch.Tensor, side: str) -> torch.Tensor:
    """
    Score di N varianti `x` [N,1,H,W] dell'immagine di `side` ('a' o 'b')
    contro l'altra immagine fissa `other` [1,1,H,W].
    bce: probabilita' del classificatore (rispetta l'ordine a,b);
    contrastive/triplet: cosine similarity.
    """
    n = x.shape[0]
    if model_type == "bce":
        o = other.expand(n, -1, -1, -1)
        a, b = (x, o) if side == "a" else (o, x)
        return model(a, b).reshape(-1)
    emb_x = model(x)
    emb_o = model(other).expand(n, -1)
    return F.cosine_similarity(emb_x, emb_o, dim=1)


def _scores_chunked(model, model_type, xs, other, side, batch_size) -> np.ndarray:
    out = []
    for i in range(0, len(xs), batch_size):
        out.append(_score_batch(model, model_type, xs[i:i + batch_size], other, side).float().cpu())
    return torch.cat(out).numpy()


def _split(img_a, img_b, side, device):
    img_a, img_b = img_a.detach().to(device), img_b.detach().to(device)
    return (img_a, img_b) if side == "a" else (img_b, img_a)


def _positions(length: int, patch: int, stride: int) -> List[int]:
    pos = list(range(0, max(length - patch, 0) + 1, stride))
    if pos[-1] + patch < length:          # copri anche il bordo
        pos.append(length - patch)
    return pos


# --------------------------------------------------------------------------
# Occlusion map
# --------------------------------------------------------------------------
def occlusion_map(
    model, model_type: str, img_a: torch.Tensor, img_b: torch.Tensor,
    side: str = "a", patch: int = 64, stride: int = 32,
    fill: float = 1.0, batch_size: int = 32,
    progress: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Occlude a finestra scorrevole l'immagine `side` tenendo fissa l'altra.

    fill: valore dello "sfondo bianco" NELLO SPAZIO DEL TENSORE (dopo il
          transform di test). Calcolalo con white_value().
    progress: se True mostra una barra tqdm sui batch di finestre
              (leave=False: sparisce a fine mappa).
    Ritorna (mappa HxW di delta mediato sulle finestre sovrapposte, score base).
    """
    model.eval()
    device = next(model.parameters()).device
    target, other = _split(img_a, img_b, side, device)
    H, W = target.shape[-2:]

    base = float(_scores_chunked(model, model_type, target, other, side, 1)[0])

    boxes = [(y, x) for y in _positions(H, patch, stride) for x in _positions(W, patch, stride)]
    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.float64)

    starts = range(0, len(boxes), batch_size)
    if progress:
        starts = tqdm(starts, desc="  finestre", leave=False)

    for s in starts:
        chunk = boxes[s:s + batch_size]
        batch = target.repeat(len(chunk), 1, 1, 1)
        for k, (y, x) in enumerate(chunk):
            batch[k, :, y:y + patch, x:x + patch] = fill
        scores = _score_batch(model, model_type, batch, other, side).float().cpu().numpy()
        for k, (y, x) in enumerate(chunk):
            acc[y:y + patch, x:x + patch] += base - scores[k]
            cnt[y:y + patch, x:x + patch] += 1

    return (acc / np.maximum(cnt, 1)).astype(np.float32), base


def normalize_signed(m: np.ndarray) -> np.ndarray:
    """Porta la mappa in [-1, 1] mantenendo lo zero (per overlay_cam_on_image)."""
    mx = np.abs(m).max()
    return m / mx if mx > 1e-12 else m


def cam_to_pixel_map(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Riporta una CAM (bassa risoluzione) a HxW. size = (H, W)."""
    return cv2.resize(cam, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# Deletion / insertion
# --------------------------------------------------------------------------
def _cell_ranking(importance: np.ndarray, cell: int) -> List[Tuple[int, int]]:
    """Celle cell x cell ordinate per importanza media decrescente."""
    H, W = importance.shape
    cells = []
    for y in range(0, H, cell):
        for x in range(0, W, cell):
            cells.append((float(importance[y:y + cell, x:x + cell].mean()), y, x))
    cells.sort(key=lambda c: -c[0])
    return [(y, x) for _, y, x in cells]


def deletion_insertion_curve(
    model, model_type: str, img_a: torch.Tensor, img_b: torch.Tensor,
    importance: np.ndarray, side: str = "a", mode: str = "deletion",
    cell: int = 32, n_steps: int = 25, fill: float = 1.0, batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    deletion : cancella (fill) le celle dalla piu' alla meno importante;
               una buona mappa fa CROLLARE lo score in fretta.
    insertion: parte da tutto vuoto e reinserisce le celle piu' importanti;
               una buona mappa fa RISALIRE lo score in fretta.
    Ritorna (frazione di celle modificate, score).
    """
    assert mode in ("deletion", "insertion")
    model.eval()
    device = next(model.parameters()).device
    target, other = _split(img_a, img_b, side, device)

    order = _cell_ranking(importance, cell)
    n = len(order)
    ks = np.unique(np.linspace(0, n, n_steps + 1).round().astype(int))

    imgs = []
    for k in ks:
        if mode == "deletion":
            img = target.clone()
            for (y, x) in order[:k]:
                img[..., y:y + cell, x:x + cell] = fill
        else:
            img = torch.full_like(target, fill)
            for (y, x) in order[:k]:
                img[..., y:y + cell, x:x + cell] = target[..., y:y + cell, x:x + cell]
        imgs.append(img)

    scores = _scores_chunked(model, model_type, torch.cat(imgs, 0), other, side, batch_size)
    return ks / n, scores


def curve_auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def compare_rankings(
    model, model_type: str, img_a, img_b, side: str,
    maps: Dict[str, np.ndarray], mode: str = "deletion",
    cell: int = 32, n_steps: int = 25, fill: float = 1.0,
    n_random: int = 5, seed: int = 0,
) -> Dict[str, dict]:
    """
    Curve per ogni mappa in `maps` piu' una baseline casuale (media di n_random).
    Ritorna {nome: {"x", "y", "auc", ("y_std")}}.
    """
    res = {}
    for name, imp in maps.items():
        x, y = deletion_insertion_curve(model, model_type, img_a, img_b, imp, side, mode, cell, n_steps, fill)
        res[name] = {"x": x, "y": y, "auc": curve_auc(x, y)}

    H, W = next(iter(maps.values())).shape
    rng = np.random.default_rng(seed)
    ys = []
    for _ in range(n_random):
        x, y = deletion_insertion_curve(model, model_type, img_a, img_b, rng.random((H, W)),
                                        side, mode, cell, n_steps, fill)
        ys.append(y)
    ys = np.stack(ys)
    res["random"] = {"x": x, "y": ys.mean(0), "y_std": ys.std(0), "auc": curve_auc(x, ys.mean(0))}
    return res


def first_crossing(x: np.ndarray, y: np.ndarray, thr: float, started_above: bool) -> float:
    """Prima frazione a cui lo score attraversa la soglia (la decisione si ribalta). nan se mai."""
    for xi, yi in zip(x, y):
        if (yi >= thr) != started_above:
            return float(xi)
    return float("nan")


# --------------------------------------------------------------------------
# Statistiche sulle mappe
# --------------------------------------------------------------------------
def positive_concentration(m: np.ndarray, top_frac: float = 0.1) -> float:
    """
    Quota della massa positiva contenuta nel top `top_frac` dei pixel.
    Alta (>0.6) = evidenza localizzata; vicina a top_frac = evidenza diffusa.
    """
    p = np.clip(m, 0, None).ravel()
    tot = p.sum()
    if tot <= 1e-12:
        return 0.0
    k = max(1, int(len(p) * top_frac))
    return float(np.sort(p)[-k:].sum() / tot)


def ink_mask(gray_uint8: np.ndarray, thr: int = 128, dilate: int = 15) -> np.ndarray:
    """Maschera booleana dell'inchiostro, dilatata (le patch coprono anche l'intorno)."""
    ink = (gray_uint8 < thr).astype(np.uint8)
    if dilate > 1:
        ink = cv2.dilate(ink, np.ones((dilate, dilate), np.uint8))
    return ink.astype(bool)


def mass_on_ink(m: np.ndarray, gray_uint8: np.ndarray, thr: int = 128, dilate: int = 15) -> Tuple[float, float]:
    """(quota della massa positiva su inchiostro, quota di area occupata dall'inchiostro).
    Se la prima e' ~ la seconda, la mappa non distingue testo da sfondo."""
    mask = ink_mask(gray_uint8, thr, dilate)
    p = np.clip(m, 0, None)
    tot = p.sum()
    return (float(p[mask].sum() / tot) if tot > 1e-12 else 0.0), float(mask.mean())