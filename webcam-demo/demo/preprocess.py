"""
Preprocessing per ROI selezionate MANUALMENTE dall'utente col mouse.

Niente piu' rilevamento automatico del foglio o dell'inchiostro (erano la
fonte di instabilita': soglie Otsu/adattive che saltano a seconda della
scena). Qui l'utente disegna gia' un rettangolo stretto intorno alla
scrittura, quindi il preprocessing si limita a:
  1) binarizzare (sfondo -> bianco puro, inchiostro -> nero)
  2) incollare su un canvas quadrato e ridimensionare a TARGET_SIZE
"""

import cv2
import numpy as np

TARGET_SIZE = 448


def make_square_resize(img, size=TARGET_SIZE):
    h, w = img.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    y = (side - h) // 2
    x = (side - w) // 2
    canvas[y:y + h, x:x + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_manual_roi(roi_bgr):
    """ROI (BGR) disegnata a mano -> immagine 448x448 binarizzata.

    Ritorna None se la ROI e' vuota (rettangolo non ancora disegnato).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # sfondo quasi bianco -> bianco puro, poi binarizzazione netta
    background = gray > 150
    gray = gray.copy()
    gray[background] = 255
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return make_square_resize(gray, TARGET_SIZE)


def quality_stats(img):
    focus = cv2.Laplacian(img, cv2.CV_64F).var()
    ink_ratio = (img < 128).mean() * 100
    return focus, ink_ratio


def to_tensor(img):
    import torch
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def quality_warnings(img):
    warnings = []
    focus, ink = quality_stats(img)
    if focus < 50:
        warnings.append("LOW_FOCUS")
    if ink < 0.5:
        warnings.append("LOW_INK")
    if ink > 40:
        warnings.append("TOO_MUCH_INK")
    return warnings