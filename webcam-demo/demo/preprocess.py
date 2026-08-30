"""
Preprocessing delle immagini da webcam, allineato alla pipeline usata per IAM
(notebooks/Datasets Preprocessing.ipynb del repo HandVerify):

    grayscale -> binarizzazione Otsu -> tight crop sul testo ->
    padding a quadrato su sfondo bianco -> resize 448x448 -> ToTensor [0,1]

In piu', rispetto al dataset (scansioni pulite), qui serve compensare la
webcam: rimozione dell'ombreggiatura, sbiancamento dello sfondo e ritaglio
automatico del post-it dentro la ROI.
"""

import cv2
import numpy as np
import torch

TARGET_SIZE = 448


# ----------------------------------------------------------------------------
# funzioni prese dalla pipeline IAM originale
# ----------------------------------------------------------------------------
def compute_tight_crop_coords(binary_image, padding=5, min_pixel_ratio=0.02):
    """Coordinate di crop aggressive attorno al testo: (y1, y2, x1, x2)."""
    binary_inv = 255 - binary_image
    h, w = binary_image.shape

    min_row_threshold = max(10, int(w * min_pixel_ratio))
    min_col_threshold = max(10, int(h * min_pixel_ratio))

    row_sums = np.sum(binary_inv > 0, axis=1)
    col_sums = np.sum(binary_inv > 0, axis=0)

    text_rows = np.where(row_sums > min_row_threshold)[0]
    text_cols = np.where(col_sums > min_col_threshold)[0]

    if len(text_rows) == 0 or len(text_cols) == 0:
        return None

    y1, y2 = text_rows[0], text_rows[-1]
    x1, x2 = text_cols[0], text_cols[-1]

    extra_trim = 3
    y1 = min(y1 + extra_trim, y2 - 1)
    y2 = max(y2 - extra_trim, y1 + 1)

    y1 = max(0, y1 - padding)
    y2 = min(h, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(w, x2 + padding)

    return (y1, y2, x1, x2)


def resize_to_square(image, size=TARGET_SIZE):
    """Resize mantenendo l'aspect ratio dentro un canvas quadrato bianco."""
    h, w = image.shape
    side = max(h, w)

    square = np.full((side, side), 255, dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = image

    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# adattamenti per la webcam
# ----------------------------------------------------------------------------
def remove_shading(gray, ksize=51):
    """
    Divide per lo sfondo stimato: toglie ombre e gradienti di luce.
    Lo sfondo viene stimato su un'immagine ridotta a 1/4 (il median blur su
    kernel grandi e' costoso) e poi riportato alla dimensione originale.
    """
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(3, int(ksize * 0.25) | 1)
    background = cv2.medianBlur(small, k)
    background = cv2.resize(background, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    background = np.maximum(background, 1).astype(np.float32)
    normalized = (gray.astype(np.float32) / background) * 220.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def detect_note(gray, min_area_ratio=0.05, inset=0.04):
    """
    Trova il post-it (regione chiara piu' grande) dentro la ROI.
    Ritorna (y1, y2, x1, x2) oppure None se non trova nulla di sensato.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((15, 15), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_ratio * h * w:
        return None

    x, y, cw, ch = cv2.boundingRect(c)
    if cw < 0.15 * w or ch < 0.15 * h:
        return None

    # rientra un po' per togliere bordo e ombra del post-it
    dx, dy = int(cw * inset), int(ch * inset)
    return (y + dy, y + ch - dy, x + dx, x + cw - dx)


def whiten_background(gray, binary):
    """Porta a bianco puro tutto cio' che Otsu considera sfondo (carta colorata,
    texture del post-it), lasciando i tratti d'inchiostro in grayscale."""
    out = gray.copy()
    out[binary == 255] = 255
    return out


def preprocess_roi(roi_bgr, mode="scan", detect=True, return_stages=False):
    """
    Da ritaglio della webcam (BGR) a immagine 448x448 in stile IAM.

    mode:
        "scan" -> rimozione ombre + sfondo sbiancato (consigliato con la webcam)
        "raw"  -> solo grayscale, come una scansione gia' pulita
    detect:
        True -> cerca il post-it dentro la ROI prima del crop sul testo

    Ritorna l'immagine uint8 448x448 (None se non trova testo), oppure
    (immagine, dict_di_stage) se return_stages=True.
    """
    stages = {}
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr.copy()
    stages["gray"] = gray

    if detect:
        box = detect_note(gray)
        if box is not None:
            y1, y2, x1, x2 = box
            gray = gray[y1:y2, x1:x2]
    stages["note"] = gray

    work = remove_shading(gray) if mode == "scan" else gray

    # se l'inchiostro risultasse piu' chiaro della carta (foto in negativo,
    # lavagna scura...) si inverte, il modello si aspetta nero su bianco
    if np.median(work) < 110:
        work = 255 - work

    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stages["binary"] = binary

    cleaned = whiten_background(work, binary) if mode == "scan" else work

    coords = compute_tight_crop_coords(binary, padding=5, min_pixel_ratio=0.02)
    if coords is None:
        # nessun testo trovato: si usa tutta la ROI invece di fallire
        cropped = cleaned
    else:
        y1, y2, x1, x2 = coords
        cropped = cleaned[y1:y2, x1:x2]

    if cropped.size == 0:
        return (None, stages) if return_stages else None

    final = resize_to_square(cropped, TARGET_SIZE)
    stages["final"] = final
    stages["found_text"] = coords is not None

    return (final, stages) if return_stages else final


def to_tensor(img_448):
    """448x448 uint8 -> tensore (1, 1, 448, 448) in [0, 1], come ToTensor()."""
    t = torch.from_numpy(img_448.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)


# ----------------------------------------------------------------------------
# diagnostica
# ----------------------------------------------------------------------------
def quality_stats(img_448):
    """
    Due indicatori per capire se l'acquisizione e' buona:
      focus = varianza del laplaciano (bassa = immagine sfocata)
      ink   = percentuale di pixel scuri (poco inchiostro = troppo poco testo)
    """
    focus = float(cv2.Laplacian(img_448, cv2.CV_64F).var())
    ink = float((img_448 < 128).mean() * 100.0)
    return focus, ink


def quality_warnings(img_448, focus_min=150.0, ink_min=2.5, ink_max=25.0):
    """Lista di problemi rilevati sull'immagine preprocessata (vuota = ok)."""
    focus, ink = quality_stats(img_448)
    warns = []
    if focus < focus_min:
        warns.append("sfocata (focus {:.0f})".format(focus))
    if ink < ink_min:
        warns.append("poco testo ({:.1f}% inchiostro)".format(ink))
    elif ink > ink_max:
        warns.append("troppo scura ({:.1f}% inchiostro)".format(ink))
    return warns


def debug_sheet(stages, height=260):
    """Composito con gli stadi del preprocessing, per capire dove sbaglia."""
    order = [("gray", "1. grayscale"), ("note", "2. post-it"),
             ("binary", "3. Otsu"), ("final", "4. 448x448")]
    tiles = []
    for key, label in order:
        img = stages.get(key)
        if img is None:
            continue
        scale = height / img.shape[0]
        tile = cv2.resize(img, (max(1, int(img.shape[1] * scale)), height))
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 26, 6, 4, 4, cv2.BORDER_CONSTANT, value=(35, 35, 35))
        cv2.putText(tile, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        return None
    return np.hstack(tiles)
