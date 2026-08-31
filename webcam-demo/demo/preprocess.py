"""
Preprocessing delle immagini da webcam, allineato alla pipeline IAM:

    grayscale -> rimozione ombreggiatura -> binarizzazione Otsu ->
    sfondo sbiancato -> tight crop sul testo -> padding a quadrato -> 448x448

Rispetto alla versione originale e' stata tolta la ricerca automatica del
post-it dentro la ROI (detect_note): il riquadro disegnato a schermo dalla
demo isola gia' il post-it, quindi quel passaggio era ridondante ed era
anche la causa principale dello "sfarfallio" dell'anteprima (bounding box
ricalcolato ad ogni frame da un contorno instabile).
"""

import cv2
import numpy as np
import torch

TARGET_SIZE = 448


def compute_tight_crop_coords(binary_image, padding=5, min_pixel_ratio=0.02):
    """Coordinate di crop attorno al testo: (y1, y2, x1, x2), o None se non trova nulla."""
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


def remove_shading(gray, ksize=51):
    """Divide per lo sfondo stimato: toglie ombre e gradienti di luce della webcam."""
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = max(3, int(ksize * 0.25) | 1)
    background = cv2.medianBlur(small, k)
    background = cv2.resize(background, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    background = np.maximum(background, 1).astype(np.float32)
    normalized = (gray.astype(np.float32) / background) * 220.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def whiten_background(gray, binary):
    """Porta a bianco puro lo sfondo (secondo Otsu), lascia l'inchiostro in grayscale."""
    out = gray.copy()
    out[binary == 255] = 255
    return out


def preprocess_roi(roi_bgr, return_stages=False):
    """
    Da ritaglio della webcam (BGR, gia' isolato dal riquadro a schermo) a
    immagine 448x448 in stile IAM. Sempre in modalita' "scan" (rimozione
    ombre + sfondo sbiancato), la piu' adatta a foto da webcam.

    Ritorna l'immagine uint8 448x448 (None se non trova testo), oppure
    (immagine, dict_di_stage) se return_stages=True.
    """
    stages = {}
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr.copy()

    work = remove_shading(gray)

    # se l'inchiostro risultasse piu' chiaro della carta si inverte: il
    # modello si aspetta tratto scuro su sfondo chiaro
    if np.median(work) < 110:
        work = 255 - work

    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = whiten_background(work, binary)

    coords = compute_tight_crop_coords(binary, padding=5, min_pixel_ratio=0.02)
    if coords is None:
        cropped = cleaned  # nessun testo trovato: usa tutta la ROI
    else:
        y1, y2, x1, x2 = coords
        cropped = cleaned[y1:y2, x1:x2]

    if cropped.size == 0:
        return (None, stages) if return_stages else None

    final = resize_to_square(cropped, TARGET_SIZE)
    stages["found_text"] = coords is not None

    return (final, stages) if return_stages else final


def to_tensor(img_448):
    """448x448 uint8 -> tensore (1, 1, 448, 448) in [0, 1]."""
    t = torch.from_numpy(img_448.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)


def quality_stats(img_448):
    """focus = varianza del laplaciano (bassa = sfocata); ink = % pixel scuri."""
    focus = float(cv2.Laplacian(img_448, cv2.CV_64F).var())
    ink = float((img_448 < 128).mean() * 100.0)
    return focus, ink


def quality_warnings(img_448, focus_min=150.0, ink_min=2.5, ink_max=25.0):
    focus, ink = quality_stats(img_448)
    warns = []
    if focus < focus_min:
        warns.append("sfocata (focus {:.0f})".format(focus))
    if ink < ink_min:
        warns.append("poco testo ({:.1f}% inchiostro)".format(ink))
    elif ink > ink_max:
        warns.append("troppo scura ({:.1f}% inchiostro)".format(ink))
    return warns