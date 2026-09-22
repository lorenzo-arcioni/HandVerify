"""
Grad-CAM per reti di verifica pairwise (BCE Siamese / Contrastive / Triplet).
Non richiede una classe di output: usa come "target" lo score di
similarita' prodotto dalla coppia di immagini.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def find_last_conv2d(module: torch.nn.Module) -> torch.nn.Conv2d:
    """
    Trova l'ultimo layer Conv2d incontrato scorrendo module.modules()
    (che segue l'ordine di definizione, coincide con l'ordine di
    esecuzione per architetture feed-forward come le nostre).
    Funziona per ResNet/EfficientNet/MobileNet/DenseNet/RegNet senza
    bisogno di logica specifica per architettura.
    """
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("Nessun layer Conv2d trovato nel modulo fornito.")
    return last_conv


class GradCAM:
    """Grad-CAM generico agganciato a un singolo layer convoluzionale."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or find_last_conv2d(model.encoder)
        self._activations = None
        self._gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, forward_fn: Callable[[], torch.Tensor]) -> Tuple[np.ndarray, float]:
        """
        forward_fn: funzione senza argomenti che esegue il forward e
        ritorna uno scalare da cui far partire il backward.

        IMPORTANTE: l'hook forward si attiva ad OGNI passaggio nel
        target_layer, anche quelli sotto torch.no_grad(). Se forward_fn
        fa piu' forward pass nel target_layer (es. uno per l'immagine
        "congelata" e uno per quella "attiva"), assicurati che l'ULTIMO
        a essere eseguito sia quello che richiede grad, altrimenti
        self._activations verra' sovrascritta con quella sbagliata.
        """
        self.model.zero_grad(set_to_none=True)
        score = forward_fn()

        if score.numel() != 1:
            raise ValueError("Lo score da cui fare backward deve essere scalare.")

        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Hook non popolati: il target_layer non e' stato "
                "attraversato durante forward_fn()."
            )

        activations = self._activations.detach()
        gradients = self._gradients.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()

        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()

        return cam, float(score.item())


def _embed(model: torch.nn.Module, model_type: str, img: torch.Tensor) -> torch.Tensor:
    """Estrae la feature/embedding di una singola immagine."""
    if model_type == "bce":
        return model.forward_one(img)          # feature grezza, pre-fc
    return model(img)                           # embedding L2-normalizzato (contrastive/triplet)


def _pair_score(model: torch.nn.Module, model_type: str,
                 emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    if model_type == "bce":
        combined = torch.cat([emb_a, emb_b], dim=1)
        return model.fc(combined).squeeze()
    return F.cosine_similarity(emb_a, emb_b).squeeze()


def compute_pairwise_gradcam(
    model: torch.nn.Module,
    model_type: str,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    device: torch.device,
    target_layer: Optional[torch.nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calcola due mappe Grad-CAM (per img_a e per img_b) rispetto allo score
    di verifica della coppia.

    Args:
        model: modello già in eval() e sul device giusto
        model_type: 'bce' | 'contrastive' | 'triplet'
        img_a, img_b: tensori [1, 1, H, W] già preprocessati
        device: torch.device
        target_layer: layer conv su cui agganciarsi (default: ultimo
                       Conv2d dell'encoder, trovato automaticamente)

    Returns:
        (cam_a, cam_b, score): mappe 2D in [0,1] alla risoluzione del
        target layer (da ridimensionare all'immagine originale) + lo
        score usato per il backward.
    """
    model.eval()
    img_a = img_a.to(device)
    img_b = img_b.to(device)

    cam_engine = GradCAM(model, target_layer)

    def forward_a():
        # img_b PRIMA (no_grad, non lascia gradiente), img_a DOPO (con grad):
        # cosi' l'hook forward, che si attiva su OGNI passaggio, salva per
        # ultimo le attivazioni di img_a, che sono quelle coerenti col
        # backward hook (l'unico ramo che richiede gradiente).
        with torch.no_grad():
            emb_b = _embed(model, model_type, img_b)
        emb_a = _embed(model, model_type, img_a)
        return _pair_score(model, model_type, emb_a, emb_b)

    cam_a, score_a = cam_engine(forward_a)

    def forward_b():
        with torch.no_grad():
            emb_a = _embed(model, model_type, img_a)
        emb_b = _embed(model, model_type, img_b)
        return _pair_score(model, model_type, emb_a, emb_b)

    cam_b, score_b = cam_engine(forward_b)
    cam_engine.remove_hooks()

    assert abs(score_a - score_b) < 1e-4, "score incoerente tra i due backward"

    return cam_a, cam_b, score_a


def overlay_cam_on_image(gray_img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Sovrappone la CAM (2D, [0,1], qualunque risoluzione) su un'immagine
    in scala di grigi uint8. Ritorna un'immagine BGR uint8.
    """
    import cv2

    h, w = gray_img_uint8.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(gray_img_uint8, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(heatmap, alpha, base_bgr, 1 - alpha, 0)