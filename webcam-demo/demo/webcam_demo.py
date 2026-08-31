"""
Demo live: verifica biometrica della scrittura a mano da webcam.

Versione semplificata: niente thread, niente anteprima preprocessata in
tempo reale, niente modalita' doppia/singola commutabile. C'e' sempre un
riquadro per la scrittura A e uno per la B; SPAZIO cattura ed elabora
entrambi e mostra subito il confronto, R torna al video live.

Il focus della webcam viene disattivato (autofocus off) subito all'avvio
per evitare che la webcam rifocalizzi da sola durante le catture. Si
regola in modo puntuale con i tasti F (piu' vicino) / G (piu' lontano):
cambia solo quando premi un tasto, non deriva mai da sola.

Tasti:
    SPAZIO   cattura ed elabora A e B, mostra il confronto
    R        torna al video live
    T        cambia soglia attiva (eer / far1% / far0.1%)
    F / G    focus manuale +/-
    Q / ESC  esci
"""

import argparse
import os
import time

import cv2
import numpy as np

from engine import HandwritingVerifier, DEFAULT_THRESHOLDS, DEFAULT_CHECKPOINT
from preprocess import quality_stats, quality_warnings

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 220, 100)
RED = (60, 60, 235)
AMBER = (0, 190, 250)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
DARK = (35, 35, 35)

FOCUS_STEP = 4
FOCUS_MIN, FOCUS_MAX = 0, 255   # range tipico DirectShow/V4L2; adattare se serve

THRESHOLD_LABELS = {"eer": "EER", "far1": "FAR 1%", "far01": "FAR 0.1%"}


def roi_boxes(w, h):
    """Due riquadri fissi affiancati, uno per scrittura."""
    top, bottom = int(0.14 * h), int(0.93 * h)
    return [(int(0.03 * w), top, int(0.48 * w), bottom),
            (int(0.52 * w), top, int(0.97 * w), bottom)]


def draw_live_overlay(frame, boxes, thr_key, thr_value, focus):
    out = frame.copy()
    h, w = out.shape[:2]

    for (x1, y1, x2, y2), label in zip(boxes, ("SCRITTURA A", "SCRITTURA B")):
        cv2.rectangle(out, (x1, y1), (x2, y2), GREY, 2)
        cv2.putText(out, label, (x1 + 8, y1 - 12), FONT, 0.6, DARK, 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1 + 8, y1 - 12), FONT, 0.6, WHITE, 1, cv2.LINE_AA)

    band = int(0.09 * h)
    out[0:band, :] = cv2.addWeighted(out[0:band, :], 0.35,
                                     np.zeros((band, w, 3), np.uint8), 0.65, 0)
    cv2.putText(out, "SPAZIO = cattura e confronta   F/G = focus   Q = esci",
                (14, int(0.055 * h)), FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    focus_txt = "auto" if focus is None else str(focus)
    status = "soglia: {} ({:.3f})   focus: {}".format(thr_key, thr_value, focus_txt)
    cv2.putText(out, status, (14, h - 12), FONT, 0.52, WHITE, 1, cv2.LINE_AA)
    return out


def score_bar(width, score, thresholds, selected_key, ok):
    """
    Barra della cosine similarity da -1 a +1, con un marker per ciascuna
    soglia disponibile (eer / far1% / far0.1%): quella attiva e' evidenziata
    in ambra sopra la barra, le altre sono tacche sottili sotto, con una
    legenda testuale a fianco per i valori numerici.
    """
    bar = np.full((110, width, 3), DARK, np.uint8)
    pad = 30
    inner = width - 2 * pad

    def to_x(value):
        return int(pad + (max(-1.0, min(1.0, value)) + 1.0) / 2.0 * inner)

    cv2.rectangle(bar, (pad, 40), (pad + inner, 58), (70, 70, 70), -1)
    x_score = max(pad, min(to_x(score), pad + inner))
    cv2.rectangle(bar, (pad, 40), (x_score, 58), GREEN if ok else RED, -1)
    cv2.putText(bar, "-1", (6, 56), FONT, 0.42, GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, "+1", (width - 26, 56), FONT, 0.42, GREY, 1, cv2.LINE_AA)

    # ordina le soglie per valore cosi' le tacche vicine non si accavallano nel testo
    items = sorted(thresholds.items(), key=lambda kv: kv[1])
    for key, val in items:
        tx = to_x(val)
        if key == selected_key:
            cv2.line(bar, (tx, 24), (tx, 66), AMBER, 2)
            cv2.putText(bar, "soglia attiva", (max(0, tx - 44), 16), FONT, 0.4, AMBER, 1, cv2.LINE_AA)
        else:
            cv2.line(bar, (tx, 34), (tx, 58), (150, 150, 150), 1)

    legend_x = pad
    legend_y = 92
    for key, val in items:
        label = THRESHOLD_LABELS.get(key, key)
        color = AMBER if key == selected_key else GREY
        text = "{} = {:.4f}".format(label, val)
        cv2.putText(bar, text, (legend_x, legend_y), FONT, 0.48, color, 1, cv2.LINE_AA)
        legend_x += 12 + 9 * len(text)

    cv2.putText(bar, "cos = {:+.4f}".format(score), (width - 190, legend_y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return bar


def build_result_view(raw_a, raw_b, img_a, img_b, result, thresholds, thr_key, width=1000):
    """Schermata di esito: crop grezzo + immagine che vede la rete, per A e B, + barra soglie."""
    tile = 300
    raw_tile = 110

    a = cv2.resize(img_a, (tile, tile))
    b = cv2.resize(img_b, (tile, tile))
    a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

    ok = result["same_writer"]
    color = GREEN if ok else RED
    header_h = 62
    bar_h = 110
    canvas_h = header_h + raw_tile + 26 + tile + 30 + bar_h + 30
    canvas = np.full((canvas_h, width, 3), DARK, np.uint8)

    gap = (width - 2 * tile) // 3
    y_raw = header_h + 14
    y_proc = y_raw + raw_tile + 26

    for raw, img, prep, x, label in ((raw_a, a, img_a, gap, "A"), (raw_b, b, img_b, 2 * gap + tile, "B")):
        # crop grezzo (quello ritagliato dal riquadro, prima del preprocessing)
        raw_resized = cv2.resize(raw, (raw_tile, raw_tile))
        rx = x + (tile - raw_tile) // 2
        canvas[y_raw:y_raw + raw_tile, rx:rx + raw_tile] = raw_resized
        cv2.rectangle(canvas, (rx - 1, y_raw - 1), (rx + raw_tile + 1, y_raw + raw_tile + 1), GREY, 1)
        cv2.putText(canvas, "{} - crop grezzo".format(label), (x, y_raw - 6), FONT, 0.42, GREY, 1, cv2.LINE_AA)

        # immagine finale 448x448 data in pasto alla rete
        canvas[y_proc:y_proc + tile, x:x + tile] = img
        cv2.rectangle(canvas, (x - 2, y_proc - 2), (x + tile + 2, y_proc + tile + 2), GREY, 1)
        focus, ink = quality_stats(prep)
        cv2.putText(canvas, "{} - dopo preprocessing (focus {:.0f} | inchiostro {:.1f}%)".format(
            label, focus, ink), (x, y_proc - 8), FONT, 0.42, GREY, 1, cv2.LINE_AA)

    verdict = "CALLIGRAFIA UGUALE" if ok else "CALLIGRAFIE DIVERSE"
    cv2.rectangle(canvas, (0, 0), (width, header_h), color, -1)
    cv2.putText(canvas, verdict, (24, 44), FONT, 1.1, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, "margine: {:+.4f}".format(result["margin"]),
                (width - 260, 40), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    y_bar = y_proc + tile + 20
    canvas[y_bar:y_bar + bar_h, :] = score_bar(width, result["score"], thresholds, thr_key, ok)

    warns = []
    for label, prep in (("A", img_a), ("B", img_b)):
        for wmsg in quality_warnings(prep):
            warns.append("{}: {}".format(label, wmsg))
    footer = "R = live   T = cambia soglia   Q = esci"
    if warns:
        footer = "qualita' scarsa -> " + "   ".join(warns) + "   |   " + footer
    cv2.putText(canvas, footer, (24, canvas_h - 10), FONT, 0.48, AMBER if warns else GREY, 1, cv2.LINE_AA)

    return canvas


def apply_focus_delta(cap, state, delta):
    """Sposta il focus manuale di un passo. L'autofocus e' gia' disattivato all'avvio."""
    state["focus"] = max(FOCUS_MIN, min(FOCUS_MAX, state["focus"] + delta))
    cap.set(cv2.CAP_PROP_FOCUS, state["focus"])
    print("  [focus] {}".format(state["focus"]))


def read_key(delay=20):
    key = cv2.waitKey(delay) & 0xFF
    if key == 255:
        return 255
    if ord('A') <= key <= ord('Z'):
        key += 32
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--threshold", choices=list(DEFAULT_THRESHOLDS), default="eer")
    ap.add_argument("--arch", default=None,
                    help="forza il backbone invece di dedurlo dal nome del checkpoint")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--focus", type=int, default=30,
                    help="valore di focus manuale iniziale (0-255 circa, dipende "
                    "dalla webcam); regolabile poi con F/G")
    args = ap.parse_args()

    print("Carico il modello...")
    verifier = HandwritingVerifier(args.checkpoint, arch=args.arch)
    print("Modello pronto su {} -> {}".format(verifier.device, verifier.describe()))
    thr_key = args.threshold

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("Webcam {} non disponibile (prova --camera 0)".format(args.camera))

    # Autofocus disattivato subito: il focus cambia solo quando lo decidi tu (F/G).
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    focus_state = {"focus": args.focus}
    print("Autofocus disattivato, focus iniziale: {}".format(args.focus))

    win = "HandVerify - demo webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    result_view = None
    last_capture = None  # (raw_a, raw_b, img_a, img_b) dell'ultimo confronto, per il tasto T

    def render_result(thr_key):
        raw_a, raw_b, img_a, img_b = last_capture
        res = verifier.verify(img_a, img_b, thr_key)
        print("  cos={:+.4f}  soglia {} ={:.4f}  -> {}".format(
            res["score"], thr_key, res["threshold"],
            "STESSA PERSONA" if res["same_writer"] else "PERSONE DIVERSE"))
        return build_result_view(raw_a, raw_b, img_a, img_b, res, verifier.thresholds, thr_key)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame non letto dalla webcam.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            boxes = roi_boxes(w, h)

            if result_view is None:
                cv2.imshow(win, draw_live_overlay(
                    frame, boxes, thr_key, verifier.thresholds[thr_key], focus_state["focus"]))
            else:
                cv2.imshow(win, result_view)

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = read_key()

            if key in (ord('q'), 27):
                break

            elif key == 32:  # SPAZIO: taglia dai riquadri, elabora A e B, poi confronta
                raws, imgs = {}, {}
                for (x1, y1, x2, y2), name in zip(boxes, ("A", "B")):
                    raws[name] = frame[y1:y2, x1:x2].copy()   # crop grezzo, prima di ogni preprocessing
                    out = verifier.prepare(raws[name])        # preprocessing sul crop appena tagliato
                    if out is None:
                        print("  [{}] nessuna scrittura leggibile.".format(name))
                    else:
                        for wmsg in quality_warnings(out):
                            print("  [{}] attenzione: {}".format(name, wmsg))
                    imgs[name] = out

                if imgs["A"] is None or imgs["B"] is None:
                    print("  Cattura fallita, riprova.")
                else:
                    t0 = time.time()
                    last_capture = (raws["A"], raws["B"], imgs["A"], imgs["B"])
                    result_view = render_result(thr_key)
                    print("  ({:.0f} ms)".format((time.time() - t0) * 1000))

            elif key == ord('r'):
                result_view = None

            elif key == ord('t'):
                keys = list(verifier.thresholds)
                thr_key = keys[(keys.index(thr_key) + 1) % len(keys)]
                print("  soglia attiva: {} = {:.4f}".format(thr_key, verifier.thresholds[thr_key]))
                if last_capture is not None and result_view is not None:
                    result_view = render_result(thr_key)

            elif key == ord('f'):
                apply_focus_delta(cap, focus_state, FOCUS_STEP)
            elif key == ord('g'):
                apply_focus_delta(cap, focus_state, -FOCUS_STEP)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()