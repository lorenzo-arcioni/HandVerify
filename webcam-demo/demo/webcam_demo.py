"""
Demo live: verifica biometrica della scrittura a mano da webcam.

Due modalita' di acquisizione:

  DOPPIA  (default) i due post-it stanno insieme nel frame, uno per riquadro.
          Comoda per la demo, ma ogni post-it occupa meta' inquadratura.

  SINGOLA un post-it alla volta che riempie tutto il frame: piu' pixel sul
          tratto e messa a fuoco migliore. Si cattura A, si cambia post-it,
          si cattura B, poi si confronta. E' la modalita' da usare se i
          risultati in doppia non convincono.

Architettura del modello:
  Il backbone (efficientnet_b1, resnet18, mobilenet_v3_large, densenet121,
  regnet_y_400mf, ...) viene dedotto automaticamente dal nome del file del
  checkpoint, es. "efficientnet_b1_triplet_iam_to_iam_best.pth" ->
  efficientnet_b1. Se il nome non e' chiaro (o si vuole forzare comunque
  un'architettura diversa da quella dedotta), si puo' usare --arch.

Tasti:
    SPAZIO        cattura (doppia: entrambi + confronto | singola: slot libero)
    1 / 2         cattura nello slot A / nello slot B
    INVIO         confronta gli slot A e B
    M             cambia modalita' (doppia <-> singola)
    C             svuota gli slot
    R             torna al live
    B             cambia preprocessing (scan <-> raw)
    D             attiva/disattiva il ritaglio automatico del post-it
    T             cambia soglia
    X             salva il dettaglio degli stadi di preprocessing
    S             salva l'ultimo confronto in demo/captures/
    Q / ESC       esci
"""

import argparse
import os
import time

import cv2
import numpy as np

from engine import HandwritingVerifier, DEFAULT_THRESHOLDS, DEFAULT_CHECKPOINT

# Usato solo per stampare a schermo l'architettura dedotta *prima* di
# costruire il modello (HandwritingVerifier/engine.py fa comunque la sua
# stessa deduzione internamente quando arch=None). Se il file con la
# definizione dei backbone si chiama diversamente, aggiornare l'import qui
# sotto di conseguenza.
from model_defs import guess_arch, guess_net_type, BACKBONES, NET_TYPES
from preprocess import debug_sheet, quality_stats, quality_warnings

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 220, 100)
RED = (60, 60, 235)
AMBER = (0, 190, 250)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
DARK = (35, 35, 35)

CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

NO_KEY = 255          # valore restituito da waitKey quando non e' stato premuto nulla
DEBUG_KEYS = False    # True per stampare il codice di ogni tasto premuto


def read_key(delay=20):
    """
    Legge un tasto dalla finestra OpenCV.

    Normalizza le maiuscole: con il Caps Lock attivo (o tenendo Shift) waitKey
    restituisce 'Q' = 81 invece di 'q' = 113, e nessuna scorciatoia a lettera
    verrebbe riconosciuta. Il delay di 20 ms lascia alla finestra il tempo di
    smaltire gli eventi senza rallentare il video.
    """
    key = cv2.waitKey(delay) & 0xFF
    if key == NO_KEY:
        return NO_KEY
    if ord('A') <= key <= ord('Z'):       # maiuscola -> minuscola
        key += 32
    if DEBUG_KEYS:
        print("  [tasto] codice {} = {!r}".format(
            key, chr(key) if 32 <= key < 127 else ""))
    return key


def roi_boxes(w, h, mode):
    """Riquadri di acquisizione: due affiancati in doppia, uno solo in singola."""
    top, bottom = int(0.14 * h), int(0.93 * h)
    if mode == "singola":
        return [(int(0.06 * w), top, int(0.94 * w), bottom)]
    return [(int(0.03 * w), top, int(0.48 * w), bottom),
            (int(0.52 * w), top, int(0.97 * w), bottom)]


PANEL_W = 190


def draw_thumb(canvas, img_448, x, y, size, label, sublabel=None, sub_color=GREY):
    """Miniatura di un'immagine preprocessata con etichetta."""
    thumb = cv2.cvtColor(cv2.resize(img_448, (size, size)), cv2.COLOR_GRAY2BGR)
    canvas[y:y + size, x:x + size] = thumb
    cv2.rectangle(canvas, (x - 1, y - 1), (x + size, y + size), GREY, 1)
    cv2.putText(canvas, label, (x, y - 7), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
    if sublabel:
        cv2.putText(canvas, sublabel, (x, y + size + 14), FONT, 0.38,
                    sub_color, 1, cv2.LINE_AA)


def side_panel(height, preview, slots):
    """Colonna a destra del video: cosa vede la rete adesso e cosa e' catturato."""
    panel = np.full((height, PANEL_W, 3), (25, 25, 25), np.uint8)
    size = PANEL_W - 32
    x = 16
    y = 34

    if preview is not None:
        focus, ink = quality_stats(preview)
        warns = quality_warnings(preview)
        draw_thumb(panel, preview, x, y, size, "anteprima live",
                   "focus {:.0f}  inchiostro {:.1f}%".format(focus, ink),
                   AMBER if warns else GREEN)
    else:
        cv2.putText(panel, "anteprima live", (x, y - 7), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (x, y), (x + size, y + size), (60, 60, 60), 1)

    y += size + 42
    for name in ("A", "B"):
        img = slots[name]
        if img is not None:
            draw_thumb(panel, img, x, y, size, "slot " + name, "catturato", GREEN)
        else:
            cv2.putText(panel, "slot " + name, (x, y - 7), FONT, 0.42, GREY, 1, cv2.LINE_AA)
            cv2.rectangle(panel, (x, y), (x + size, y + size), (60, 60, 60), 1)
            cv2.putText(panel, "vuoto", (x, y + size + 14), FONT, 0.38,
                        (110, 110, 110), 1, cv2.LINE_AA)
        y += size + 42

    return panel


def draw_live_overlay(frame, boxes, verifier, thr_key, acq_mode, slots, preview):
    out = frame.copy()
    h, w = out.shape[:2]

    labels = ("INQUADRA QUI",) if acq_mode == "singola" else ("SCRITTURA A", "SCRITTURA B")
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(out, (x1, y1), (x2, y2), GREY, 2)
        cv2.putText(out, label, (x1 + 8, y1 - 12), FONT, 0.6, DARK, 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1 + 8, y1 - 12), FONT, 0.6, WHITE, 1, cv2.LINE_AA)

    band = int(0.09 * h)
    out[0:band, :] = cv2.addWeighted(out[0:band, :], 0.35,
                                     np.zeros((band, w, 3), np.uint8), 0.65, 0)
    cv2.putText(out, "SPAZIO = cattura   1/2 = slot A/B   INVIO = confronta   "
                     "M = modalita'   X = debug   Q = esci",
                (14, int(0.055 * h)), FONT, 0.52, WHITE, 1, cv2.LINE_AA)

    status = "modalita': {} | slot: A={} B={} | preproc: {} | auto-crop: {} | soglia: {} ({:.3f})".format(
        acq_mode,
        "ok" if slots["A"] is not None else "-",
        "ok" if slots["B"] is not None else "-",
        verifier.mode, "on" if verifier.detect else "off",
        thr_key, verifier.thresholds[thr_key])
    cv2.putText(out, status, (14, h - 12), FONT, 0.52, WHITE, 1, cv2.LINE_AA)

    return np.hstack([out, side_panel(h, preview, slots)])


def score_bar(width, score, threshold, ok):
    """Barra della cosine similarity da -1 a 1 con il marker della soglia."""
    bar = np.full((70, width, 3), DARK, np.uint8)
    pad = 30
    inner = width - 2 * pad

    def to_x(value):
        return int(pad + (value + 1.0) / 2.0 * inner)

    cv2.rectangle(bar, (pad, 26), (pad + inner, 44), (70, 70, 70), -1)
    x_score = max(pad, min(to_x(score), pad + inner))
    cv2.rectangle(bar, (pad, 26), (x_score, 44), GREEN if ok else RED, -1)

    tx = to_x(threshold)
    cv2.line(bar, (tx, 18), (tx, 52), AMBER, 2)
    cv2.putText(bar, "soglia", (tx - 22, 14), FONT, 0.4, AMBER, 1, cv2.LINE_AA)
    cv2.putText(bar, "-1", (6, 42), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, "+1", (width - 24, 42), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    cv2.putText(bar, "cos = {:+.4f}".format(score), (pad, 66), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return bar


def build_result_view(prep_a, prep_b, result, mode, thr_key, width=1000):
    """Schermata di esito: le due immagini che vede la rete + verdetto."""
    tile = 330
    a = cv2.cvtColor(cv2.resize(prep_a, (tile, tile)), cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(cv2.resize(prep_b, (tile, tile)), cv2.COLOR_GRAY2BGR)

    ok = result["same_writer"]
    color = GREEN if ok else RED

    canvas = np.full((tile + 240, width, 3), DARK, np.uint8)

    gap = (width - 2 * tile) // 3
    for img, prep, x, label in ((a, prep_a, gap, "A"), (b, prep_b, 2 * gap + tile, "B")):
        canvas[95:95 + tile, x:x + tile] = img
        cv2.rectangle(canvas, (x - 2, 93), (x + tile + 2, 95 + tile + 2), GREY, 1)
        focus, ink = quality_stats(prep)
        cv2.putText(canvas, "{}   focus {:.0f} | inchiostro {:.1f}%".format(label, focus, ink),
                    (x + 4, 88), FONT, 0.5, GREY, 1, cv2.LINE_AA)

    verdict = "CALLIGRAFIA UGUALE" if ok else "CALLIGRAFIE DIVERSE"
    cv2.rectangle(canvas, (0, 0), (width, 62), color, -1)
    cv2.putText(canvas, verdict, (24, 44), FONT, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, "margine dalla soglia: {:+.4f}".format(result["margin"]),
                (width - 340, 40), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    canvas[tile + 105:tile + 175, :] = score_bar(width, result["score"],
                                                 result["threshold"], ok)

    warns = []
    for label, prep in (("A", prep_a), ("B", prep_b)):
        for wmsg in quality_warnings(prep):
            warns.append("{}: {}".format(label, wmsg))
    if warns:
        cv2.putText(canvas, "qualita' scarsa -> " + "   ".join(warns),
                    (30, tile + 196), FONT, 0.5, AMBER, 1, cv2.LINE_AA)

    cv2.putText(canvas, "preproc: {}   soglia {} = {:.4f}   |   "
                        "R = live, X = debug, S = salva, Q = esci".format(
                            mode, thr_key, result["threshold"]),
                (30, tile + 224), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    return canvas


def capture(verifier, frame, box, name):
    """Preprocessa una ROI e stampa gli avvisi di qualita'."""
    x1, y1, x2, y2 = box
    out, stages = verifier.prepare(frame[y1:y2, x1:x2])
    if out is None:
        print("  [{}] nessuna scrittura leggibile.".format(name))
        return None, None
    if not stages.get("found_text", True):
        print("  [{}] testo non rilevato, uso tutto il riquadro.".format(name))
    for wmsg in quality_warnings(out):
        print("  [{}] attenzione: {}".format(name, wmsg))
    return out, stages


def announce_architecture(checkpoint_path, forced_arch, forced_net_type):
    """
    Stampa a schermo, prima ancora di costruire il modello, quale backbone e
    quale tipologia (triplet/contrastive/siamese) verranno usati: quelli
    forzati da riga di comando, oppure quelli dedotti dal nome del file del
    checkpoint. E' solo informativo: il caricamento vero e proprio (con
    eventuale fallback su tutte le combinazioni) avviene dentro
    HandwritingVerifier / model_defs.load_model.
    """
    name = os.path.basename(checkpoint_path)

    if forced_net_type is not None:
        print("Tipologia forzata da riga di comando: {}".format(forced_net_type))
    else:
        guess_t = guess_net_type(checkpoint_path)
        if guess_t is not None:
            print("Tipologia dedotta dal nome del checkpoint '{}': {}".format(name, guess_t))
        else:
            print("Impossibile dedurre la tipologia dal nome '{}'; "
                  "verranno provate triplet/contrastive/siamese.".format(name))

    if forced_arch is not None:
        print("Backbone forzato da riga di comando: {}".format(forced_arch))
    else:
        guess_a = guess_arch(checkpoint_path)
        if guess_a is not None:
            print("Backbone dedotto dal nome del checkpoint '{}': {}".format(name, guess_a))
        else:
            print("Impossibile dedurre il backbone dal nome '{}'; "
                  "verranno provati tutti.".format(name))

    if forced_net_type == "siamese" or (forced_net_type is None and guess_net_type(checkpoint_path) == "siamese"):
        print("ATTENZIONE: checkpoint 'siamese' rilevato/forzato. Questa demo confronta "
              "embedding via cosine similarity, ma la rete siamese e' stata addestrata "
              "come classificatore su coppie. Il confronto potrebbe non essere accurato "
              "a meno che engine.py non usi SiameseNetwork.predict_pair() invece della "
              "cosine similarity per questa tipologia.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mode", choices=["scan", "raw"], default="scan")
    ap.add_argument("--threshold", choices=list(DEFAULT_THRESHOLDS), default="eer")
    ap.add_argument("--threshold-value", type=float,
                    help="soglia numerica personalizzata (da calibrate.py)")
    ap.add_argument("--acquisition", choices=["doppia", "singola"], default="doppia")
    ap.add_argument("--arch", choices=sorted(BACKBONES), default=None,
                    help="forza il backbone invece di dedurlo dal nome del "
                    "checkpoint (es. 'efficientnet_b1_triplet_iam_to_iam_best.pth' "
                    "-> efficientnet_b1 automaticamente). Disponibili: {}".format(
                        ", ".join(sorted(BACKBONES))))
    ap.add_argument("--net-type", choices=NET_TYPES, default=None,
                    help="forza la tipologia di rete (triplet, contrastive, "
                    "siamese) invece di dedurla dal nome del checkpoint "
                    "(es. '..._contrastive_...' -> contrastive automaticamente). "
                    "NOTA: se il tuo engine.py non conosce ancora questo "
                    "parametro, va aggiornato per passarlo a model_defs.load_model.")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--mirror", action="store_true",
                    help="specchia l'anteprima (comodo con webcam frontali)")
    ap.add_argument("--debug-keys", action="store_true",
                    help="stampa il codice di ogni tasto premuto")
    args = ap.parse_args()

    global DEBUG_KEYS
    DEBUG_KEYS = args.debug_keys

    announce_architecture(args.checkpoint, args.arch, args.net_type)

    print("Carico il modello...")
    verifier = HandwritingVerifier(args.checkpoint, mode=args.mode,
                                   detect=not args.no_detect, arch=args.arch)
    print("Modello pronto su {} -> {}".format(verifier.device, verifier.describe()))

    thr_key = args.threshold
    if args.threshold_value is not None:
        verifier.thresholds["custom"] = args.threshold_value
        thr_key = "custom"
        print("Soglia personalizzata: {:.4f}".format(args.threshold_value))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Webcam {} non disponibile (prova --camera 0)".format(args.camera))

    win = "HandVerify - demo webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    acq_mode = args.acquisition
    slots = {"A": None, "B": None}
    stages_cache = {"A": None, "B": None}
    result_view = None
    last = None
    preview = None
    frame_idx = 0

    def compare():
        if slots["A"] is None or slots["B"] is None:
            print("  Servono entrambi gli slot: cattura con 1 e 2 (o SPAZIO).")
            return None, None
        t0 = time.time()
        res = verifier.verify(slots["A"], slots["B"], thr_key)
        print("  cos={:+.4f}  soglia={:.4f}  -> {}  ({:.0f} ms)".format(
            res["score"], res["threshold"],
            "STESSA PERSONA" if res["same_writer"] else "PERSONE DIVERSE",
            (time.time() - t0) * 1000))
        return res, build_result_view(slots["A"], slots["B"], res, verifier.mode, thr_key)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame non letto dalla webcam.")
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        boxes = roi_boxes(w, h, acq_mode)
        frame_idx += 1

        if result_view is None:
            # anteprima aggiornata ogni 4 frame per non rallentare il video
            if frame_idx % 4 == 0:
                x1, y1, x2, y2 = boxes[0]
                out, _ = verifier.prepare(frame[y1:y2, x1:x2])
                preview = out if out is not None else preview
            cv2.imshow(win, draw_live_overlay(frame, boxes, verifier, thr_key,
                                              acq_mode, slots, preview))
        else:
            cv2.imshow(win, result_view)

        # chiusura con la X della finestra: senza questo controllo il loop
        # continuerebbe a riaprirla a ogni imshow
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = read_key()

        if key in (ord('q'), 27):
            break

        elif key == 32:  # spazio
            if acq_mode == "doppia":
                got = []
                for box, name in zip(boxes, ("A", "B")):
                    out, st = capture(verifier, frame, box, name)
                    if out is None:
                        got = []
                        break
                    got.append((name, out, st))
                for name, out, st in got:
                    slots[name], stages_cache[name] = out, st
                if got:
                    res, result_view = compare()
                    if res is not None:
                        last = (frame.copy(), slots["A"], slots["B"], res)
            else:
                target = "A" if slots["A"] is None else "B"
                out, st = capture(verifier, frame, boxes[0], target)
                if out is not None:
                    slots[target], stages_cache[target] = out, st
                    print("  slot {} catturato.".format(target))
                    if slots["A"] is not None and slots["B"] is not None:
                        res, result_view = compare()
                        if res is not None:
                            last = (frame.copy(), slots["A"], slots["B"], res)

        elif key in (ord('1'), ord('2')):
            target = "A" if key == ord('1') else "B"
            box = boxes[0] if acq_mode == "singola" else boxes[0 if target == "A" else 1]
            out, st = capture(verifier, frame, box, target)
            if out is not None:
                slots[target], stages_cache[target] = out, st
                print("  slot {} catturato.".format(target))

        elif key in (13, 10):  # invio
            res, view = compare()
            if res is not None:
                result_view = view
                last = (frame.copy(), slots["A"], slots["B"], res)

        elif key == ord('m'):
            acq_mode = "singola" if acq_mode == "doppia" else "doppia"
            print("  modalita' di acquisizione: {}".format(acq_mode))
            result_view = None

        elif key == ord('c'):
            slots["A"] = slots["B"] = None
            stages_cache["A"] = stages_cache["B"] = None
            result_view = None
            print("  slot svuotati.")

        elif key == ord('r'):
            result_view = None

        elif key == ord('b'):
            verifier.mode = "raw" if verifier.mode == "scan" else "scan"
            print("  preprocessing: {}".format(verifier.mode))
            result_view = None

        elif key == ord('d'):
            verifier.detect = not verifier.detect
            print("  auto-crop post-it: {}".format("on" if verifier.detect else "off"))
            result_view = None

        elif key == ord('t'):
            keys = list(verifier.thresholds)
            thr_key = keys[(keys.index(thr_key) + 1) % len(keys)]
            print("  soglia: {} = {:.4f}".format(thr_key, verifier.thresholds[thr_key]))
            if slots["A"] is not None and slots["B"] is not None:
                res, result_view = compare()
                if res is not None:
                    last = (last[0] if last else frame.copy(), slots["A"], slots["B"], res)

        elif key == ord('x'):
            os.makedirs(CAPTURES_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            saved = False
            for name in ("A", "B"):
                if stages_cache[name] is None:
                    continue
                sheet = debug_sheet(stages_cache[name])
                if sheet is not None:
                    path = os.path.join(CAPTURES_DIR, "{}_debug_{}.png".format(stamp, name))
                    cv2.imwrite(path, sheet)
                    print("  stadi di preprocessing salvati: {}".format(path))
                    saved = True
            if not saved:
                print("  niente da salvare: cattura prima con 1 / 2 / SPAZIO.")

        elif key == ord('s') and last is not None:
            os.makedirs(CAPTURES_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            frame_img, pa, pb, res = last
            cv2.imwrite(os.path.join(CAPTURES_DIR, stamp + "_frame.png"), frame_img)
            cv2.imwrite(os.path.join(CAPTURES_DIR, stamp + "_A.png"), pa)
            cv2.imwrite(os.path.join(CAPTURES_DIR, stamp + "_B.png"), pb)
            if result_view is not None:
                cv2.imwrite(os.path.join(CAPTURES_DIR, stamp + "_result.png"), result_view)
            log_path = os.path.join(CAPTURES_DIR, "log.csv")
            new_file = not os.path.exists(log_path)
            with open(log_path, "a", encoding="utf-8") as f:
                if new_file:
                    f.write("timestamp,score,threshold,same_writer,mode,detect,acquisition\n")
                f.write("{},{:.6f},{:.6f},{},{},{},{}\n".format(
                    stamp, res["score"], res["threshold"],
                    int(res["same_writer"]), verifier.mode,
                    int(verifier.detect), acq_mode))
            print("  salvato in {} ({})".format(CAPTURES_DIR, stamp))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()