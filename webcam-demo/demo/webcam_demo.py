"""
HandVerify - webcam live + ROI disegnata a mano.

Stessa interfaccia della vecchia demo (feed live threaded, controllo
focus manuale, cambio camera, barra soglie) ma le due ROI (campione A e
campione B) non sono piu' rettangoli fissi: le disegni tu trascinando il
mouse sul video, poi premi SPAZIO per catturare ed eseguire la verifica.

Controlli:
    [Click sinistro + trascina]  disegna la ROI attiva (A o B, vedi pulsanti)
    [SPACE]   Cattura ed esegue la verifica sulle due ROI correnti (un click = una verifica)
    [R]       Reset pannello risultati
    [T]       Cambia soglia (EER / FAR 1% / FAR 0.1%)
    [F/G]     Focus +4 / -4
    [D/H]     Focus +20 / -20
    [1-5]     Focus preset (0, 64, 128, 192, 255)
    [C]       Cambia camera
    [Q/Esc]   Esci

Slider "SOGLIA BINARIZZAZIONE (GRIGIO)" nel pannello di destra: regola in
tempo reale la soglia di binarizzazione usata nel preprocessing (0-255,
default 128). Si vede subito l'effetto nel MODEL INPUT PREVIEW.

Nota: SPAZIO funziona sempre, anche senza aver prima cliccato su un altro
controllo (e' collegato a una scorciatoia globale dell'applicazione,
indipendente dal focus da tastiera).
"""
import sys
import os
import time
import cv2
import numpy as np
import argparse

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QRect, QPoint, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush, QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout,
                               QVBoxLayout, QFrame, QGraphicsDropShadowEffect, QPushButton, QSlider)

from engine import HandwritingVerifier, DEFAULT_THRESHOLDS, DEFAULT_CHECKPOINT
from preprocess import quality_stats, quality_warnings, DEFAULT_GRAY_THRESHOLD

# ==============================================================================
# CONFIGURAZIONE STILE (TEMA "BIOMETRIC SCANNER")
# ==============================================================================
DARK_BG = "#0B0F19"
PANEL_BG = "#111827"
CARD_BG = "#1F2937"
NEON_CYAN = "#00F0FF"
NEON_GREEN = "#00FF9D"
NEON_RED = "#FF3366"
NEON_ORANGE = "#FFAA00"
TEXT_PRIMARY = "#F9FAFB"
TEXT_SECONDARY = "#9CA3AF"

GLOBAL_QSS = f"""
    QMainWindow {{ background-color: {DARK_BG}; }}
    QLabel {{ color: {TEXT_PRIMARY}; font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif; }}
    #MainTitle {{ font-size: 24px; font-weight: 700; color: {NEON_CYAN}; letter-spacing: 2px; }}
    #SubTitle {{ font-size: 12px; color: {TEXT_SECONDARY}; letter-spacing: 1px; }}
    .Card {{
        background-color: {CARD_BG};
        border: 1px solid #374151;
        border-radius: 12px;
    }}
    #VerdictLabel {{ font-size: 32px; font-weight: 800; letter-spacing: 1px; }}
    #ScoreValue {{ font-size: 48px; font-weight: 800; }}
    QPushButton {{
        background-color: {NEON_CYAN}; color: {DARK_BG}; border: none; border-radius: 8px;
        font-size: 14px; font-weight: bold; padding: 10px 20px;
    }}
    QPushButton:hover {{ background-color: #66F5FF; }}
    QPushButton:pressed {{ background-color: #00C8D4; }}
    QPushButton:checked {{ background-color: {NEON_GREEN}; }}
    QSlider::groove:horizontal {{
        border: 1px solid #374151;
        height: 8px;
        background: #374151;
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: {NEON_CYAN};
        border: 1px solid {NEON_CYAN};
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        background: #66F5FF;
    }}
"""

ROI_NAMES = ("A", "B")
ROI_COLORS = {"A": NEON_CYAN, "B": NEON_GREEN}

# colori dedicati ai marker delle soglie sulla confidence bar
THRESHOLD_MARKER_COLORS = {
    "eer": "#FFAA00",     # arancio
    "far1": "#C084FC",    # viola
    "far01": "#38BDF8",   # azzurro
}


# ==============================================================================
# WORKER PER LA WEBCAM (THREAD DEDICATO) - live feed, evita il grain del
# singolo scatto (l'esposizione/guadagno del sensore si stabilizza solo
# dopo qualche frame di streaming continuo).
# ==============================================================================
class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)
    focus_changed = Signal(int)
    error_occurred = Signal(str)
    camera_opened = Signal(int)

    def __init__(self, camera_id, width, height, focus_initial):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.focus = focus_initial
        self.running = True
        self.cap = None
        self.focus_delta_queue = 0
        self.focus_absolute_queue = None
        self.camera_change_queue = None

    def run(self):
        self._open_camera(self.camera_id)

    def _open_camera(self, camera_id):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_id, backend)

        if not self.cap.isOpened():
            error_msg = f"Impossibile aprire la camera {camera_id}"
            print(f"[ERROR] {error_msg}")
            self.error_occurred.emit(error_msg)
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        self.focus_changed.emit(self.focus)

        self.camera_id = camera_id
        print(f"[Camera] Camera {camera_id} aperta correttamente")
        print(f"[Camera] Autofocus disabilitato, focus iniziale: {self.focus}")
        self.camera_opened.emit(camera_id)

        while self.running:
            if self.camera_change_queue is not None:
                new_cam_id = self.camera_change_queue
                self.camera_change_queue = None
                print(f"[Camera] Cambio a camera {new_cam_id}...")
                if not self._open_camera(new_cam_id):
                    continue

            if self.focus_absolute_queue is not None:
                self.focus = max(0, min(255, self.focus_absolute_queue))
                self.focus_absolute_queue = None
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
                self.focus_changed.emit(self.focus)
            elif self.focus_delta_queue != 0:
                self.focus = max(0, min(255, self.focus + self.focus_delta_queue))
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
                self.focus_changed.emit(self.focus)
                self.focus_delta_queue = 0

            if self.cap is None:
                self.msleep(100)
                continue

            ok, frame = self.cap.read()
            if ok:
                self.frame_ready.emit(frame)
            else:
                self.msleep(10)

        return True

    def apply_focus_delta(self, delta):
        self.focus_delta_queue = delta

    def set_focus_absolute(self, value):
        self.focus_absolute_queue = value

    def switch_camera(self, camera_id):
        self.camera_change_queue = camera_id

    def stop(self):
        self.running = False
        self.wait()
        if self.cap is not None:
            self.cap.release()


# ==============================================================================
# CANVAS VIDEO LIVE + ROI DISEGNATE A MANO
# ==============================================================================
class VideoCanvas(QLabel):
    """Feed live della camera. L'utente trascina il mouse per disegnare la
    ROI 'attiva' (self.active_roi, 'A' o 'B'); le due ROI persistono finche'
    non vengono ridisegnate, in coordinate del frame originale."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {DARK_BG}; border-radius: 16px;")
        self.setMouseTracking(True)
        self.current_frame = None
        self.pulse_opacity = 0.0
        self.pulse_direction = 1
        self.focus_value = 30
        self.gray_threshold = DEFAULT_GRAY_THRESHOLD
        self.error_message = None
        self.current_camera_id = 0

        self.active_roi = "A"
        self.roi_rects = {"A": None, "B": None}  # QRect in coordinate frame
        self._drag_start = None
        self._drag_current = None

        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._update_pulse)
        self.pulse_timer.start(30)

    # ---- feed --------------------------------------------------------------
    def update_frame(self, cv_frame):
        self.error_message = None
        self.current_frame = cv_frame
        self._render()

    def set_focus_display(self, focus_val):
        self.focus_value = focus_val
        if self.current_frame is not None:
            self._render()

    def set_gray_threshold_display(self, value):
        self.gray_threshold = value
        if self.current_frame is not None:
            self._render()

    def set_camera_id(self, camera_id):
        self.current_camera_id = camera_id
        if self.current_frame is not None:
            self._render()

    def show_error(self, error_msg):
        self.error_message = error_msg
        self.current_frame = None
        self._render()

    def _update_pulse(self):
        self.pulse_opacity += 0.03 * self.pulse_direction
        if self.pulse_opacity >= 1.0: self.pulse_direction = -1
        if self.pulse_opacity <= 0.2: self.pulse_direction = 1
        self._render()

    # ---- selezione ROI attiva ------------------------------------------------
    def set_active_roi(self, name):
        self.active_roi = name

    def get_roi_crop(self, name):
        rect = self.roi_rects.get(name)
        if self.current_frame is None or rect is None:
            return None
        h, w = self.current_frame.shape[:2]
        x1, y1 = max(0, rect.left()), max(0, rect.top())
        x2, y2 = min(w, rect.right()), min(h, rect.bottom())
        if x2 <= x1 or y2 <= y1:
            return None
        return self.current_frame[y1:y2, x1:x2].copy()

    # ---- geometria: widget <-> frame -----------------------------------------
    def _pixmap_geometry(self):
        if self.current_frame is None:
            return None
        img_h, img_w = self.current_frame.shape[:2]
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None
        scale = min(w / img_w, h / img_h)
        disp_w, disp_h = img_w * scale, img_h * scale
        off_x = (w - disp_w) / 2
        off_y = (h - disp_h) / 2
        return scale, off_x, off_y

    def _widget_to_frame(self, pos):
        geom = self._pixmap_geometry()
        if geom is None:
            return None
        scale, off_x, off_y = geom
        img_h, img_w = self.current_frame.shape[:2]
        x = (pos.x() - off_x) / scale
        y = (pos.y() - off_y) / scale
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        return QPoint(int(x), int(y))

    # ---- mouse: disegno rettangolo -------------------------------------------
    def mousePressEvent(self, event):
        if self.current_frame is None:
            return
        pt = self._widget_to_frame(event.position().toPoint())
        if pt is not None:
            self._drag_start = pt
            self._drag_current = pt

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            return
        pt = self._widget_to_frame(event.position().toPoint())
        if pt is not None:
            self._drag_current = pt

    def mouseReleaseEvent(self, event):
        if self._drag_start is None:
            return
        pt = self._widget_to_frame(event.position().toPoint())
        if pt is not None:
            self._drag_current = pt
        rect = QRect(self._drag_start, self._drag_current).normalized()
        self._drag_start = None
        self._drag_current = None
        if rect.width() > 5 and rect.height() > 5:
            self.roi_rects[self.active_roi] = rect

    # ---- disegno --------------------------------------------------------------
    def _render(self):
        if self.error_message:
            w, h = self.width(), self.height()
            q_img = QImage(w, h, QImage.Format_RGB32)
            q_img.fill(QColor(DARK_BG))

            painter = QPainter(q_img)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)

            font_title = QFont("Segoe UI", 24, QFont.Bold)
            painter.setFont(font_title)
            painter.setPen(QColor(NEON_RED))
            painter.drawText(q_img.rect(), Qt.AlignCenter, "⚠ CAMERA ERROR")

            font_msg = QFont("Segoe UI", 14)
            painter.setFont(font_msg)
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(QRectF(20, h//2 + 40, w - 40, 60), Qt.AlignCenter, self.error_message)

            font_hint = QFont("Segoe UI", 12)
            painter.setFont(font_hint)
            painter.setPen(QColor(TEXT_SECONDARY))
            painter.drawText(QRectF(20, h//2 + 120, w - 40, 40), Qt.AlignCenter,
                           "Premi [C] per cambiare camera  |  [Q] per uscire")

            painter.end()
            self.setPixmap(QPixmap.fromImage(q_img))
            return

        if self.current_frame is None:
            return

        h, w, ch = self.current_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(self.current_frame.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()

        painter = QPainter(q_img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        alpha = int(255 * self.pulse_opacity)

        # ROI confermate (in coordinate frame, quindi direttamente sul q_img)
        for name, rect in self.roi_rects.items():
            if rect is None:
                continue
            base_color = QColor(ROI_COLORS[name])
            is_active = (name == self.active_roi)
            pen_alpha = alpha if is_active else 160
            pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), pen_alpha))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            font_title = QFont("Segoe UI", 13, QFont.Bold)
            painter.setFont(font_title)
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(QRectF(rect.left(), max(0, rect.top() - 26), 160, 24),
                              Qt.AlignLeft, f"SAMPLE {name}")

        # rettangolo mentre si trascina
        if self._drag_start is not None and self._drag_current is not None:
            pen = QPen(QColor(NEON_ORANGE))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            drag_rect = QRect(self._drag_start, self._drag_current).normalized()
            painter.drawRect(drag_rect)

        font_small = QFont("Segoe UI", 10)
        painter.setFont(font_small)
        painter.setPen(QColor(156, 163, 175))
        footer_text = (f"[trascina] disegna ROI {self.active_roi} attiva  |  [SPACE] Verifica  |  [R] Reset  |  "
                       f"[F/G] Focus ±4  |  [D/H] Focus ±20  |  [C] Cambia camera  |  [T] Cambia soglia  |  "
                       f"Camera: {self.current_camera_id}  |  Focus: {self.focus_value}  |  "
                       f"Soglia grigio: {self.gray_threshold}")
        painter.drawText(QRectF(20, h - 40, w - 40, 30), Qt.AlignLeft, footer_text)

        painter.end()

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)


# ==============================================================================
# PANNELLO RISULTATI (CON ANIMAZIONI)
# ==============================================================================
class ResultDashboard(QWidget):
    def __init__(self, thresholds, active_key):
        super().__init__()
        self.setFixedWidth(380)
        self.setStyleSheet(f"background-color: {PANEL_BG}; border-left: 1px solid #1F2937;")

        self.thresholds = thresholds
        self.active_thr_key = active_key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        self.title = QLabel("ANALYSIS RESULT")
        self.title.setObjectName("MainTitle")
        layout.addWidget(self.title)

        self.subtitle = QLabel("Waiting for capture...")
        self.subtitle.setObjectName("SubTitle")
        layout.addWidget(self.subtitle)

        self.verdict_card = QFrame()
        self.verdict_card.setObjectName("verdictCard")
        self.verdict_card.setStyleSheet(f"#verdictCard {{ background-color: {CARD_BG}; border-radius: 16px; border: 2px solid #374151; }}")
        v_layout = QVBoxLayout(self.verdict_card)
        v_layout.setContentsMargins(20, 30, 20, 30)

        self.verdict_label = QLabel("STANDBY")
        self.verdict_label.setObjectName("VerdictLabel")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        v_layout.addWidget(self.verdict_label)

        self.margin_label = QLabel("")
        self.margin_label.setAlignment(Qt.AlignCenter)
        self.margin_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px;")
        v_layout.addWidget(self.margin_label)

        layout.addWidget(self.verdict_card)

        self.score_card = QFrame()
        self.score_card.setProperty("class", "Card")
        s_layout = QVBoxLayout(self.score_card)
        s_layout.setContentsMargins(20, 20, 20, 20)

        s_layout.addWidget(QLabel("COSINE SIMILARITY"))

        self.score_value = QLabel("--")
        self.score_value.setObjectName("ScoreValue")
        self.score_value.setAlignment(Qt.AlignCenter)
        self.score_value.setStyleSheet(f"color: {NEON_CYAN};")
        s_layout.addWidget(self.score_value)

        # container della barra: qui dentro disegniamo il fill + i marker
        # delle soglie (eer / far1 / far01), cosi' si vede sempre dove sta
        # la soglia attualmente in uso rispetto allo score ottenuto.
        self.score_bar_bg = QFrame()
        self.score_bar_bg.setFixedHeight(16)
        self.score_bar_bg.setStyleSheet(f"background-color: #374151; border-radius: 8px;")

        self.score_bar_fill = QFrame(self.score_bar_bg)
        self.score_bar_fill.setFixedHeight(16)
        self.score_bar_fill.setStyleSheet(f"background-color: {NEON_CYAN}; border-radius: 8px;")
        self.score_bar_fill.setFixedWidth(0)

        # marker fisso per lo zero (centro della barra, dato che il range è [-1, 1])
        self.zero_marker = QFrame(self.score_bar_bg)
        self.zero_marker.setFixedWidth(2)
        self.zero_marker.setStyleSheet(f"background-color: {TEXT_PRIMARY};")
        self.zero_marker.raise_()

        # marker verticali per ciascuna soglia, disegnati sopra la barra
        self.threshold_markers = {}
        for key in self.thresholds:
            marker = QFrame(self.score_bar_bg)
            marker.setFixedWidth(3)
            color = THRESHOLD_MARKER_COLORS.get(key, TEXT_PRIMARY)
            marker.setStyleSheet(f"background-color: {color};")
            marker.raise_()
            self.threshold_markers[key] = marker

        s_layout.addWidget(self.score_bar_bg)

        # legenda soglie sotto la barra, con evidenza sulla soglia attiva
        legend_row = QHBoxLayout()
        legend_row.setSpacing(10)
        self.threshold_legend_labels = {}
        for key in self.thresholds:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            self.threshold_legend_labels[key] = lbl
            legend_row.addWidget(lbl)
        s_layout.addLayout(legend_row)

        layout.addWidget(self.score_card)

        self.quality_card = QFrame()
        self.quality_card.setProperty("class", "Card")
        q_layout = QVBoxLayout(self.quality_card)
        q_layout.setContentsMargins(20, 20, 20, 20)

        q_layout.addWidget(QLabel("IMAGE QUALITY"))
        self.quality_text = QLabel("Focus: - | Ink: -")
        self.quality_text.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        q_layout.addWidget(self.quality_text)
        layout.addWidget(self.quality_card)

        layout.addStretch()

        self.bar_animation = QPropertyAnimation(self.score_bar_fill, b"minimumWidth")
        self.bar_animation.setDuration(800)
        self.bar_animation.setEasingCurve(QEasingCurve.OutQuart)

        self._refresh_threshold_legend()
        self._reposition_threshold_markers()

    # ---- soglie: posizionamento marker + legenda -----------------------------
    def _score_to_x(self, score):
        """Converte uno score cosine [-1, 1] in una posizione x dentro la barra."""
        bar_w = max(0, self.score_bar_bg.width() - 4)
        pct = (score + 1.0) / 2.0
        pct = max(0.0, min(1.0, pct))
        return int(pct * bar_w)

    def _reposition_threshold_markers(self):
        bar_h = self.score_bar_bg.height()
        zero_x = self._score_to_x(0.0)
        self.zero_marker.setGeometry(zero_x, 0, 2, bar_h)
        self.zero_marker.raise_()
        for key, marker in self.threshold_markers.items():
            thr_val = self.thresholds[key]
            x = self._score_to_x(thr_val)
            marker.setGeometry(x, 0, 3, bar_h)
            marker.raise_()

    def _refresh_threshold_legend(self):
        labels_map = {"eer": "EER", "far1": "FAR 1%", "far01": "FAR 0.1%"}
        for key, lbl in self.threshold_legend_labels.items():
            color = THRESHOLD_MARKER_COLORS.get(key, TEXT_PRIMARY)
            name = labels_map.get(key, key.upper())
            value = self.thresholds[key]
            is_active = (key == self.active_thr_key)
            if is_active:
                lbl.setText(f"● {name} {value:.3f}")
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 11px; font-weight: bold; "
                    f"border: 1px solid {color}; border-radius: 6px; padding: 2px 6px;")
            else:
                lbl.setText(f"{name} {value:.3f}")
                lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 2px 6px;")

    def set_active_threshold(self, key):
        self.active_thr_key = key
        self._refresh_threshold_legend()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_threshold_markers()

    def update_result(self, result, img_a, img_b, thresholds, thr_key):
        is_match = result["same_writer"]
        color = NEON_GREEN if is_match else NEON_RED
        text = "MATCH CONFIRMED" if is_match else "NO MATCH"

        self.verdict_label.setText(text)
        self.verdict_label.setStyleSheet(f"color: {color};")
        self.verdict_card.setStyleSheet(f"#verdictCard {{ background-color: {CARD_BG}; border-radius: 16px; border: 2px solid {color}; }}")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(color))
        shadow.setOffset(0, 0)
        self.verdict_card.setGraphicsEffect(shadow)

        self.margin_label.setText(f"Margin: {result['margin']:+.4f} | Threshold: {thr_key.upper()}")

        score = result["score"]
        self.score_value.setText(f"{score:.4f}")
        self.score_value.setStyleSheet(f"color: {color};")

        # riempimento classico da sinistra: 0% = -1, 100% = +1. Il marker
        # bianco fisso al centro (self.zero_marker) resta il riferimento
        # visivo per lo zero, cosi' la barra resta piena e leggibile anche
        # per punteggi alti, invece di crescere solo da un segmento centrale.
        target_width = self._score_to_x(score)

        self.score_bar_fill.move(0, 0)
        self.bar_animation.setStartValue(self.score_bar_fill.width())
        self.bar_animation.setEndValue(max(0, target_width))
        self.score_bar_fill.setStyleSheet(f"background-color: {color}; border-radius: 8px;")
        self.bar_animation.start()

        self._reposition_threshold_markers()

        f_a, i_a = quality_stats(img_a)
        f_b, i_b = quality_stats(img_b)
        self.quality_text.setText(f"A: Focus {f_a:.0f} | Ink {i_a:.1f}%\nB: Focus {f_b:.0f} | Ink {i_b:.1f}%")

        warns = quality_warnings(img_a) + quality_warnings(img_b)
        if warns:
            self.quality_text.setStyleSheet(f"color: #FBBF24; font-size: 13px;")
        else:
            self.quality_text.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")


# ==============================================================================
# PANNELLO PREVIEW PREPROCESSING (LIVE, SOTTO LA WEBCAM)
# ==============================================================================
class PreprocessPreview(QWidget):
    """Mostra le due ROI cosi' come escono dal preprocessing, cioe'
    esattamente quello che il modello vede (448x448, binarizzato).
    Thumbnail grandi per poterle controllare bene ad occhio nudo."""

    THUMB_SIZE = 340

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 12px;")
        self.setFixedHeight(self.THUMB_SIZE + 70)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)

        header = QLabel("MODEL INPUT PREVIEW (live)")
        header.setStyleSheet(f"color: {NEON_CYAN}; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        outer.addWidget(header)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.slot_a = self._build_slot("A")
        self.slot_b = self._build_slot("B")
        row.addWidget(self.slot_a["widget"])
        row.addWidget(self.slot_b["widget"])
        row.addStretch()
        outer.addLayout(row)

    def _build_slot(self, label_text):
        widget = QWidget()
        h = QHBoxLayout(widget)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)

        img_label = QLabel()
        img_label.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet(
            f"background-color: {DARK_BG}; border-radius: 8px; border: 1px solid #374151; color: {TEXT_SECONDARY};")
        img_label.setText("No ROI")
        h.addWidget(img_label)

        v = QVBoxLayout()
        v.setSpacing(4)
        tag = QLabel(f"SAMPLE {label_text}")
        tag.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        v.addWidget(tag)
        info = QLabel("Focus: -\nInk: -")
        info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        v.addWidget(info)
        v.addStretch()
        h.addLayout(v)

        return {"widget": widget, "img": img_label, "info": info}

    def update_slot(self, which, img_448):
        slot = self.slot_a if which == "A" else self.slot_b

        if img_448 is None:
            slot["img"].setPixmap(QPixmap())
            slot["img"].setText("No ROI")
            slot["info"].setText("Focus: -\nInk: -")
            return

        arr = np.ascontiguousarray(img_448)
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.THUMB_SIZE, self.THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        slot["img"].setText("")
        slot["img"].setPixmap(pixmap)

        focus, ink = quality_stats(arr)
        warns = quality_warnings(arr)
        color = "#FBBF24" if warns else TEXT_SECONDARY
        slot["info"].setStyleSheet(f"color: {color}; font-size: 12px;")
        slot["info"].setText(f"Focus: {focus:.0f}\nInk: {ink:.1f}%")


# ==============================================================================
# BARRA SOGLIE
# ==============================================================================
class ThresholdBar(QWidget):
    """Riga di pulsanti per scegliere quale soglia usare (EER / FAR 1% / FAR 0.1%)."""

    LABELS = {"eer": "EER", "far1": "FAR 1%", "far01": "FAR 0.1%"}

    def __init__(self, thresholds, active_key, on_change):
        super().__init__()
        self.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 12px;")
        self.on_change = on_change
        self.buttons = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        header = QLabel("SOGLIA DI DECISIONE")
        header.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(header)

        row = QHBoxLayout()
        row.setSpacing(8)
        for key in thresholds:
            btn = QPushButton(f"{self.LABELS.get(key, key.upper())}\n{thresholds[key]:.4f}")
            btn.setCheckable(True)
            btn.setChecked(key == active_key)
            btn.clicked.connect(lambda _checked, k=key: self._select(k))
            self.buttons[key] = btn
            row.addWidget(btn)
        layout.addLayout(row)

    def _select(self, key):
        for k, btn in self.buttons.items():
            btn.setChecked(k == key)
        self.on_change(key)

    def set_active(self, key):
        for k, btn in self.buttons.items():
            btn.setChecked(k == key)


# ==============================================================================
# MAIN WINDOW
# ==============================================================================
class HandVerifyApp(QMainWindow):
    def __init__(self, checkpoint, arch, camera_id, threshold_key, initial_focus):
        super().__init__()
        self.setWindowTitle("HandVerify // Biometric Scanner")
        self.resize(1500, 1000)
        self.setStyleSheet(GLOBAL_QSS)

        print("Caricamento modello AI...")
        self.verifier = HandwritingVerifier(checkpoint, arch=arch)
        print(f"Modello pronto: {self.verifier.describe()}")

        self.thr_key = threshold_key
        self.available_cameras = self._detect_cameras()
        print(f"Camere disponibili: {self.available_cameras}")

        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        self.video_canvas = VideoCanvas()
        self.preprocess_preview = PreprocessPreview()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        # pulsanti per scegliere quale ROI si sta disegnando
        roi_select_row = QHBoxLayout()
        self.roi_buttons = {}
        for name in ROI_NAMES:
            btn = QPushButton(f"Disegna ROI {name}")
            btn.setCheckable(True)
            btn.setChecked(name == self.video_canvas.active_roi)
            btn.setStyleSheet(
                f"background-color: {ROI_COLORS[name]}; color: {DARK_BG};" if name == self.video_canvas.active_roi
                else "")
            btn.clicked.connect(lambda _checked, n=name: self._select_active_roi(n))
            self.roi_buttons[name] = btn
            roi_select_row.addWidget(btn)
        left_layout.addLayout(roi_select_row)

        left_layout.addWidget(self.video_canvas, stretch=1)
        left_layout.addWidget(self.preprocess_preview)
        main_layout.addWidget(left_panel, stretch=1)

        self._last_preview_update = 0.0
        self._preview_interval = 0.1  # ~10 fps

        self.dashboard = ResultDashboard(self.verifier.thresholds, self.thr_key)

        focus_widget = QWidget()
        focus_widget.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 12px; padding: 12px;")
        focus_layout = QVBoxLayout(focus_widget)
        focus_layout.setContentsMargins(16, 12, 16, 12)

        focus_header = QLabel("MANUAL FOCUS CONTROL")
        focus_header.setStyleSheet(f"color: {NEON_CYAN}; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        focus_layout.addWidget(focus_header)

        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setMinimum(0)
        self.focus_slider.setMaximum(255)
        self.focus_slider.setValue(initial_focus)
        self.focus_slider.valueChanged.connect(self.on_focus_slider_changed)
        focus_layout.addWidget(self.focus_slider)

        self.focus_label = QLabel(f"Focus: {initial_focus}")
        self.focus_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        self.focus_label.setAlignment(Qt.AlignCenter)
        focus_layout.addWidget(self.focus_label)

        self.gray_threshold = DEFAULT_GRAY_THRESHOLD

        gray_widget = QWidget()
        gray_widget.setStyleSheet(f"background-color: {PANEL_BG}; border-radius: 12px; padding: 12px;")
        gray_layout = QVBoxLayout(gray_widget)
        gray_layout.setContentsMargins(16, 12, 16, 12)

        gray_header = QLabel("SOGLIA BINARIZZAZIONE (GRIGIO)")
        gray_header.setStyleSheet(f"color: {NEON_CYAN}; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        gray_layout.addWidget(gray_header)

        self.gray_slider = QSlider(Qt.Horizontal)
        self.gray_slider.setMinimum(0)
        self.gray_slider.setMaximum(255)
        self.gray_slider.setValue(self.gray_threshold)
        self.gray_slider.valueChanged.connect(self.on_gray_threshold_changed)
        gray_layout.addWidget(self.gray_slider)

        self.gray_label = QLabel(f"Soglia: {self.gray_threshold}")
        self.gray_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        self.gray_label.setAlignment(Qt.AlignCenter)
        gray_layout.addWidget(self.gray_label)

        self.threshold_bar = ThresholdBar(self.verifier.thresholds, self.thr_key, self._on_threshold_changed)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self.dashboard)
        right_layout.addWidget(focus_widget)
        right_layout.addWidget(gray_widget)
        right_layout.addWidget(self.threshold_bar)

        main_layout.addWidget(right_panel)

        # Scorciatoia per SPAZIO indipendente dal focus da tastiera: senza
        # questo, se nessun widget con focus policy ha mai ricevuto il
        # focus (es. non si e' ancora cliccato su uno slider/pulsante),
        # keyPressEvent sulla finestra non riceve gli eventi e SPAZIO
        # sembra "non funzionare" finche' non si clicca altrove prima.
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.ApplicationShortcut)
        self._space_shortcut.activated.connect(self.capture_and_verify)

        if camera_id not in self.available_cameras and self.available_cameras:
            print(f"[WARNING] Camera {camera_id} non disponibile, uso la prima disponibile: {self.available_cameras[0]}")
            camera_id = self.available_cameras[0]

        self.camera_worker = CameraWorker(camera_id, 1280, 720, initial_focus)
        self.camera_worker.frame_ready.connect(self.process_frame)
        self.camera_worker.focus_changed.connect(self.on_focus_changed)
        self.camera_worker.error_occurred.connect(self.show_camera_error)
        self.camera_worker.camera_opened.connect(self.on_camera_opened)
        self.camera_worker.start()

    def _detect_cameras(self):
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def _select_active_roi(self, name):
        self.video_canvas.set_active_roi(name)
        for n, btn in self.roi_buttons.items():
            btn.setChecked(n == name)
            btn.setStyleSheet(f"background-color: {ROI_COLORS[n]}; color: {DARK_BG};" if n == name else "")

    def _on_threshold_changed(self, key):
        self.thr_key = key
        self.dashboard.set_active_threshold(key)
        print(f"Soglia attiva: {self.thr_key} = {self.verifier.thresholds[self.thr_key]:.4f}")

    def process_frame(self, frame):
        self.video_canvas.update_frame(frame)
        self._update_preprocess_preview()

    def _update_preprocess_preview(self):
        now = time.time()
        if now - self._last_preview_update < self._preview_interval:
            return
        self._last_preview_update = now

        for name in ROI_NAMES:
            roi = self.video_canvas.get_roi_crop(name)
            img_448 = self.verifier.prepare(roi, self.gray_threshold) if roi is not None else None
            self.preprocess_preview.update_slot(name, img_448)

    def on_gray_threshold_changed(self, value):
        self.gray_threshold = value
        self.gray_label.setText(f"Soglia: {value}")
        self.video_canvas.set_gray_threshold_display(value)

    def on_focus_slider_changed(self, value):
        self.camera_worker.set_focus_absolute(value)
        self.focus_label.setText(f"Focus: {value}")

    def on_focus_changed(self, value):
        self.video_canvas.set_focus_display(value)
        if self.focus_slider.value() != value:
            self.focus_slider.blockSignals(True)
            self.focus_slider.setValue(value)
            self.focus_slider.blockSignals(False)
        self.focus_label.setText(f"Focus: {value}")

    def on_camera_opened(self, camera_id):
        self.video_canvas.set_camera_id(camera_id)

    def show_camera_error(self, error_msg):
        self.video_canvas.show_error(error_msg)

    def keyPressEvent(self, event):
        # Ignora gli auto-repeat generati dal sistema quando un tasto resta
        # premuto: senza questo controllo, tenendo premuto SPAZIO si otterrebbero
        # decine di verifiche al secondo invece di una singola per pressione.
        if event.isAutoRepeat():
            return

        key = event.key()

        if key == Qt.Key_R:
            self.reset_view()
        elif key == Qt.Key_T:
            keys = list(self.verifier.thresholds)
            idx = keys.index(self.thr_key)
            new_key = keys[(idx + 1) % len(keys)]
            self.threshold_bar.set_active(new_key)
            self._on_threshold_changed(new_key)
        elif key == Qt.Key_F:
            new_val = self.focus_slider.value() + 4
            self.focus_slider.setValue(min(255, new_val))
        elif key == Qt.Key_G:
            new_val = self.focus_slider.value() - 4
            self.focus_slider.setValue(max(0, new_val))
        elif key == Qt.Key_D:
            new_val = self.focus_slider.value() + 20
            self.focus_slider.setValue(min(255, new_val))
        elif key == Qt.Key_H:
            new_val = self.focus_slider.value() - 20
            self.focus_slider.setValue(max(0, new_val))
        elif key == Qt.Key_C:
            self._switch_to_next_camera()
        elif key == Qt.Key_A:
            self._select_active_roi("A")
        elif key == Qt.Key_B:
            self._select_active_roi("B")
        elif key == Qt.Key_1:
            self.focus_slider.setValue(0)
        elif key == Qt.Key_2:
            self.focus_slider.setValue(64)
        elif key == Qt.Key_3:
            self.focus_slider.setValue(128)
        elif key == Qt.Key_4:
            self.focus_slider.setValue(192)
        elif key == Qt.Key_5:
            self.focus_slider.setValue(255)
        elif key == Qt.Key_Escape or key == Qt.Key_Q:
            self.close()

    def _switch_to_next_camera(self):
        if not self.available_cameras:
            print("Nessuna camera disponibile")
            return

        current = self.camera_worker.camera_id
        idx = self.available_cameras.index(current) if current in self.available_cameras else -1
        next_idx = (idx + 1) % len(self.available_cameras)
        next_camera = self.available_cameras[next_idx]

        print(f"Cambio camera: {current} -> {next_camera}")
        self.camera_worker.switch_camera(next_camera)

    def capture_and_verify(self):
        roi_a = self.video_canvas.get_roi_crop("A")
        roi_b = self.video_canvas.get_roi_crop("B")

        if roi_a is None or roi_b is None:
            self.dashboard.subtitle.setText("Disegna prima entrambe le ROI (A e B).")
            return

        img_a = self.verifier.prepare(roi_a, self.gray_threshold)
        img_b = self.verifier.prepare(roi_b, self.gray_threshold)

        if img_a is None or img_b is None:
            self.dashboard.subtitle.setText("Error: ROI non valida in uno dei due campioni.")
            return

        t0 = time.time()
        result = self.verifier.verify(img_a, img_b, self.thr_key)
        print(f"  cos={result['score']:+.4f} | {self.thr_key}={result['threshold']:.4f} | {(time.time()-t0)*1000:.0f}ms")

        self.dashboard.subtitle.setText(f"Analysis completed in {(time.time()-t0)*1000:.0f} ms")
        self.dashboard.update_result(result, img_a, img_b, self.verifier.thresholds, self.thr_key)

    def reset_view(self):
        self.dashboard.subtitle.setText("Waiting for capture...")
        self.dashboard.verdict_label.setText("STANDBY")
        self.dashboard.verdict_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.dashboard.verdict_card.setStyleSheet(f"#verdictCard {{ background-color: {CARD_BG}; border-radius: 16px; border: 2px solid #374151; }}")
        self.dashboard.verdict_card.setGraphicsEffect(None)
        self.dashboard.score_value.setText("--")
        zero_x = self.dashboard._score_to_x(0.0)
        self.dashboard.score_bar_fill.move(zero_x, 0)
        self.dashboard.score_bar_fill.setFixedWidth(0)
        self.dashboard.quality_text.setText("Focus: - | Ink: -")

    def closeEvent(self, event):
        self.camera_worker.stop()
        event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", choices=list(DEFAULT_THRESHOLDS), default="eer")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--focus", type=int, default=30, help="Valore focus iniziale (0-255)")
    args = parser.parse_args()

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = HandVerifyApp(args.checkpoint, args.arch, args.camera, args.threshold, args.focus)
    window.show()

    sys.exit(app.exec())