import os
import cv2
import time
import queue
import threading
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image, ImageTk
import tensorflow as tf
from ultralytics import YOLO

# =====================================================
# SUPPRESS TF LOGS
# =====================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =====================================================
# KONSTANTA
# =====================================================
WIDTH, HEIGHT = 640, 480
MIN_CONF      = 0.15
MIN_AREA      = 150
MIN_ASPECT    = 0.2
MAX_ASPECT    = 4.0
ALLOWED_CLS   = {14}

SIZE_THRESHOLDS = {
    "Kecil":  (0,     1500),
    "Sedang": (1500,  4000),
    "Besar":  (4000,  float("inf")),
}

BEHAVIOR_COLOR = {
    "Sangat Aktif": (0,   255,   0),
    "Aktif":        (0,   200, 255),
    "Diam":         (255, 200,   0),
    "Tidak Aktif":  (0,     0, 255),
}

SIZE_COLOR = {
    "Kecil":  (255, 200,   0),
    "Sedang": (0,   200, 255),
    "Besar":  (0,   100, 255),
}

FESES_MAP = {
    "cocci"  : ("Coccidiosis",       "#e74c3c"),
    "healthy" : ("Feses Sehat",       "#2ecc71"),
    "ncd"    : ("Newcastle Disease", "#e67e22"),
    "salmo"  : ("Salmonella",        "#9b59b6"),
}

FESES_COLOR_BGR = {
    "cocci"  : (0,   0, 230),
    "healthy" : (0, 200,  50),
    "ncd"    : (0, 150, 230),
    "salmo"  : (180,  50, 150),
}

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLO model...")
yolo_model = YOLO("yolo11s.pt")

print("Loading feses model...")
feses_model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "models", "model_ayam.h5")
)
with open(os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

print("Semua model loaded!")

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def is_valid_chicken(cls_id, conf, w, h):
    if cls_id not in ALLOWED_CLS:
        return False
    if conf < MIN_CONF:
        return False
    if w * h < MIN_AREA:
        return False
    aspect = w / (h + 1e-6)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    return True


def classify_size(area):
    for label, (lo, hi) in SIZE_THRESHOLDS.items():
        if lo <= area < hi:
            return label
    return "Besar"


def estimate_weight(area):
    weight = 0.25 * area + 200
    return int(np.clip(weight, 300, 3500))


def classify_behavior(movement, inactive):
    if inactive > 40:
        return "Tidak Aktif"
    elif movement < 2:
        return "Diam"
    elif movement < 15:
        return "Aktif"
    else:
        return "Sangat Aktif"


def segmentasi_feses(img_bgr):
    """Segmentasi area feses menggunakan HSV thresholding."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Range 1: coklat/gelap (feses umum)
    mask1 = cv2.inRange(hsv,
        np.array([0,20,20]),
        np.array([40,255,180]))

    # Range 2: putih/krem (feses sehat)
    mask2 = cv2.inRange(hsv,
                         np.array([0,  0,  160]),
                         np.array([30, 40, 255]))

    # Range 3: hijau (feses sakit)
    mask3 = cv2.inRange(hsv,
                         np.array([35, 40, 40]),
                         np.array([85, 255, 180]))

    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def predict_feses(region_bgr):
    """Prediksi jenis feses dari crop region."""
    img = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (180, 180))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    prob     = feses_model.predict(img, verbose=0)[0]
    idx      = np.argmax(prob)
    label    = le.inverse_transform([idx])[0]
    conf     = prob[idx] * 100
    return label, conf, prob


def update_heatmap(heatmap, cx, cy, w, h):
    radius    = max(int((w + h) / 4), 20)
    size      = radius * 2 + 1
    kernel    = cv2.getGaussianKernel(size, radius / 2)
    kernel_2d = (kernel @ kernel.T)
    kernel_2d = kernel_2d / kernel_2d.max()
    x1, y1   = cx - radius, cy - radius
    x2, y2   = cx + radius + 1, cy + radius + 1
    kx1 = max(0, -x1); ky1 = max(0, -y1)
    kx2 = size - max(0, x2 - WIDTH)
    ky2 = size - max(0, y2 - HEIGHT)
    hx1 = max(0, x1); hy1 = max(0, y1)
    hx2 = min(WIDTH,  x2); hy2 = min(HEIGHT, y2)
    if hx2 > hx1 and hy2 > hy1:
        heatmap[hy1:hy2, hx1:hx2] += kernel_2d[ky1:ky2, kx1:kx2]


# =====================================================
# SKEMA EVENT TERPUSAT
# =====================================================
# Tiga sumber deteksi (perilaku, ukuran, feses) menulis ke struktur yang
# SAMA ini, supaya panel "Riwayat Deteksi" tidak perlu tahu detail tiap
# model -- dia cuma tahu cara menampilkan DetectionEvent.

STATUS_COLOR_HEX = {
    "normal":    "#a6e3a1",
    "perhatian": "#f9e2af",
    "kritis":    "#f38ba8",
}

# Pemetaan label asli model -> status kesehatan, dipakai biar panel
# riwayat & ringkasan tahu mana yang perlu disorot merah/kuning.
BEHAVIOR_STATUS = {
    "Sangat Aktif": "normal",
    "Aktif":        "normal",
    "Diam":         "perhatian",
    "Tidak Aktif":  "kritis",
}

SIZE_STATUS = {
    "Kecil":  "perhatian",
    "Sedang": "normal",
    "Besar":  "normal",
}

FESES_STATUS = {
    "healthy": "normal",
    "cocci":   "kritis",
    "ncd":     "kritis",
    "salmo":   "kritis",
}


@dataclass
class DetectionEvent:
    modul: str                 # "perilaku" | "ukuran" | "feses"
    label: str                  # contoh: "Tidak Aktif", "Kecil", "Coccidiosis"
    confidence: float           # 0.0 - 1.0
    status: str                  # "normal" | "perhatian" | "kritis"
    ekor_id: int | None = None     # track id dari ByteTrack, kalau ada
    waktu: datetime = field(default_factory=datetime.now)

    def warna(self) -> str:
        return STATUS_COLOR_HEX.get(self.status, "#cdd6f4")


class DetectionBus:
    """Queue thread-safe antara loop video (thread terpisah) dan GUI (main thread).

    _loop_video jalan di background thread, jadi tidak boleh langsung
    menyentuh widget Tkinter. Dia publish ke sini, lalu GUI poll lewat
    root.after() di main thread.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[DetectionEvent]" = queue.Queue()

    def publish(self, event: DetectionEvent) -> None:
        self._q.put(event)

    def drain(self) -> list[DetectionEvent]:
        events = []
        while True:
            try:
                events.append(self._q.get_nowait())
            except queue.Empty:
                break
        return events


# =====================================================
# APLIKASI TKINTER
# =====================================================

class AplikasiAyam:
    def __init__(self, root):
        self.root       = root
        self.root.title(" Sistem Monitoring Ayam")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1280x780")
        self.root.resizable(True, True)

        # State
        self.cap             = None
        self.is_running      = False
        self.video_path      = None
        self.frame_count     = 0
        self.track_history   = defaultdict(list)
        self.inactive_frames = defaultdict(int)
        self.heatmap         = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.last_annotated  = None
        self.feses_done      = False 

        # Event terpusat: tempat ketiga model menulis hasil deteksinya
        self.bus              = DetectionBus()
        self.history: list    = []
        self.last_status      = {}   # (modul, ekor_id) -> status terakhir, biar tidak spam event tiap frame

        # Stats
        self.stat_total      = tk.StringVar(value="0")
        self.stat_aktif      = tk.StringVar(value="0")
        self.stat_diam       = tk.StringVar(value="0")
        self.stat_tidak_aktif= tk.StringVar(value="0")
        self.stat_kecil      = tk.StringVar(value="0")
        self.stat_sedang     = tk.StringVar(value="0")
        self.stat_besar      = tk.StringVar(value="0")
        self.stat_avg_berat  = tk.StringVar(value="0g")
        self.stat_status     = tk.StringVar(value="-")
        self.stat_feses      = tk.StringVar(value="-")
        self.stat_fps        = tk.StringVar(value="0")

        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self.root, bg="#181825", pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🐔  Sistem Monitoring Ayam",
                  font=("Segoe UI", 16, "bold"),
                  bg="#181825", fg="#cdd6f4").pack(side="left", padx=16)
        tk.Label(hdr, text="Deteksi: Perilaku | Ukuran | Feses",
                  font=("Segoe UI", 10),
                  bg="#181825", fg="#6c7086").pack(side="left")

        # ── Body ──
        body = tk.Frame(self.root, bg="#1e1e2e")
        body.pack(fill="both", expand=True, padx=10, pady=6)

        # Kolom kiri: video
        left = tk.Frame(body, bg="#1e1e2e")
        left.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(left, bg="#11111b",
                                     relief="flat", bd=0)
        self.video_label.pack(fill="both", expand=True, pady=(0, 4))

        # Tombol kontrol
        ctrl = tk.Frame(left, bg="#1e1e2e")
        ctrl.pack(fill="x")

        btn_cfg = {"font": ("Segoe UI", 10, "bold"), "bd": 0,
                   "padx": 14, "pady": 7, "cursor": "hand2"}

        self.btn_open = tk.Button(ctrl, text="Buka Video",
                                   bg="#89b4fa", fg="#1e1e2e",
                                   command=self.buka_video, **btn_cfg)
        self.btn_open.pack(side="left", padx=(0, 6))

        self.btn_play = tk.Button(ctrl, text="Mulai",
                                   bg="#a6e3a1", fg="#1e1e2e",
                                   command=self.toggle_play,
                                   state="disabled", **btn_cfg)
        self.btn_play.pack(side="left", padx=(0, 6))

        self.btn_stop = tk.Button(ctrl, text="Stop",
                                   bg="#f38ba8", fg="#1e1e2e",
                                   command=self.stop_video,
                                   state="disabled", **btn_cfg)
        self.btn_stop.pack(side="left")

        fps_lbl = tk.Label(ctrl, textvariable=self.stat_fps,
                            font=("Segoe UI", 10),
                            bg="#1e1e2e", fg="#6c7086")
        fps_lbl.pack(side="right", padx=8)
        tk.Label(ctrl, text="FPS:", font=("Segoe UI", 10),
                  bg="#1e1e2e", fg="#6c7086").pack(side="right")

        # Kolom kanan: panel info
        right = tk.Frame(body, bg="#181825", width=290,
                          relief="flat", bd=0)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        self._panel_perilaku(right)
        self._panel_ukuran(right)
        self._panel_feses(right)
        self._panel_status(right)
        self._panel_riwayat(right)

        self._poll_bus()

    def _section(self, parent, title, color):
        frm = tk.Frame(parent, bg="#181825")
        frm.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(frm, text=title, font=("Segoe UI", 11, "bold"),
                  bg="#181825", fg=color).pack(anchor="w")
        sep = tk.Frame(frm, bg=color, height=2)
        sep.pack(fill="x", pady=(2, 6))
        return frm

    def _stat_row(self, parent, label, var, color="#cdd6f4"):
        row = tk.Frame(parent, bg="#181825")
        row.pack(fill="x", padx=4, pady=1)
        tk.Label(row, text=label, font=("Segoe UI", 9),
                  bg="#181825", fg="#6c7086", width=18,
                  anchor="w").pack(side="left")
        tk.Label(row, textvariable=var, font=("Segoe UI", 9, "bold"),
                  bg="#181825", fg=color).pack(side="left")

    def _panel_perilaku(self, parent):
        f = self._section(parent, "Perilaku", "#a6e3a1")
        self._stat_row(f, "Total Ayam",    self.stat_total,       "#cdd6f4")
        self._stat_row(f, "Aktif",         self.stat_aktif,       "#a6e3a1")
        self._stat_row(f, "Diam",          self.stat_diam,        "#f9e2af")
        self._stat_row(f, "Tidak Aktif",   self.stat_tidak_aktif, "#f38ba8")

    def _panel_ukuran(self, parent):
        f = self._section(parent, "Ukuran Tubuh", "#89b4fa")
        self._stat_row(f, "Kecil",         self.stat_kecil,  "#f9e2af")
        self._stat_row(f, "Sedang",        self.stat_sedang, "#89dceb")
        self._stat_row(f, "Besar",         self.stat_besar,  "#fab387")
        self._stat_row(f, "Rata-rata Berat", self.stat_avg_berat, "#cdd6f4")

    def _panel_feses(self, parent):
        f = self._section(parent, "Deteksi Feses", "#cba6f7")
        self._stat_row(f, "Kondisi Feses", self.stat_feses, "#cba6f7")

    def _panel_status(self, parent):
        f = self._section(parent, "Status Kandang", "#f38ba8")
        lbl = tk.Label(f, textvariable=self.stat_status,
                        font=("Segoe UI", 13, "bold"),
                        bg="#181825", fg="#f38ba8")
        lbl.pack(pady=8)

    def _panel_riwayat(self, parent):
        f = self._section(parent, "Riwayat Deteksi", "#89b4fa")
        cols = ("waktu", "hasil")
        self.tree_riwayat = ttk.Treeview(f, columns=cols, show="headings", height=8)
        self.tree_riwayat.heading("waktu", text="Waktu")
        self.tree_riwayat.heading("hasil", text="Hasil")
        self.tree_riwayat.column("waktu", width=60, anchor="w")
        self.tree_riwayat.column("hasil", width=200, anchor="w")
        self.tree_riwayat.pack(fill="both", expand=True, pady=(0, 8))

        style = ttk.Style()
        style.configure("Treeview", background="#181825", fieldbackground="#181825",
                        foreground="#cdd6f4", font=("Segoe UI", 8))
        style.configure("Treeview.Heading", background="#1e1e2e", foreground="#6c7086")

        self.tree_riwayat.tag_configure("normal", foreground="#a6e3a1")
        self.tree_riwayat.tag_configure("perhatian", foreground="#f9e2af")
        self.tree_riwayat.tag_configure("kritis", foreground="#f38ba8")

    def _poll_bus(self):
        """Dipanggil berkala lewat root.after() di main thread -- satu-satunya
        tempat yang aman buat update widget Tkinter dari hasil di DetectionBus."""
        events = self.bus.drain()
        for ev in events:
            self.history.insert(0, ev)
            teks = ev.label if ev.ekor_id is None else f"ID:{ev.ekor_id} {ev.label}"
            self.tree_riwayat.insert("", 0, values=(ev.waktu.strftime("%H:%M:%S"), teks),
                                      tags=(ev.status,))
            children = self.tree_riwayat.get_children()
            if len(children) > 50:
                self.tree_riwayat.delete(children[-1])
        self.history = self.history[:200]
        self.root.after(500, self._poll_bus)

    # ── VIDEO CONTROL ────────────────────────────────────────────────────────

    def buka_video(self):
        path = filedialog.askopenfilename(
            title="Pilih Video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")]
        )
        if not path:
            return
        self.video_path = path
        self.btn_play.config(state="normal")
        self.stat_status.set("Video siap")
        # Preview frame pertama
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._show_frame(cv2.resize(frame, (WIDTH, HEIGHT)))

    def toggle_play(self):
        if self.is_running:
            self.is_running = False
            self.btn_play.config(text="▶  Lanjut")
        else:
            self.is_running = True
            self.btn_play.config(text="⏸  Pause")
            self.btn_stop.config(state="normal")
            if self.cap is None or not self.cap.isOpened():
                self._reset_state()
                self.cap = cv2.VideoCapture(self.video_path)
            threading.Thread(target=self._loop_video,
                              daemon=True).start()

    def stop_video(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_play.config(text="▶  Mulai", state="normal")
        self.btn_stop.config(state="disabled")
        self.stat_status.set("Dihentikan")

    def _reset_state(self):
        self.frame_count     = 0
        self.track_history   = defaultdict(list)
        self.inactive_frames = defaultdict(int)
        self.heatmap          = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        self.last_annotated  = None
        self.feses_done      = False 

    # LOOP VIDEO 

    def _loop_video(self):
        final_behavior = {}
        final_size     = {}
        final_weight   = {}
        feses_results  = []
        feses_interval = 0   # hitung frame untuk deteksi feses berkala

        fps_video = self.cap.get(cv2.CAP_PROP_FPS) or 25

        while self.is_running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.stat_status.set("Video selesai"))
                break

            self.frame_count += 1
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # ── Skip frame (tampilkan frame terakhir) ──
            if self.frame_count % 3 != 0:
                if self.last_annotated is not None:
                    self._show_frame(self.last_annotated)
                else:
                    self._show_frame(frame)
                delay = max(1, int(1000 / fps_video) - int((time.time()-t0)*1000))
                time.sleep(delay / 1000)
                continue

            annotated = frame.copy()

            # ── YOLO TRACKING ──
            results = yolo_model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                verbose=False, imgsz=640, conf=MIN_CONF,
                classes=list(ALLOWED_CLS),
            )

            if results[0].boxes.id is not None:
                boxes     = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs     = results[0].boxes.conf.cpu().numpy()
                cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, tid, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                    x, y, w, h = box
                    if not is_valid_chicken(cls_id, conf, w, h):
                        continue

                    center = (int(x), int(y))
                    track  = self.track_history[tid]
                    track.append(center)
                    if len(track) > 30:
                        track.pop(0)

                    movement = 0
                    if len(track) >= 2:
                        px, py   = track[-2]
                        movement = np.sqrt((center[0]-px)**2 + (center[1]-py)**2)

                    if movement < 2:
                        self.inactive_frames[tid] += 1
                    else:
                        self.inactive_frames[tid] = 0

                    behavior = classify_behavior(movement, self.inactive_frames[tid])
                    final_behavior[tid] = behavior

                    area   = w * h
                    size   = classify_size(area)
                    weight = estimate_weight(area)
                    final_size[tid]   = size
                    final_weight[tid] = weight

                    # ── Publish event ke DetectionBus, cuma kalau statusnya berubah ──
                    beh_status = BEHAVIOR_STATUS.get(behavior, "normal")
                    if self.last_status.get(("perilaku", tid)) != beh_status:
                        self.last_status[("perilaku", tid)] = beh_status
                        self.bus.publish(DetectionEvent(
                            modul="perilaku", label=behavior,
                            confidence=float(conf), status=beh_status, ekor_id=int(tid),
                        ))

                    size_status = SIZE_STATUS.get(size, "normal")
                    if self.last_status.get(("ukuran", tid)) != size_status:
                        self.last_status[("ukuran", tid)] = size_status
                        self.bus.publish(DetectionEvent(
                            modul="ukuran", label=f"{size} (~{weight}g)",
                            confidence=float(conf), status=size_status, ekor_id=int(tid),
                        ))

                    # ── Gambar bbox ──
                    beh_color  = BEHAVIOR_COLOR.get(behavior, (255,255,255))
                    size_color = SIZE_COLOR.get(size, (255,255,255))

                    x1 = int(x - w/2); y1 = int(y - h/2)
                    x2 = int(x + w/2); y2 = int(y + h/2)

                    cv2.rectangle(annotated, (x1,y1), (x2,y2), beh_color, 2)

                    lbl1 = f"ID:{tid} {size} ~{weight}g"
                    lbl2 = behavior
                    ly   = max(y1-18, 12)
                    cv2.putText(annotated, lbl1, (x1, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                size_color, 1, cv2.LINE_AA)
                    cv2.putText(annotated, lbl2, (x1, ly+13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                beh_color, 1, cv2.LINE_AA)

                    if len(track) >= 2:
                        pts = np.array(track, dtype=np.int32)
                        cv2.polylines(annotated, [pts], False, (255,0,0), 1)

                    update_heatmap(self.heatmap, int(x), int(y), int(w), int(h))

            # ── DETEKSI FESES (HANYA FRAME AWAL, 1 KALI SAJA) ──
            if self.frame_count % 30 == 0:
                mask_feses = segmentasi_feses(frame)

                roi_start = int(HEIGHT * 0.75)
                mask_feses[:roi_start, :] = 0

                contours, _ = cv2.findContours(mask_feses,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                feses_results = []
                for cnt in contours:
                    area_cnt = cv2.contourArea(cnt)
                    if area_cnt < 500 or area_cnt > 10000:
                        continue
                    fx, fy, fw, fh = cv2.boundingRect(cnt)
                    extent = area_cnt / (fw * fh + 1e-6)

                    if extent < 0.4:
                        continue

                    rasio = fw / (fh + 1e-6)
                    if rasio > 5 or rasio < 0.2:
                        continue
                    fx2 = min(fx + fw, WIDTH)
                    fy2 = min(fy + fh, HEIGHT)
                    crop = frame[fy:fy2, fx:fx2]
                    if crop.size == 0:
                        continue
                    flabel, fconf, _ = predict_feses(crop)
                    feses_results.append((fx, fy, fx2, fy2, flabel, fconf))

                    nama_f, _ = FESES_MAP.get(flabel, (flabel, "#fff"))
                    self.bus.publish(DetectionEvent(
                        modul="feses", label=nama_f,
                        confidence=float(fconf) / 100,
                        status=FESES_STATUS.get(flabel, "perhatian"),
                    ))

                self.feses_done = True  

            # Gambar hasil feses di frame
            for (fx, fy, fx2, fy2, flabel, fconf) in feses_results:
                color_f = FESES_COLOR_BGR.get(flabel, (200,200,200))
                cv2.rectangle(annotated, (fx, fy), (fx2, fy2), color_f, 2)
                nama_f, _ = FESES_MAP.get(flabel, (flabel, "#fff"))
                cv2.putText(annotated,
                            f"{nama_f} {fconf:.0f}%",
                            (fx, max(fy-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            color_f, 1, cv2.LINE_AA)

            # ── LEGEND PERILAKU ──
            ly = HEIGHT - 95
            for lbl, col in BEHAVIOR_COLOR.items():
                cv2.rectangle(annotated, (8, ly), (20, ly+12), col, -1)
                cv2.putText(annotated, lbl, (24, ly+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
                ly += 18

            # ── FPS ──
            fps_real = 1 / (time.time() - t0 + 1e-6)
            cv2.putText(annotated, f"FPS:{fps_real:.1f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2, cv2.LINE_AA)

            self.last_annotated = annotated
            self._show_frame(annotated)

            # ── UPDATE STATS ──
            self._update_stats(final_behavior, final_size,
                                final_weight, feses_results)

            delay = max(1, int(1000/fps_video) - int((time.time()-t0)*1000))
            time.sleep(delay / 1000)

        if self.is_running:
            self.is_running = False
            self.root.after(0, lambda: self.btn_play.config(text="▶  Mulai"))

    # ── STATS ────────────────────────────────────────────────────────────────

    def _update_stats(self, behavior, size, weight, feses_results):
        total   = len(behavior)
        aktif   = sum(1 for b in behavior.values() if b in ("Aktif","Sangat Aktif"))
        diam    = sum(1 for b in behavior.values() if b == "Diam")
        t_aktif = sum(1 for b in behavior.values() if b == "Tidak Aktif")

        kecil   = sum(1 for s in size.values() if s == "Kecil")
        sedang  = sum(1 for s in size.values() if s == "Sedang")
        besar   = sum(1 for s in size.values() if s == "Besar")
        avg_w   = int(np.mean(list(weight.values()))) if weight else 0

        pct_tidak = (t_aktif / total * 100) if total > 0 else 0
        if pct_tidak > 30:
            status = "Potensi Sakit Tinggi"
        elif pct_tidak > 15:
            status = "Waspada"
        else:
            status = "Normal"

        # Feses dominan
        if feses_results:
            from collections import Counter
            cnt = Counter(r[4] for r in feses_results)
            dom = cnt.most_common(1)[0][0]
            nama_f, _ = FESES_MAP.get(dom, (dom, "#fff"))
            feses_txt = nama_f
        else:
            feses_txt = "Tidak terdeteksi"

        def _set():
            self.stat_total.set(str(total))
            self.stat_aktif.set(str(aktif))
            self.stat_diam.set(str(diam))
            self.stat_tidak_aktif.set(str(t_aktif))
            self.stat_kecil.set(str(kecil))
            self.stat_sedang.set(str(sedang))
            self.stat_besar.set(str(besar))
            self.stat_avg_berat.set(f"~{avg_w}g")
            self.stat_status.set(status)
            self.stat_feses.set(feses_txt)
            self.stat_fps.set(f"{1/(time.time()-time.time()+1e-6):.0f}")
        self.root.after(0, _set)

    # ── SHOW FRAME ───────────────────────────────────────────────────────────

    def _show_frame(self, frame_bgr):
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)

        # Fit ke label size
        lw = self.video_label.winfo_width()
        lh = self.video_label.winfo_height()
        if lw > 10 and lh > 10:
            img = img.resize((lw, lh), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.root.after(0, self._set_image, imgtk)

    def _set_image(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    root = tk.Tk()
    app  = AplikasiAyam(root)
    root.mainloop()