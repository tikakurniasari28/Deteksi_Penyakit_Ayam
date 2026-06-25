import os
import time
import numpy as np
import csv
from ultralytics import YOLO
from collections import defaultdict
import cv2

# =====================================================
# LOAD MODEL
# =====================================================

model = YOLO("yolo11s.pt")

# =====================================================
# FOLDER
# =====================================================

VIDEO_FOLDER  = "ayam"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

csv_path   = os.path.join(OUTPUT_FOLDER, "hasil_perilaku.csv")
csv_file   = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Video", "ID Ayam", "Behavior", "Status Akhir"])

# =====================================================
# FILTER AYAM
# =====================================================

ALLOWED_CLASSES = {14}
MIN_ASPECT = 0.2
MAX_ASPECT = 4.0
MIN_AREA   = 150
MIN_CONF   = 0.15

def is_valid_chicken(cls_id, conf, w, h):
    if cls_id not in ALLOWED_CLASSES:
        return False
    if conf < MIN_CONF:
        return False
    if w * h < MIN_AREA:
        return False
    aspect = w / (h + 1e-6)
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    return True

# =====================================================
# WARNA PER BEHAVIOR
# =====================================================

BEHAVIOR_COLOR = {
    "Sangat Aktif": (0, 255, 0),
    "Aktif":        (0, 200, 255),
    "Diam":         (255, 200, 0),
    "Tidak Aktif":  (0, 0, 255),
}

# =====================================================
# KLASIFIKASI BEHAVIOR
# =====================================================

def classify_behavior(movement, inactive):
    if inactive > 40:
        return "Tidak Aktif"
    elif movement < 2:
        return "Diam"
    elif movement < 15:
        return "Aktif"
    else:
        return "Sangat Aktif"

# =====================================================
# TRACKING DATA
# =====================================================

track_history   = defaultdict(list)
inactive_frames = defaultdict(int)

# =====================================================
# HEATMAP
# =====================================================

def update_heatmap(heatmap, cx, cy, w, h, HEIGHT, WIDTH):
    radius = max(int((w + h) / 4), 20)
    size   = radius * 2 + 1
    kernel = cv2.getGaussianKernel(size, radius / 2)
    kernel_2d = kernel @ kernel.T
    kernel_2d = kernel_2d / kernel_2d.max()

    x1, y1 = cx - radius, cy - radius
    x2, y2 = cx + radius + 1, cy + radius + 1

    kx1 = max(0, -x1);  ky1 = max(0, -y1)
    kx2 = size - max(0, x2 - WIDTH)
    ky2 = size - max(0, y2 - HEIGHT)

    hx1 = max(0, x1);  hy1 = max(0, y1)
    hx2 = min(WIDTH, x2);  hy2 = min(HEIGHT, y2)

    if hx2 > hx1 and hy2 > hy1:
        heatmap[hy1:hy2, hx1:hx2] += kernel_2d[ky1:ky2, kx1:kx2]

# =====================================================
# VIDEO LIST
# =====================================================

video_files = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.endswith((".mp4", ".avi", ".mov"))
]

print("Jumlah video:", len(video_files))

# =====================================================
# LOOP VIDEO
# =====================================================

for video_name in video_files[:1]:

    print(f"\nMemproses: {video_name}")

    video_path = os.path.join(VIDEO_FOLDER, video_name)
    cap        = cv2.VideoCapture(video_path)
    fps_video  = cap.get(cv2.CAP_PROP_FPS)

    WIDTH, HEIGHT  = 640, 480
    final_behavior = {}
    frame_count    = 0
    heatmap        = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    cv2.namedWindow("Chicken - Perilaku", cv2.WINDOW_NORMAL)

    annotated_frame = None

    # =====================================================
    # LOOP FRAME
    # =====================================================

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        if frame_count % 3 != 0:
            if annotated_frame is not None:
                cv2.imshow("Chicken - Perilaku", annotated_frame)
            else:
                cv2.imshow("Chicken - Perilaku", frame)
            if cv2.waitKey(int(1000 / fps_video)) == ord('q'):
                break
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            imgsz=640,
            conf=MIN_CONF,
            classes=list(ALLOWED_CLASSES),
        )

        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:

            boxes     = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs     = results[0].boxes.conf.cpu().numpy()
            cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                x, y, w, h = box

                if not is_valid_chicken(cls_id, conf, w, h):
                    continue

                center = (int(x), int(y))
                track  = track_history[track_id]
                track.append(center)
                if len(track) > 30:
                    track.pop(0)

                movement = 0
                if len(track) >= 2:
                    prev_x, prev_y = track[-2]
                    movement = np.sqrt(
                        (center[0] - prev_x) ** 2 +
                        (center[1] - prev_y) ** 2
                    )

                if movement < 2:
                    inactive_frames[track_id] += 1
                else:
                    inactive_frames[track_id] = 0

                behavior = classify_behavior(movement, inactive_frames[track_id])
                final_behavior[track_id] = behavior

                color = BEHAVIOR_COLOR.get(behavior, (255, 255, 255))

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                label   = f"ID:{track_id} {behavior}"
                label_y = max(y1 - 5, 12)
                cv2.putText(annotated_frame, label,
                            (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

                if len(track) >= 2:
                    points = np.array(track, dtype=np.int32)
                    cv2.polylines(annotated_frame, [points], False, (255, 0, 0), 1)

                update_heatmap(heatmap, center[0], center[1], int(w), int(h), HEIGHT, WIDTH)

        # =====================================================
        # LEGEND
        # =====================================================

        legend_y = HEIGHT - 95
        for label, color in BEHAVIOR_COLOR.items():
            cv2.rectangle(annotated_frame,
                          (10, legend_y), (25, legend_y + 15), color, -1)
            cv2.putText(annotated_frame, label,
                        (30, legend_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            legend_y += 22

        # FPS
        fps = 1 / (time.time() - start_time + 1e-6)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Chicken - Perilaku", annotated_frame)

        if cv2.waitKey(int(1000 / fps_video)) == ord('q'):
            break

    # =====================================================
    # SIMPAN HEATMAP SEBAGAI IMAGE
    # =====================================================

    heatmap_blur  = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_norm  = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Simpan heatmap murni
    heatmap_path = os.path.join(OUTPUT_FOLDER, f"heatmap_{video_name}.png")
    cv2.imwrite(heatmap_path, heatmap_color)
    print(f"Heatmap disimpan: {heatmap_path}")

    # Simpan heatmap overlay di atas frame terakhir video
    cap2 = cv2.VideoCapture(video_path)
    last_frame = None
    while True:
        ok, f = cap2.read()
        if not ok:
            break
        last_frame = f
    cap2.release()

    if last_frame is not None:
        last_frame   = cv2.resize(last_frame, (WIDTH, HEIGHT))
        overlay      = cv2.addWeighted(last_frame, 0.5, heatmap_color, 0.5, 0)
        overlay_path = os.path.join(OUTPUT_FOLDER, f"heatmap_overlay_{video_name}.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"Heatmap overlay disimpan: {overlay_path}")

    # =====================================================
    # HASIL
    # =====================================================

    total_ayam        = len(final_behavior)
    aktif_count       = sum(1 for b in final_behavior.values() if b in ("Aktif", "Sangat Aktif"))
    diam_count        = sum(1 for b in final_behavior.values() if b == "Diam")
    tidak_aktif_count = sum(1 for b in final_behavior.values() if b == "Tidak Aktif")

    persentase_tidak_aktif = (
        (tidak_aktif_count / total_ayam) * 100 if total_ayam > 0 else 0
    )

    if persentase_tidak_aktif > 30:
        status = "Potensi Sakit Tinggi"
    elif persentase_tidak_aktif > 15:
        status = "Waspada"
    else:
        status = "Normal"

    print("\n===== HASIL PERILAKU AYAM =====")
    print(f"Total Ayam            : {total_ayam}")
    print(f"Aktif + Sangat Aktif  : {aktif_count}")
    print(f"Diam                  : {diam_count}")
    print(f"Tidak Aktif           : {tidak_aktif_count}")
    print(f"Persentase Tidak Aktif: {persentase_tidak_aktif:.2f}%")
    print(f"Status                : {status}")

    for tid, beh in final_behavior.items():
        csv_writer.writerow([video_name, tid, beh, status])

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai!")

csv_file.close()
print("\nCSV perilaku disimpan:", csv_path)