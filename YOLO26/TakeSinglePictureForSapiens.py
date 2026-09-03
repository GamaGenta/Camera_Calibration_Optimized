import os
import time
from datetime import datetime

import cv2
from ximea import xiapi

# =========================
# KONFIGURATION  (Kameraeinstellungen identisch zu TakeManyPicturesForSapiens.py)
# =========================
EXPOSURE_US       = 30000
GAIN              = 5.0
WB_RED            = 1.75
WB_GREEN          = 1.0
WB_BLUE           = 2.25

# Countdown vor der Aufnahme
COUNTDOWN_SEC = 5

SAVE_BASE = "captures1"

# Vorschau
PREVIEW_SCALE = 0.35        # Skalierung für das Vorschaufenster

# ── Bandbreiten-Management ─────────────────────────────────────────────────
# USB3-Host-Controller liefert ~350 MB/s brutto; 3 Kameras teilen sich das.
TOTAL_BANDWIDTH_MBPS = 350          # MB/s für den gesamten USB-Bus

# Hardware-Downsampling direkt auf der Kamera:
#   'XI_DWN_1x1' → volle Auflösung  (4112×3008 – passt 1:1 zur Kalibrierung)
#   'XI_DWN_2x2' → halbe Auflösung  (2056×1504)
DOWNSAMPLING = 'XI_DWN_2x2'

# Seriennummern → logischer Kamera-Index
CAM_SERIALS = {
    "CUCAU1829019": 0,  # Cam1
    "CUCAU1829041": 1,  # Cam2
    "CUCAU1829031": 2,  # Cam3
    #: 3,  # Cam4
}

# =========================
# KAMERA-SETUP  (1:1 aus TakeManyPicturesForSapiens.py)
# =========================
def setup_camera(cam, name):
    try:
        print(f"\n[INFO] Öffne {name}...")
        cam.open_device()
        sn = cam.get_device_sn().decode()
        print(f"[OK] {name} geöffnet. SN: {sn}")

        try:
            cam.set_imgdataformat('XI_IMG_FORMAT_RGB24')
        except Exception:
            cam.set_param('imgdataformat', 'XI_RGB24')

        try:
            cam.disable_aeag()
        except Exception:
            pass

        cam.set_offsetX(0)
        cam.set_offsetY(0)

        cam.set_exposure(EXPOSURE_US)
        cam.set_gain(GAIN)
        cam.disable_auto_wb()
        cam.set_wb_kr(WB_RED)
        cam.set_wb_kg(WB_GREEN)
        cam.set_wb_kb(WB_BLUE)

        # ── Hardware-Downsampling ──────────────────────────────
        try:
            cam.set_downsampling(DOWNSAMPLING)
            print(f"[OK] {name}: Downsampling = {DOWNSAMPLING}")
        except Exception as e:
            print(f"[WARN] {name}: Downsampling nicht gesetzt ({e})")

        # ── Bandbreiten-Limit pro Kamera ───────────────────────
        if TOTAL_BANDWIDTH_MBPS is not None:
            per_cam_mbps = TOTAL_BANDWIDTH_MBPS // len(CAM_SERIALS)
            try:
                cam.set_limit_bandwidth(per_cam_mbps)
                print(f"[OK] {name}: Bandbreiten-Limit = {per_cam_mbps} MB/s")
            except Exception as e:
                print(f"[WARN] {name}: Bandbreiten-Limit nicht gesetzt ({e})")

        cam.start_acquisition()

        # Warmup
        for _ in range(5):
            img = xiapi.Image()
            cam.get_image(img)

        print(f"✅ {name} bereit | SN: {sn} | "
              f"Exposure: {cam.get_exposure()} µs | Gain: {cam.get_gain()}")
        return sn

    except Exception as e:
        print(f"❌ Fehler bei {name}: {e}")
        return None


def capture_frame(cam):
    img = xiapi.Image()
    cam.get_image(img)
    return img.get_image_data_numpy()


# =========================
# EINZELAUFNAHME
# =========================
def take_single_shot(cams, session_dir, shot_idx):
    """
    Nimmt mit allen Kameras zeitgleich EIN Bild auf und speichert es als
    session_dir/cam1/img_XXXX.png  (gleicher Dateiname über alle Kameras,
    damit die Triangulations-Pipeline die drei Ansichten zuordnen kann).

    xiapi ist nicht thread-safe – die Kameras werden daher direkt
    nacheinander ausgelesen (Versatz < 5 ms), was zuverlässiger ist als
    Threading.
    """
    n_cams = len(cams)
    fname = f"img_{shot_idx:04d}.png"

    # Alle Kameras so schnell wie möglich hintereinander auslesen
    frames = []
    for cam_idx, cam in enumerate(cams):
        try:
            img = xiapi.Image()
            cam.get_image(img)
            frames.append(img.get_image_data_numpy().copy())
        except Exception as e:
            print(f"[ERR] Cam{cam_idx+1}: {e}")
            frames.append(None)

    # Speichern
    saved = 0
    for cam_idx, frame in enumerate(frames):
        if frame is None:
            continue
        cam_dir = os.path.join(session_dir, f"cam{cam_idx+1}")
        os.makedirs(cam_dir, exist_ok=True)
        path = os.path.join(cam_dir, fname)
        cv2.imwrite(path, frame)
        saved += 1

    print(f"✅ Aufnahme #{shot_idx:04d}: {saved}/{n_cams} Bilder gespeichert "
          f"→ {session_dir}/camN/{fname}")
    return saved == n_cams


# =========================
# HAUPTPROGRAMM
# =========================
def main():
    # ── Kameras öffnen ──────────────────────────────────────────
    raw_cams = [xiapi.Camera(0), xiapi.Camera(1), xiapi.Camera(2), xiapi.Camera(3)]
    serial_to_cam = {}

    for i, cam in enumerate(raw_cams):
        sn = setup_camera(cam, f"Cam{i}")
        if sn is None:
            print("Abbruch wegen Kamera-Fehler")
            return
        serial_to_cam[sn] = cam

    # Kameras in die richtige Reihenfolge bringen
    if not all(sn in CAM_SERIALS for sn in serial_to_cam):
        print("⚠️  Unbekannte Seriennummern — bitte CAM_SERIALS aktualisieren.")
        print("   Gefundene SNs:", list(serial_to_cam.keys()))
        print("   Nutze Reihenfolge wie erkannt (0, 1, 2, 3)…")
        cams = list(serial_to_cam.values())
    else:
        slot_to_cam = {CAM_SERIALS[sn]: cam for sn, cam in serial_to_cam.items()}
        cams = [slot_to_cam[i] for i in range(4)] #range 3 -> 4
        print("✅ Kameras korrekt sortiert nach Seriennummer")

    # ── Neuer Aufnahme-Ordner (Zeitstempel) ─────────────────────
    session_dir = os.path.join(
        SAVE_BASE, "single_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for i in range(len(cams)):
        os.makedirs(os.path.join(session_dir, f"cam{i+1}"), exist_ok=True)
    print(f"📁 Speicherordner: {session_dir}  (cam1/ cam2/ cam3/ cam4/)")

    # ── Vorschaufenster ─────────────────────────────────────────
    WIN = "Vorschau  |  SPACE = Foto (5s Countdown)  |  Q / ESC = Beenden"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    shot_count   = 0
    status_text  = "Bereit — SPACE druecken fuer Foto"
    status_color = (0, 220, 0)   # Grün

    print(f"\n🎬 Live-Vorschau aktiv")
    print(f"   SPACE   →  Foto aufnehmen ({COUNTDOWN_SEC}s Countdown)")
    print(f"   Q / ESC →  Beenden\n")

    try:
        while True:
            # ── Live-Vorschau ────────────────────────────────────
            previews = []
            for cam in cams:
                frame = capture_frame(cam)
                small = cv2.resize(frame, (0, 0),
                                   fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                previews.append(small)

            combined = cv2.hconcat(previews)
            _, w = combined.shape[:2]

            # Status-Overlay
            cv2.rectangle(combined, (0, 0), (w, 70), (0, 0, 0), -1)
            cv2.putText(combined, status_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, status_color, 2, cv2.LINE_AA)
            cv2.putText(combined,
                        f"Fotos: {shot_count}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(WIN, combined)
            key = cv2.waitKey(1) & 0xFF

            # ── Tastenbelegung ───────────────────────────────────
            if key in (ord('q'), 27):           # Q oder ESC → Beenden
                break

            elif key == ord(' '):               # SPACE → Countdown + Foto
                # ── 5-Sekunden-Countdown ─────────────────────────
                for remaining in range(COUNTDOWN_SEC, 0, -1):
                    countdown_previews = []
                    for cam in cams:
                        frame = capture_frame(cam)
                        small = cv2.resize(frame, (0, 0),
                                           fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                        countdown_previews.append(small)
                    cd = cv2.hconcat(countdown_previews)
                    _, cd_w = cd.shape[:2]

                    cv2.rectangle(cd, (0, 0), (cd_w, 70), (0, 0, 0), -1)
                    cv2.putText(cd, f"Foto in  {remaining} …",
                                (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (0, 200, 255), 3, cv2.LINE_AA)
                    cv2.imshow(WIN, cd)
                    cv2.waitKey(1)
                    time.sleep(1.0)

                # ── Foto aufnehmen ───────────────────────────────
                shot_count += 1
                ok = take_single_shot(cams, session_dir, shot_count)

                if ok:
                    status_text  = f"✓ Foto #{shot_count:04d} gespeichert — bereit"
                    status_color = (0, 220, 0)
                else:
                    status_text  = f"✗ Foto #{shot_count:04d} fehlgeschlagen"
                    status_color = (0, 0, 255)

    except KeyboardInterrupt:
        print("\n[INFO] Unterbrochen")
    finally:
        cv2.destroyAllWindows()
        for cam in cams:
            try:
                cam.stop_acquisition()
                cam.close_device()
            except Exception:
                pass
        print(f"✅ Alle Kameras gestoppt | {shot_count} Fotos in {session_dir}")


if __name__ == "__main__":
    main()
