import os
import time
import cv2
from ximea import xiapi

# =========================
# KONFIGURATION
# =========================
EXPOSURE_US       = 30000
GAIN              = 5.0
WB_RED            = 1.75
WB_GREEN          = 1.0
WB_BLUE           = 2.25

# Burst-Parameter
BURST_INTERVAL_SEC = 0.05   # Pause zwischen den Frames im Burst (0 = so schnell wie möglich)
NUM_FRAMES         = 150     # Anzahl Frames pro Burst
SAVE_BASE          = "captures"

# Vorschau
PREVIEW_SCALE = 0.35        # Skalierung für das Vorschaufenster

# ── Bandbreiten-Management ─────────────────────────────────────────────────
# USB3-Host-Controller liefert ~350 MB/s brutto; 3 Kameras teilen sich das.
# TOTAL_BANDWIDTH_MBPS aufgeteilt durch Kameraanzahl = Limit pro Kamera.
# Auf None setzen, um das Limit zu deaktivieren.
TOTAL_BANDWIDTH_MBPS = 350          # MB/s für den gesamten USB-Bus

# Hardware-Downsampling direkt auf der Kamera:
#   'XI_DWN_1x1' → volle Auflösung  (z.B. 1280×1024, RGB24 ≈ 75 MB/s @ 20 fps)
#   'XI_DWN_2x2' → halbe Auflösung  (z.B.  640× 512, RGB24 ≈ 19 MB/s @ 20 fps)
# 2x2 reduziert den USB-Transfer um Faktor 4 und entlastet die Bandbreite stark.
DOWNSAMPLING = 'XI_DWN_2x2'

# Seriennummern → logischer Kamera-Index
CAM_SERIALS = {
    "CUCAU1829019": 0,  # Cam1
    "CUCAU1829041": 1,  # Cam2
    "CUCAU1829031": 2,  # Cam3
}

# =========================
# KAMERA-SETUP
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
# BURST-AUFNAHME
# =========================
def take_burst(cams, burst_idx):
    """
    Nimmt NUM_FRAMES Frames mit allen Kameras auf.
    Pro Frame werden alle Kameras direkt nacheinander ausgelesen
    (Versatz < 5 ms) – zuverlässiger als Threading, da xiapi
    nicht thread-safe ist, wenn Kamera-Objekte thread-übergreifend
    genutzt werden.
    """
    n_cams = len(cams)
    print(f"\n📸 Burst #{burst_idx:03d} — {NUM_FRAMES} Frames × {n_cams} Kameras "
          f"(Intervall {BURST_INTERVAL_SEC*1000:.0f} ms) …")

    # Ordner anlegen: captures/burst_001/cam0/, cam1/, cam2/
    burst_dir = os.path.join(SAVE_BASE, f"burst_{burst_idx:03d}")
    for i in range(n_cams):
        os.makedirs(os.path.join(burst_dir, f"cam{i+1}"), exist_ok=True)

    # all_frames[cam_idx] = Liste der aufgenommenen Frames
    all_frames = [[] for _ in range(n_cams)]

    t_start = time.time()
    for frame_idx in range(NUM_FRAMES):
        t_frame = time.time()

        # Alle Kameras in einem Durchlauf auslesen
        for cam_idx, cam in enumerate(cams):
            try:
                img = xiapi.Image()
                cam.get_image(img)
                all_frames[cam_idx].append(img.get_image_data_numpy().copy())
            except Exception as e:
                print(f"[ERR] Cam{cam_idx} Frame {frame_idx}: {e}")

        # Intervall einhalten
        if BURST_INTERVAL_SEC > 0:
            elapsed = time.time() - t_frame
            wait = BURST_INTERVAL_SEC - elapsed
            if wait > 0:
                time.sleep(wait)

    duration = time.time() - t_start

    # Ergebnis loggen und speichern
    saved = 0
    for cam_idx, frames in enumerate(all_frames):
        print(f"[INFO] Cam{cam_idx+1}: {len(frames)}/{NUM_FRAMES} Frames aufgenommen")
        for frame_idx, frame in enumerate(frames):
            path = os.path.join(burst_dir, f"cam{cam_idx+1}", f"img_{frame_idx:04d}.png")
            cv2.imwrite(path, frame)
            saved += 1

    fps = NUM_FRAMES / duration if duration > 0 else 0
    print(f"✅ Burst #{burst_idx:03d}: {saved} Bilder in {duration:.2f} s "
          f"({fps:.1f} fps) → {burst_dir}")
    return saved > 0


# =========================
# HAUPTPROGRAMM
# =========================
def main():
    # ── Kameras öffnen ──────────────────────────────────────────
    raw_cams = [xiapi.Camera(0), xiapi.Camera(1), xiapi.Camera(2)]
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
        print("   Nutze Reihenfolge wie erkannt (0, 1, 2)…")
        cams = list(serial_to_cam.values())
    else:
        slot_to_cam = {CAM_SERIALS[sn]: cam for sn, cam in serial_to_cam.items()}
        cams = [slot_to_cam[i] for i in range(3)]
        print("✅ Kameras korrekt sortiert nach Seriennummer")

    os.makedirs(SAVE_BASE, exist_ok=True)

    # ── Vorschaufenster ─────────────────────────────────────────
    WIN = "Vorschau  |  SPACE = Burst  |  Q / ESC = Beenden"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    burst_count  = 0
    status_text  = "Bereit — SPACE drücken um Burst zu starten"
    status_color = (0, 220, 0)   # Grün

    print(f"\n🎬 Live-Vorschau aktiv")
    print(f"   SPACE  →  Burst starten  ({NUM_FRAMES} Frames, "
          f"{BURST_INTERVAL_SEC*1000:.0f} ms Intervall)")
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
                        f"Bursts: {burst_count}  |  "
                        f"Frames/Burst: {NUM_FRAMES}  |  "
                        f"Intervall: {BURST_INTERVAL_SEC*1000:.0f} ms",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(WIN, combined)
            key = cv2.waitKey(1) & 0xFF

            # ── Tastenbelegung ───────────────────────────────────
            if key in (ord('q'), 27):           # Q oder ESC → Beenden
                break

            elif key == ord(' '):               # SPACE → Countdown + Burst
                # ── 5-Sekunden-Countdown ─────────────────────────
                COUNTDOWN_SEC = 5
                for remaining in range(COUNTDOWN_SEC, 0, -1):
                    countdown_previews = []
                    for cam in cams:
                        frame = capture_frame(cam)
                        small = cv2.resize(frame, (0, 0),
                                           fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
                        countdown_previews.append(small)
                    cd = cv2.hconcat(countdown_previews)
                    _, cd_w2 = cd.shape[:2]

                    cv2.rectangle(cd, (0, 0), (cd_w2, 70), (0, 0, 0), -1)
                    cv2.putText(cd, f"Startet in  {remaining} …",
                                (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (0, 200, 255), 3, cv2.LINE_AA)
                    cv2.imshow(WIN, cd)
                    cv2.waitKey(1)
                    time.sleep(1.0)

                # ── REC-Overlay, dann Burst ───────────────────────
                rec = cd.copy()
                cv2.rectangle(rec, (0, 0), (cd_w2, 70), (0, 0, 0), -1)
                cv2.putText(rec, "● REC …",
                            (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow(WIN, rec)
                cv2.waitKey(1)

                burst_count += 1
                ok = take_burst(cams, burst_count)

                if ok:
                    status_text  = f"✓ Burst #{burst_count:03d} gespeichert — bereit"
                    status_color = (0, 220, 0)
                else:
                    status_text  = f"✗ Burst #{burst_count:03d} fehlgeschlagen"
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
        print("✅ Alle Kameras gestoppt")


if __name__ == "__main__":
    main()
