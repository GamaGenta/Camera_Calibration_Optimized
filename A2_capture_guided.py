"""
A2_capture_guided.py
====================
Kalibrier-Aufnahme mit LIVE-COVERAGE-FEEDBACK.

Motivation
----------
Die Analyse hat gezeigt: nicht die Mathematik, sondern die DATENABDECKUNG
ist jetzt der limitierende Faktor (Cam2 25%, Cam3 40%, Cam1 48% der
Bildregionen). Distortion und Hauptpunkt werden nur dort zuverlaessig
geschaetzt, wo auch Messpunkte liegen -- besonders an den Bildraendern und
bei stark gekippten Boards.

Dieses Skript ersetzt das blinde Abdruecken durch gezielte Fuehrung:
  * Live-Detektion des ChArUco-Boards (halbe Aufloesung, ~4 fps).
  * Ein 8x6-Raster zeigt pro Kamera, welche Bildregionen schon abgedeckt
    sind (gruen) und welche noch fehlen (rot). Die aktuelle Detektion ist
    gelb -> "wenn ich jetzt speichere, kommen diese Zellen dazu".
  * Kippwinkel-Anzeige + Histogramm -> erzwingt Perspektiven-Vielfalt.
  * Richtungs-Hinweis ("Board nach oben-links") zur naechsten Luecke.
  * Beim Start wird die Coverage aus den BEREITS vorhandenen Bildern
    initialisiert (--noseed zum Ueberspringen) -> du siehst sofort, was
    relativ zum bestehenden Datensatz noch fehlt.

Die gespeicherten Bilder bleiben VOLL aufgeloest und landen in genau den
Ordnern (calib_1_2, calib_1_3, ...), die die restliche Pipeline erwartet.
Steuerung identisch zu A_ThreeCameraStreamTakePics.py.

Aufruf (Live):      python A2_capture_guided.py
Aufruf (Selbsttest, ohne Kameras, schreibt annotierte Beispielbilder):
                    python A2_capture_guided.py --selftest


                    Erweiterung um die 4. Kamerea: im rahmen eines Master Projekts 
"""

import sys
import os
import time
import glob
import numpy as np
import cv2

import calib_common as cc

# ---- Kamera-Einstellungen (identisch zu A) ----
EXPOSURE_US = 30000
GAIN = 5.0
WB_RED, WB_GREEN, WB_BLUE = 1.75, 1.0, 2.25

# ---- Feedback-Einstellungen ----
DET_SCALE = 0.5                 # Detektion auf halber Aufloesung (Speed)
GRID = (8, 6)                   # (GX, GY) Raster fuer Coverage -- wie im Calib-Score
DISPLAY_W = 980                 # Preview-Breite (Hoehe aus Seitenverhaeltnis)
TILT_BINS = [0, 15, 30, 45, 90] # Grad-Grenzen fuer Kippwinkel-Vielfalt
MIN_CORNERS = 6

GREEN = (0, 180, 0)
RED = (40, 40, 200)
YELLOW = (0, 215, 235)
WHITE = (245, 245, 245)


# ======================================================================
# Coverage-Status pro Kamera
# ======================================================================
class CoverageTracker:
    def __init__(self, grid=GRID):
        self.gx, self.gy = grid
        self.hit = np.zeros((self.gy, self.gx), dtype=bool)     # je gespeichert
        self.tilt_counts = np.zeros(len(TILT_BINS) - 1, dtype=int)
        self.saved = 0

    def cells_of(self, norm_pts):
        """Rasterzellen (Spalte,Zeile) zu normalisierten Punkten [0,1]."""
        cx = np.clip((norm_pts[:, 0] * self.gx).astype(int), 0, self.gx - 1)
        cy = np.clip((norm_pts[:, 1] * self.gy).astype(int), 0, self.gy - 1)
        return cx, cy

    def update(self, norm_pts, tilt_deg):
        cx, cy = self.cells_of(norm_pts)
        self.hit[cy, cx] = True
        b = np.digitize([tilt_deg], TILT_BINS)[0] - 1
        b = int(np.clip(b, 0, len(self.tilt_counts) - 1))
        self.tilt_counts[b] += 1
        self.saved += 1

    def fraction(self):
        return float(self.hit.sum() / (self.gx * self.gy))

    def missing_hint(self):
        """Richtungstext zur groessten zusammenhaengenden Luecke."""
        empty = np.argwhere(~self.hit)   # (row,col)
        if len(empty) == 0:
            return "Vollstaendig - jetzt Kippwinkel variieren!"
        cy = empty[:, 0].mean() / (self.gy - 1 + 1e-9)
        cx = empty[:, 1].mean() / (self.gx - 1 + 1e-9)
        vert = "oben" if cy < 0.4 else ("unten" if cy > 0.6 else "")
        horiz = "links" if cx < 0.4 else ("rechts" if cx > 0.6 else "")
        d = "-".join([p for p in (vert, horiz) if p]) or "Rand/Ecken"
        return f"Board nach: {d}"

    def tilt_hint(self):
        if self.saved < 4:
            return ""
        weakest = int(np.argmin(self.tilt_counts))
        lo, hi = TILT_BINS[weakest], TILT_BINS[weakest + 1]
        if self.tilt_counts[weakest] < max(2, self.saved // 6):
            return f"Mehr Aufnahmen mit Kippung {lo}-{hi}deg"
        return ""


# ======================================================================
# Frame-Analyse (gemeinsam fuer Live + Selbsttest)
# ======================================================================
def analyze_frame(full_gray, detector, board, K, D):
    """
    Detektiert auf halber Aufloesung. Liefert dict mit:
      norm_pts (N,2) in [0,1], num, tilt_deg, det_pts (N,2) in DET-Pixeln
    oder None.
    """
    work = cv2.resize(full_gray, None, fx=DET_SCALE, fy=DET_SCALE,
                      interpolation=cv2.INTER_AREA)
    wh = (work.shape[1], work.shape[0])
    det = cc.detect_charuco(work, detector, board, min_corners=MIN_CORNERS)
    if det is None:
        return None
    det_pts = det["img_points"].reshape(-1, 2)
    norm = det_pts / np.array([wh[0], wh[1]], dtype=np.float64)

    # Kippwinkel via solvePnP (auf volle Aufloesung hochskaliert -> passt zu K)
    tilt = float("nan")
    try:
        obj = det["obj_points"].reshape(-1, 1, 3)
        img_full = (det_pts / DET_SCALE).reshape(-1, 1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img_full, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            n = R @ np.array([0.0, 0.0, 1.0])      # Board-Normale in Cam-Frame
            tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))
    except cv2.error:
        pass

    return dict(norm_pts=norm, num=len(det_pts), tilt_deg=tilt, det_pts=det_pts)


def guess_K(full_shape):
    h, w = full_shape[:2]
    f = 1.2 * w
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    return K, np.zeros((5, 1))


def load_intrinsics(cam, full_shape):
    for name in (f"mono_cam{cam}.pkl", f"mono_cam{cam}_pinhole.pkl"):
        if os.path.exists(name):
            import pickle
            with open(name, "rb") as f:
                d = pickle.load(f)
            return d["K"], d["D"]
    return guess_K(full_shape)


# ======================================================================
# Overlay-Zeichnung
# ======================================================================
def draw_overlay(preview, tracker, current, cam_label):
    """preview: BGR (wird in-place annotiert). current: analyze_frame-dict|None."""
    ph, pw = preview.shape[:2]
    gx, gy = tracker.gx, tracker.gy
    cw, chh = pw / gx, ph / gy

    overlay = preview.copy()
    # gespeicherte Zellen fuellen
    for j in range(gy):
        for i in range(gx):
            x0, y0 = int(i * cw), int(j * chh)
            x1, y1 = int((i + 1) * cw), int((j + 1) * chh)
            if tracker.hit[j, i]:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), GREEN, -1)
    cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, preview)

    # Rasterlinien + leere Zellen rot umranden
    for j in range(gy):
        for i in range(gx):
            x0, y0 = int(i * cw), int(j * chh)
            x1, y1 = int((i + 1) * cw), int((j + 1) * chh)
            col = GREEN if tracker.hit[j, i] else RED
            cv2.rectangle(preview, (x0, y0), (x1, y1), col, 1)

    # aktuelle Detektion: gelbe Zellen + Punkte
    if current is not None:
        cx, cy = tracker.cells_of(current["norm_pts"])
        for i, j in set(zip(cx.tolist(), cy.tolist())):
            x0, y0 = int(i * cw), int(j * chh)
            x1, y1 = int((i + 1) * cw), int((j + 1) * chh)
            cv2.rectangle(preview, (x0, y0), (x1, y1), YELLOW, 2)
        for p in current["norm_pts"]:
            cv2.circle(preview, (int(p[0] * pw), int(p[1] * ph)), 2, YELLOW, -1)

    # Textzeile(n)
    def txt(s, y, col=WHITE):
        cv2.putText(preview, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(preview, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    cov = tracker.fraction()
    cov_col = GREEN if cov >= 0.55 else (YELLOW if cov >= 0.4 else RED)
    txt(f"{cam_label}  Coverage {cov:0.0%}  gespeichert: {tracker.saved}", 22, cov_col)
    if current is not None:
        t = current["tilt_deg"]
        ts = f"{t:0.0f}deg" if t == t else "n/a"   # NaN check
        txt(f"Detektion: {current['num']} Ecken  Kippung {ts}", 44)
    else:
        txt("Kein Board erkannt", 44, RED)
    txt(tracker.missing_hint(), ph - 30, YELLOW)
    th = tracker.tilt_hint()
    if th:
        txt(th, ph - 10, YELLOW)
    return preview


def to_gray(frame):
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


# ======================================================================
# Coverage aus vorhandenen Bildern initialisieren
# ======================================================================
def seed_from_disk(cam, folders, tracker, detector, board, K, D):
    files = []
    for fo in folders:
        files += glob.glob(os.path.join(fo, f"cam{cam}_*.png"))
    files = sorted(set(files))
    if not files:
        return
    print(f"  [Cam{cam}] Seede Coverage aus {len(files)} Bildern ...", flush=True)
    for k, fp in enumerate(files):
        g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        cur = analyze_frame(g, detector, board, K, D)
        if cur is not None:
            tracker.update(cur["norm_pts"], cur["tilt_deg"]
                           if cur["tilt_deg"] == cur["tilt_deg"] else 0.0)
    print(f"  [Cam{cam}] Start-Coverage {tracker.fraction():0.0%} "
          f"({tracker.saved} Bilder)", flush=True)


# ======================================================================
# SELBSTTEST (ohne Kameras)
# ======================================================================
def selftest():
    print("=== SELBSTTEST (ohne Kameras) ===")
    board = cc.make_board()
    detector = cc.make_detector(board)
    samples = {1: "calib_1_2", 2: "calib_1_2", 3: "calib_1_3", 4: "calib_1_4"}
    for cam, folder in samples.items():
        files = sorted(glob.glob(os.path.join(folder, f"cam{cam}_*.png")))
        if not files:
            print(f"Cam{cam}: keine Bilder."); continue
        g0 = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        K, D = load_intrinsics(cam, g0.shape)
        tr = CoverageTracker()
        # Coverage aus ersten 15 Bildern aufbauen
        for fp in files[:15]:
            g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            cur = analyze_frame(g, detector, board, K, D)
            if cur:
                tr.update(cur["norm_pts"],
                          cur["tilt_deg"] if cur["tilt_deg"] == cur["tilt_deg"] else 0.0)
        # annotiertes Beispielbild schreiben
        g = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        cur = analyze_frame(g, detector, board, K, D)
        ph = int(DISPLAY_W * g.shape[0] / g.shape[1])
        prev = cv2.cvtColor(cv2.resize(g, (DISPLAY_W, ph)), cv2.COLOR_GRAY2BGR)
        draw_overlay(prev, tr, cur, f"Cam{cam}")
        out = f"selftest_overlay_cam{cam}.png"
        cv2.imwrite(out, prev)
        tilt = cur["tilt_deg"] if cur else float("nan")
        print(f"Cam{cam}: Coverage {tr.fraction():0.0%}, tilt(Beispiel)="
              f"{tilt:0.1f}deg, Hinweis='{tr.missing_hint()}' -> {out}")
    print("OK.")


# ======================================================================
# LIVE-AUFNAHME (XIMEA)
# ======================================================================
def setup_camera(cam, name):
    print(f"[INFO] Oeffne {name} ...")
    cam.open_device()
    sn = cam.get_device_sn().decode()
    try:
        cam.set_imgdataformat('XI_IMG_FORMAT_RGB24')
    except Exception:
        cam.set_param('imgdataformat', 'XI_RGB24')
    cam.set_offsetX(0); cam.set_offsetY(0)
    try:
        cam.set_downsampling('XI_DWN_1x1')
    except Exception:
        pass
    cam.set_exposure(EXPOSURE_US); cam.set_gain(GAIN)
    cam.disable_auto_wb()
    cam.set_wb_kr(WB_RED); cam.set_wb_kg(WB_GREEN); cam.set_wb_kb(WB_BLUE)
    cam.start_acquisition()
    print(f"[OK] {name} SN={sn}")
    return sn


def live(seed=True):
    from ximea import xiapi
    board = cc.make_board()
    detector = cc.make_detector(board)

    cams = [xiapi.Camera(i) for i in range(4)]
    imgs = [xiapi.Image() for _ in range(4)]
    sns, ok = [], []
    for idx, c in enumerate(cams, 1):
        try:
            sns.append(setup_camera(c, f"Cam{idx}")); ok.append(True)
        except Exception as e:
            print(f"[FEHLER] Cam{idx}: {e}"); sns.append(None); ok.append(False)
    if not any(ok):
        print("Keine Kameras."); return

    for d in ["calib_1_2", "calib_1_3", "calib_1_4", "calib_single_1",
              "calib_single_2", "calib_single_3", "calib_single_4"]:
        os.makedirs(d, exist_ok=True)

    # Coverage-Tracker + Intrinsics je Kamera
    trackers, Ks, Ds = {}, {}, {}
    src = {1: ["calib_1_2", "calib_1_3"], 2: ["calib_1_2"], 3: ["calib_1_3"], 4: ["calib_1_4"]}
    for i in (1, 2, 3, 4):
        trackers[i] = CoverageTracker()
        Ks[i], Ds[i] = load_intrinsics(i, (3008, 4112, 3))
    if seed:
        print("Initialisiere Coverage aus vorhandenen Bildern ...")
        for i in (1, 2, 3, 4):
            if ok[i - 1]:
                seed_from_disk(i, src[i], trackers[i], detector, board, Ks[i], Ds[i])

    print("\n[STEUERUNG] p: alle | 6/7/8/9: einzeln | 2: Paar1+2 | 3: Paar1+3 | 4: Paar1+4 | q: Ende")
    ph = int(DISPLAY_W * 3008 / 4112)

    while True:
        frames = {1: None, 2: None, 3: None, 4: None}
        currents = {1: None, 2: None, 3: None, 4: None}
        for i, (c, im, good) in enumerate(zip(cams, imgs, ok), 1):
            if not good:
                continue
            try:
                c.get_image(im)
                raw = im.get_image_data_numpy()
                frames[i] = raw
                gray = to_gray(raw)
                cur = analyze_frame(gray, detector, board, Ks[i], Ds[i])
                currents[i] = cur
                prev = cv2.resize(raw, (DISPLAY_W, ph))
                if prev.ndim == 2:
                    prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
                draw_overlay(prev, trackers[i], cur, f"Cam{i} SN:{sns[i-1]}")
                cv2.imshow(f"Cam {i}", prev)
            except Exception as e:
                print(f"Fehler Cam{i}: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        ts = time.strftime("%Y%m%d_%H%M%S")

        def save(i, folder):
            if frames[i] is None:
                return
            cv2.imwrite(os.path.join(folder, f"cam{i}_{ts}.png"), frames[i])
            if currents[i] is not None:
                t = currents[i]["tilt_deg"]
                trackers[i].update(currents[i]["norm_pts"], t if t == t else 0.0)

        if key == ord('p'):
            save(1, "calib_single_1"); save(2, "calib_single_2"); save(3, "calib_single_3"); save(4, "calib_single_3")
            print(f"[OK] alle gespeichert ({ts})")
        elif key == ord('6'):
            save(1, "calib_single_1"); print("[OK] Cam1")
        elif key == ord('7'):
            save(2, "calib_single_2"); print("[OK] Cam2")
        elif key == ord('8'):
            save(3, "calib_single_3"); print("[OK] Cam3")
        elif key == ord('9'):
            save(4, "calib_single_4"); print("[OK] Cam4")
        elif key == ord('2'):
            if frames[1] is not None and frames[2] is not None:
                save(1, "calib_1_2"); save(2, "calib_1_2"); print("[OK] Paar 1+2")
        elif key == ord('3'):
            if frames[1] is not None and frames[3] is not None:
                save(1, "calib_1_3"); save(3, "calib_1_3"); print("[OK] Paar 1+3")
        elif key == ord('4'):
            if frames[1] is not None and frames[4] is not None:
                save(1, "calib_1_4"); save(4, "calib_1_4"); print("[OK] Paar 1+4")

    for c, good in zip(cams, ok):
        if good:
            c.stop_acquisition(); c.close_device()
    cv2.destroyAllWindows()
    print("\nEnd-Coverage:", {i: f"{trackers[i].fraction():0.0%}" for i in (1, 2, 3, 4)})


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        live(seed="--noseed" not in sys.argv)
