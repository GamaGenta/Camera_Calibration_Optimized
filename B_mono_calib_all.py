"""
B_mono_calib_all.py
===================
Wissenschaftlich rigorose Monokalibrierung fuer alle drei Kameras.

Ersetzt B_Cam1/2/3_Calibration.py. Verbesserungen:

  1. Getunte Detektion (calib_common) -> ~3x mehr nutzbare Ansichten.
  2. Harte Mindest-View-Zahl + Coverage-Pruefung MIT Warnung statt
     stillem Verwerfen. (Alt: Cam2 mit 3 Views, unbemerkt.)
  3. matchImagePoints statt manuellem Index (geometrisch korrekt, robust
     gegen Teil-Detektionen).
  4. Iterative robuste Re-Kalibrierung: Views mit hohem Reprojektions-
     fehler werden entfernt und neu kalibriert (Standard in der Photo-
     grammetrie).
  5. Distortion-Modellvergleich (5-param vs. rational) mit Entscheidung
     anhand des Informationsgewinns -- NICHT blind mehr Parameter.
  6. Aggregierte Coverage-Heatmap als Qualitaetsindikator.
  7. Speichert zusaetzlich die *Roh-Detektionen* (charuco corners+ids pro
     Bild) -> noetig fuer die spaetere globale Bundle-Adjustment-Stufe.

Aufruf:  python B_mono_calib_all.py
"""

import os
import glob
import pickle
import numpy as np
import cv2
from tqdm import tqdm

import calib_common as cc

# Fuer jede Kamera: Ordner mit Bildern (Dateinamen camN_*.png)
CAMERA_SOURCES = {
    1: ["calib_1_2", "calib_1_3", "calib_single_1"],
    2: ["calib_1_2", "calib_single_2"],
    3: ["calib_1_3", "calib_single_3"],
}

MIN_VIEWS = 12          # unter diesem Wert ist Intrinsics-Schaetzung unsicher
MIN_CORNERS = 8         # pro Bild
MIN_COVERAGE_CELLS = 0.55  # mind. 55% der Bildregionen muessen abgedeckt sein
REPROJ_OUTLIER_SIGMA = 3.0  # Views > mean+3*std werden entfernt


def load_files(cam):
    files = []
    for folder in CAMERA_SOURCES[cam]:
        if not os.path.isdir(folder):
            continue
        files += glob.glob(os.path.join(folder, f"cam{cam}_*.png"))
        files += glob.glob(os.path.join(folder, f"cam{cam}_*.jpg"))
    return sorted(set(files))


def detect_all(cam, board, detector):
    files = load_files(cam)
    views = []
    image_size = None
    print(f"\n[CAM{cam}] Detektiere ChArUco in {len(files)} Bildern ...")
    for path in tqdm(files):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        image_size = (g.shape[1], g.shape[0])
        det = cc.detect_charuco(g, detector, board, min_corners=MIN_CORNERS)
        if det is None:
            continue
        det["path"] = path
        det["sharpness"] = cc.calculate_sharpness(g)
        views.append(det)
    print(f"[CAM{cam}] {len(views)}/{len(files)} Bilder nutzbar "
          f"({100*len(views)/max(len(files),1):.0f}%).")
    return views, image_size


def robust_calibrate(views, image_size, flags, label):
    """Kalibriert, entfernt iterativ Reprojektions-Ausreisser, kalibriert neu."""
    obj = [v["obj_points"] for v in views]
    img = [v["img_points"] for v in views]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-9)

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, None, None, flags=flags, criteria=crit
    )
    for it in range(3):
        errs = cc.per_view_reprojection_errors(obj, img, rvecs, tvecs, K, D)
        thr = errs.mean() + REPROJ_OUTLIER_SIGMA * errs.std()
        keep = errs <= thr
        if keep.all() or keep.sum() < MIN_VIEWS:
            break
        n_removed = (~keep).sum()
        obj = [o for o, k in zip(obj, keep) if k]
        img = [i for i, k in zip(img, keep) if k]
        views = [v for v, k in zip(views, keep) if k]
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj, img, image_size, None, None, flags=flags, criteria=crit
        )
        print(f"   [{label}] Iter {it+1}: {n_removed} Ausreisser entfernt "
              f"(thr={thr:.3f}px) -> RMS={rms:.4f}px, {len(obj)} Views")

    errs = cc.per_view_reprojection_errors(obj, img, rvecs, tvecs, K, D)
    return dict(rms=rms, K=K, D=D, rvecs=rvecs, tvecs=tvecs,
                views=views, per_view=errs)


def calibrate_camera(cam, board, detector):
    views, image_size = detect_all(cam, board, detector)

    if len(views) < MIN_VIEWS:
        print(f"[CAM{cam}] ⚠️  NUR {len(views)} Views (<{MIN_VIEWS})! "
              f"Intrinsics werden UNZUVERLAESSIG sein. Mehr Bilder noetig.")

    cov = cc.coverage_heatmap_score([v["img_points"] for v in views], image_size)
    print(f"[CAM{cam}] Aggregierte Bildabdeckung: {cov:.0%} der Regionen")
    if cov < MIN_COVERAGE_CELLS:
        print(f"[CAM{cam}] ⚠️  Geringe Randabdeckung -> Distortion/Hauptpunkt "
              f"schlecht bestimmbar. Board naeher an Bildraender/-ecken fuehren.")

    # --- Modellvergleich: 5-Parameter vs. Rational ---
    base_flags = (cv2.CALIB_FIX_ASPECT_RATIO * 0)  # placeholder, keine Fixierung
    r5 = robust_calibrate(list(views), image_size, 0, "5-param")
    rR = robust_calibrate(list(views), image_size, cv2.CALIB_RATIONAL_MODEL, "rational")

    # Entscheidung: rational nur, wenn es den RMS *deutlich* (>5%) senkt
    # UND genug Views vorhanden sind, um 8 Distortionsparameter zu stuetzen.
    use_rational = (len(rR["views"]) >= 20) and (rR["rms"] < 0.95 * r5["rms"])
    chosen = rR if use_rational else r5
    model = "rational(8)" if use_rational else "pinhole(5)"

    K, D = chosen["K"], chosen["D"]
    print(f"\n[CAM{cam}] === ERGEBNIS ===")
    print(f"   Modell:   {model}")
    print(f"   Views:    {len(chosen['views'])}")
    print(f"   RMS:      {chosen['rms']:.4f} px  "
          f"(per-view: min={chosen['per_view'].min():.3f} "
          f"max={chosen['per_view'].max():.3f})")
    print(f"   fx,fy:    {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"   cx,cy:    {K[0,2]:.2f}, {K[1,2]:.2f}  "
          f"(Bildmitte: {image_size[0]/2:.0f}, {image_size[1]/2:.0f})")
    print(f"   D:        {np.round(D.ravel(), 5)}")

    # Schlanke Roh-Detektionen fuer Bundle Adjustment (nur Pfad + ids + corners)
    raw = [dict(path=v["path"],
                ids=v["ids_flat"],
                charuco=v["charuco_corners"].reshape(-1, 2).astype(np.float64))
           for v in chosen["views"]]

    data = dict(
        K=K, D=D, model=model,
        rms=float(chosen["rms"]),
        per_view_errors=chosen["per_view"],
        image_size=image_size,
        num_views=len(chosen["views"]),
        coverage=cov,
        raw_detections=raw,
    )
    out = f"mono_cam{cam}.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"   gespeichert -> {out}")
    return data


def main():
    board = cc.make_board()
    detector = cc.make_detector(board)
    summary = {}
    for cam in (1, 2, 3):
        summary[cam] = calibrate_camera(cam, board, detector)

    print("\n" + "=" * 64)
    print("ZUSAMMENFASSUNG MONOKALIBRIERUNG")
    print("=" * 64)
    print(f"{'Cam':<5}{'Modell':<14}{'Views':<7}{'RMS[px]':<10}"
          f"{'Cov':<7}{'fx':<9}{'cy':<9}")
    for cam, d in summary.items():
        print(f"{cam:<5}{d['model']:<14}{d['num_views']:<7}{d['rms']:<10.4f}"
              f"{d['coverage']:<7.0%}{d['K'][0,0]:<9.1f}{d['K'][1,2]:<9.1f}")
    print("\nHinweis: Niedriger RMS allein ist KEIN Qualitaetsbeweis. Auf")
    print("ausreichende View-Zahl (>=12) und Coverage (>=55%) achten.")


if __name__ == "__main__":
    main()
