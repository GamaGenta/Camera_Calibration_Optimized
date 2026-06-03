"""
D_validate.py
=============
Validierung der Kalibrierung mit den fuer Motion-Capture relevanten
Metriken -- NICHT nur Reprojektionsfehler.

Tests
-----
1. TRIANGULATIONS-TEST (absolute Skala & Tiefengenauigkeit)
   Trianguliert die in beiden Kameras sichtbaren ChArUco-Ecken und
   vergleicht die REKONSTRUIERTEN Abstaende mit der bekannten Board-
   Geometrie (SQUARE_LENGTH-Raster). Liefert:
     * Skalenfehler (%)            -> metrische Korrektheit
     * 3D-Residuum (mm)            -> Triangulationsgenauigkeit
     * rekonstruierte Tiefe Z (m)  -> Arbeitsbereich
   Das ist der direkte Indikator dafuer, wie gut spaeter eine Person im
   Raum lokalisiert werden kann.

2. MULTI-CAM-KONSISTENZ (A->B->C)
   Berueckt die relative Pose Cam2->Cam3 aus der Verkettung
   (Cam1->Cam2)^-1 ... bzw. ueber Cam1 als Hub. Prueft, ob der
   Kalibriergraph in sich konsistent ist. HINWEIS: Mit den aktuell
   aufgenommenen Daten existiert KEINE direkte 2-3-Beobachtung
   (siehe Analyse), daher kann nur die Verkettung dargestellt, aber
   nicht gegen eine Direktmessung validiert werden.

Aufruf:  python D_validate.py
"""

import os
import glob
import pickle
import numpy as np
import cv2

import calib_common as cc


def load(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def load_stereo(ref, other):
    return load(f"stereo_cam{ref}_cam{other}.pkl")


# ----------------------------------------------------------------------
# Triangulationstest
# ----------------------------------------------------------------------
def triangulate_pair(stereo, folder, ref, other, board, detector):
    K1, D1 = stereo["K1"], stereo["D1"]
    K2, D2 = stereo["K2"], stereo["D2"]
    R, T = stereo["R"], stereo["T"]

    # Projektionsmatrizen (Cam1 = Welt-Ursprung)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, T])

    ref_files = sorted(glob.glob(os.path.join(folder, f"cam{ref}_*.png")))
    scale_errors, resid_mm, depths = [], [], []
    n_frames = 0

    for rf in ref_files:
        ts = os.path.basename(rf).replace(f"cam{ref}_", "").replace(".png", "")
        of = os.path.join(folder, f"cam{other}_{ts}.png")
        if not os.path.exists(of):
            continue
        gr = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        go = cv2.imread(of, cv2.IMREAD_GRAYSCALE)
        dr = cc.detect_charuco(gr, detector, board, min_corners=8)
        do = cc.detect_charuco(go, detector, board, min_corners=8)
        if dr is None or do is None:
            continue
        common = np.intersect1d(dr["ids_flat"], do["ids_flat"])
        if len(common) < 8:
            continue
        mr = np.isin(dr["ids_flat"], common)
        mo = np.isin(do["ids_flat"], common)
        sr = np.argsort(dr["ids_flat"][mr])
        so = np.argsort(do["ids_flat"][mo])
        obj = dr["obj_points"].reshape(-1, 3)[mr][sr]      # bekannte Board-3D
        p1 = dr["img_points"].reshape(-1, 2)[mr][sr]
        p2 = do["img_points"].reshape(-1, 2)[mo][so]

        # Entzerren ist in triangulatePoints NICHT enthalten -> wir nutzen
        # die vollen K|D ueber undistortPoints und P=K[I|0]/K[R|T].
        u1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)
        u2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), K2, D2, P=K2).reshape(-1, 2)

        X = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
        X = (X[:3] / X[3]).T   # (N,3) in Cam1-Koordinaten

        depths.extend(X[:, 2].tolist())

        # Skala & Form gegen bekannte Geometrie: Vergleiche alle paarweisen
        # Abstaende (skaleninvariant gegen Translation/Rotation).
        n = len(X)
        if n < 2:
            continue
        idx = np.triu_indices(n, k=1)
        d_rec = np.linalg.norm(X[idx[0]] - X[idx[1]], axis=1)
        d_true = np.linalg.norm(obj[idx[0]] - obj[idx[1]], axis=1)
        valid = d_true > 1e-6
        ratio = d_rec[valid] / d_true[valid]
        scale_errors.append(np.median(ratio))

        # Starr ausrichten (Umeyama ohne Skala) -> echtes 3D-Residuum
        resid = rigid_residual(X, obj)
        resid_mm.append(resid * 1000.0)
        n_frames += 1

    return dict(
        n_frames=n_frames,
        scale=np.array(scale_errors),
        resid_mm=np.array(resid_mm),
        depth=np.array(depths),
    )


def rigid_residual(X, Y):
    """RMS-Abstand nach optimaler starrer Ausrichtung (Kabsch) X->Y. Meter."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dg = np.diag([1, 1, d])
    Rr = Vt.T @ Dg @ U.T
    Xa = (Rr @ Xc.T).T
    return float(np.sqrt(np.mean(np.sum((Xa - Yc) ** 2, axis=1))))


def report_triangulation(name, res):
    if res["n_frames"] == 0:
        print(f"\n[{name}] Keine triangulierbaren Frames.")
        return
    sc = res["scale"]
    print(f"\n[{name}] Triangulationstest ({res['n_frames']} Frames)")
    print(f"   Skalenfaktor (rec/true):  {sc.mean():.4f} "
          f"-> Skalenfehler {abs(sc.mean()-1)*100:.2f} %")
    print(f"   3D-Residuum (starr):      mean={res['resid_mm'].mean():.2f} mm "
          f"median={np.median(res['resid_mm']):.2f} mm "
          f"max={res['resid_mm'].max():.2f} mm")
    print(f"   Rekonstruierte Tiefe Z:   {res['depth'].mean():.2f} m "
          f"[{res['depth'].min():.2f}, {res['depth'].max():.2f}]")
    if abs(sc.mean() - 1) > 0.02:
        print("   ⚠️  >2% Skalenfehler -> SQUARE_LENGTH/MARKER_LENGTH am Druck "
              "nachmessen!")


# ----------------------------------------------------------------------
# Multi-Cam-Konsistenz
# ----------------------------------------------------------------------
def multicam_consistency():
    print("\n" + "=" * 64)
    print("MULTI-CAM-KONSISTENZ (Cam1 als Referenz/Hub)")
    print("=" * 64)
    try:
        s12 = load_stereo(1, 2)
        s13 = load_stereo(1, 3)
    except FileNotFoundError:
        print("Stereo-Dateien fehlen -- zuerst C_stereo_all.py ausfuehren.")
        return

    # Posen in Cam1-Frame:  X_camK = R_1K X_cam1 + T_1K
    R12, T12 = s12["R"], s12["T"]
    R13, T13 = s13["R"], s13["T"]
    # Verkettung Cam2 -> Cam3 :  X_c3 = R13 (R12^T (X_c2 - T12)) + T13
    R23 = R13 @ R12.T
    T23 = T13 - R23 @ T12
    base23 = float(np.linalg.norm(T23))
    ang23 = float(np.degrees(np.arccos(np.clip((np.trace(R23) - 1) / 2, -1, 1))))

    print(f"Baseline Cam1-Cam2: {s12['baseline']:.3f} m  "
          f"(Winkel {s12['rel_angle_deg']:.1f}°)")
    print(f"Baseline Cam1-Cam3: {s13['baseline']:.3f} m  "
          f"(Winkel {s13['rel_angle_deg']:.1f}°)")
    print(f"-> abgeleitet Cam2-Cam3: {base23:.3f} m  (Winkel {ang23:.1f}°)")
    print("\nHINWEIS: Es existiert keine direkte 2-3-Aufnahme. Daher kann der")
    print("A->B->C- gegen A->C-Vergleich NICHT validiert werden. Fuer einen")
    print("echten Konsistenztest werden Aufnahmen benoetigt, in denen das")
    print("Board gleichzeitig von Cam2 UND Cam3 (idealerweise allen dreien)")
    print("gesehen wird -> globale Bundle Adjustment-Stufe (E_bundle_adjust.py).")


def main():
    board = cc.make_board()
    detector = cc.make_detector(board)

    print("=" * 64)
    print("TRIANGULATIONSTEST")
    print("=" * 64)
    for ref, other, folder in [(1, 2, "calib_1_2"), (1, 3, "calib_1_3")]:
        sname = f"stereo_cam{ref}_cam{other}.pkl"
        if not os.path.exists(sname):
            print(f"[{ref}-{other}] {sname} fehlt -- C_stereo_all.py ausfuehren.")
            continue
        stereo = load_stereo(ref, other)
        res = triangulate_pair(stereo, folder, ref, other, board, detector)
        report_triangulation(f"Cam{ref}-Cam{other}", res)

    multicam_consistency()


if __name__ == "__main__":
    main()
