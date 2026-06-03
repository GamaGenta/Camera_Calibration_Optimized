"""
C_stereo_all.py
===============
Stereokalibrierung der Paare (1-2) und (1-3) mit Cam1 als Referenz.

Verbesserungen gegenueber C_Stereo_TwoCam1_2/1_3.py:

  1. Getunte Detektion (calib_common) auf BEIDEN Kameras -> deutlich mehr
     synchron sichtbare Boards.
  2. matchImagePoints + ID-Schnittmenge (geometrisch korrekte
     Korrespondenzen).
  3. CALIB_FIX_INTRINSIC mit den neuen, sauberen Mono-Intrinsics.
     (Optional: leichtes gemeinsames Nachoptimieren der Intrinsics.)
  4. Aussagekraeftige Fehlermetrik: SYMMETRISCHER EPIPOLARFEHLER statt nur
     des stereoCalibrate-RMS. Der Epipolarfehler ist die physikalisch
     relevante Groesse fuer die spaetere Triangulation.
  5. Speichert R, T (Cam1->CamX) plus die gemeinsamen Detektionen.

Aufruf:  python C_stereo_all.py
"""

import os
import glob
import pickle
import numpy as np
import cv2
from tqdm import tqdm

import calib_common as cc

PAIRS = [
    # (ref_cam, other_cam, ordner)
    (1, 2, "calib_1_2"),
    (1, 3, "calib_1_3"),
]


def load_mono(cam):
    # bevorzugt die neue Datei, faellt auf die alte zurueck
    for name in (f"mono_cam{cam}.pkl", f"mono_cam{cam}_pinhole.pkl"):
        if os.path.exists(name):
            with open(name, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Keine Monokalibrierung fuer Cam{cam} gefunden")


def find_pairs(folder, ref, other):
    ref_files = sorted(glob.glob(os.path.join(folder, f"cam{ref}_*.png")))
    pairs = []
    for rf in ref_files:
        ts = os.path.basename(rf).replace(f"cam{ref}_", "").replace(".png", "")
        of = os.path.join(folder, f"cam{other}_{ts}.png")
        if os.path.exists(of):
            pairs.append((rf, of))
    return pairs


def collect_correspondences(folder, ref, other, board, detector):
    pairs = find_pairs(folder, ref, other)
    obj_pts, ref_pts, oth_pts = [], [], []
    used = 0
    print(f"\n[Stereo {ref}-{other}] {len(pairs)} synchrone Paare in {folder}")
    for rf, of in tqdm(pairs):
        gr = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        go = cv2.imread(of, cv2.IMREAD_GRAYSCALE)
        if gr is None or go is None:
            continue
        dr = cc.detect_charuco(gr, detector, board, min_corners=6)
        do = cc.detect_charuco(go, detector, board, min_corners=6)
        if dr is None or do is None:
            continue
        common = np.intersect1d(dr["ids_flat"], do["ids_flat"])
        if len(common) < 6:
            continue
        # Punkte nach gemeinsamer ID sortiert ausrichten
        mr = np.isin(dr["ids_flat"], common)
        mo = np.isin(do["ids_flat"], common)
        sr = np.argsort(dr["ids_flat"][mr])
        so = np.argsort(do["ids_flat"][mo])
        ids = dr["ids_flat"][mr][sr]
        o = dr["obj_points"].reshape(-1, 3)[mr][sr]
        rp = dr["img_points"].reshape(-1, 2)[mr][sr]
        op = do["img_points"].reshape(-1, 2)[mo][so]
        obj_pts.append(o.reshape(-1, 1, 3).astype(np.float32))
        ref_pts.append(rp.reshape(-1, 1, 2).astype(np.float32))
        oth_pts.append(op.reshape(-1, 1, 2).astype(np.float32))
        used += 1
    print(f"[Stereo {ref}-{other}] {used} Paare mit >=6 gemeinsamen Punkten")
    return obj_pts, ref_pts, oth_pts


def symmetric_epipolar_error(ref_pts, oth_pts, F):
    """Mittlerer symmetrischer Epipolarabstand (Pixel) ueber alle Punkte."""
    errs = []
    for rp, op in zip(ref_pts, oth_pts):
        p1 = rp.reshape(-1, 2).astype(np.float64)
        p2 = op.reshape(-1, 2).astype(np.float64)
        p1h = np.hstack([p1, np.ones((len(p1), 1))])
        p2h = np.hstack([p2, np.ones((len(p2), 1))])
        l2 = (F @ p1h.T).T          # Epipolarlinien im Bild 2
        l1 = (F.T @ p2h.T).T        # Epipolarlinien im Bild 1
        d2 = np.abs(np.sum(l2 * p2h, axis=1)) / np.linalg.norm(l2[:, :2], axis=1)
        d1 = np.abs(np.sum(l1 * p1h, axis=1)) / np.linalg.norm(l1[:, :2], axis=1)
        errs.extend(0.5 * (d1 + d2))
    return float(np.mean(errs)), float(np.median(errs))


def calibrate_pair(ref, other, folder, board, detector):
    mref = load_mono(ref)
    moth = load_mono(other)
    image_size = tuple(mref["image_size"])

    obj, rp, op = collect_correspondences(folder, ref, other, board, detector)
    if len(obj) < 5:
        print(f"[Stereo {ref}-{other}] ⚠️  Zu wenige Paare ({len(obj)})!")
        return None

    K1, D1 = mref["K"], mref["D"]
    K2, D2 = moth["K"], moth["D"]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-9)

    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj, rp, op, K1, D1, K2, D2, image_size,
        criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC
    )

    mean_ep, med_ep = symmetric_epipolar_error(rp, op, F)
    baseline = float(np.linalg.norm(T))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    print(f"\n[Stereo {ref}-{other}] === ERGEBNIS ===")
    print(f"   stereoCalibrate RMS:      {rms:.4f} px")
    print(f"   symm. Epipolarfehler:     mean={mean_ep:.4f}  median={med_ep:.4f} px")
    print(f"   Baseline |T|:             {baseline:.4f} m")
    print(f"   relativer Drehwinkel:     {angle:.2f}°")
    print(f"   T (Cam{ref}->Cam{other}) [m]: {np.round(T.ravel(),4)}")

    data = dict(
        ref=ref, other=other,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        baseline=baseline, rel_angle_deg=angle,
        rms=float(rms), epipolar_mean=mean_ep, epipolar_median=med_ep,
        image_size=image_size, num_pairs=len(obj),
    )
    out = f"stereo_cam{ref}_cam{other}.pkl"
    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"   gespeichert -> {out}")
    return data


def main():
    board = cc.make_board()
    detector = cc.make_detector(board)
    for ref, other, folder in PAIRS:
        calibrate_pair(ref, other, folder, board, detector)


if __name__ == "__main__":
    main()
