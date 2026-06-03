"""
E_bundle_adjust.py
==================
Globales Bundle Adjustment ueber alle drei Kameras.

Warum
-----
Paarweise Stereokalibrierung (1-2 und 1-3 getrennt) minimiert den
Reprojektionsfehler JEDES PAARES isoliert. Fuer ein Mehrkamerasystem ist
das suboptimal: die beiden Loesungen "kennen" einander nicht und ein
gemeinsames, global konsistentes Weltkoordinatensystem entsteht nur durch
Verkettung ueber Cam1 -- Fehler akkumulieren.

Das Bundle Adjustment (BA) minimiert stattdessen den GESAMT-Reprojektions-
fehler ueber ALLE Kameras und ALLE Aufnahmen GLEICHZEITIG. Es ist der
Goldstandard der Photogrammetrie/SfM.

Modell
------
* Cam1 = Weltursprung, fest [I|0].
* Optimierte Groessen:
    - Relative Posen Cam1->Cam2 und Cam1->Cam3 (je 6 DOF).
    - Pose des Boards pro Aufnahme im Cam1-Frame (6 DOF je Shot).
* Intrinsics (K, D) bleiben fix auf den sauberen Mono-Werten
  (numerisch stabil; Freigabe optional via REFINE_INTRINSICS).
* Cam1 ist in JEDER Aufnahme sichtbar und verankert damit das System;
  Cam2 und Cam3 sind ueber die gemeinsamen Board-Posen mit Cam1 (und so
  indirekt miteinander) gekoppelt.

Ehrliche Grenze: Da keine Aufnahme das Board gleichzeitig in Cam2 UND Cam3
zeigt, bleibt die relative Pose Cam2<->Cam3 eine VERKETTUNG ueber Cam1.
Das BA macht diese Verkettung konsistent, ersetzt aber keine echten
2-3-Beobachtungen. Fuer ein Praezisions-MoCap dringend solche Aufnahmen
nachholen (am besten Frames mit allen drei Kameras gleichzeitig).

Aufruf:  python E_bundle_adjust.py
"""

import os
import glob
import pickle
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import calib_common as cc

REFINE_INTRINSICS = False   # konservativ: erst Extrinsics+Posen global loesen
MIN_CORNERS = 8


def load(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def build_shots(board, detector):
    """
    Sammelt alle Aufnahmen. Jeder Shot: Liste von (cam, obj, img)-Tripeln
    plus solvePnP-Startpose aus Cam1.
    Folder-Zuordnung: calib_1_2 -> {1,2}, calib_1_3 -> {1,3}.
    """
    mono = {c: load(f"mono_cam{c}.pkl") if os.path.exists(f"mono_cam{c}.pkl")
            else load(f"mono_cam{c}_pinhole.pkl") for c in (1, 2, 3)}
    K = {c: mono[c]["K"] for c in (1, 2, 3)}
    D = {c: mono[c]["D"] for c in (1, 2, 3)}

    shots = []
    for folder, cams in [("calib_1_2", (1, 2)), ("calib_1_3", (1, 3))]:
        ref = 1
        ref_files = sorted(glob.glob(os.path.join(folder, f"cam{ref}_*.png")))
        for rf in ref_files:
            ts = os.path.basename(rf).replace(f"cam{ref}_", "").replace(".png", "")
            obs = {}
            for c in cams:
                p = os.path.join(folder, f"cam{c}_{ts}.png")
                if not os.path.exists(p):
                    continue
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                det = cc.detect_charuco(g, detector, board, min_corners=MIN_CORNERS)
                if det is None:
                    continue
                obs[c] = (det["obj_points"].reshape(-1, 3).astype(np.float64),
                          det["img_points"].reshape(-1, 2).astype(np.float64))
            # nur verwertbar, wenn Cam1 + mind. eine weitere Kamera sehen
            if 1 in obs and len(obs) >= 2:
                o1, i1 = obs[1]
                ok, rvec, tvec = cv2.solvePnP(
                    o1.reshape(-1, 1, 3), i1.reshape(-1, 1, 2),
                    K[1], D[1], flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                shots.append(dict(ts=ts, folder=folder, obs=obs,
                                  rvec=rvec.ravel(), tvec=tvec.ravel()))
    return shots, K, D


def pack(ext, shots):
    p = list(ext)  # 12 werte: rvec12,tvec12,rvec13,tvec13
    for s in shots:
        p += list(s["rvec"]) + list(s["tvec"])
    return np.array(p, dtype=np.float64)


def residuals(params, shots, K, D):
    rvec12 = params[0:3]; tvec12 = params[3:6]
    rvec13 = params[6:9]; tvec13 = params[9:12]
    ext = {2: (rvec12, tvec12), 3: (rvec13, tvec13)}
    res = []
    base = 12
    for k, s in enumerate(shots):
        rvec_b = params[base + 6*k: base + 6*k + 3]
        tvec_b = params[base + 6*k + 3: base + 6*k + 6]
        for c, (obj, img) in s["obs"].items():
            if c == 1:
                rv, tv = rvec_b, tvec_b
            else:
                rv, tv, *_ = cv2.composeRT(rvec_b, tvec_b, ext[c][0], ext[c][1])
                rv = rv.ravel(); tv = tv.ravel()
            proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K[c], D[c])
            res.append((proj.reshape(-1, 2) - img).ravel())
    return np.concatenate(res)


def sparsity(shots):
    # Residuen-Zaehlung
    rows = 0
    counts = []
    for s in shots:
        n = sum(2 * len(img) for _, (_, img) in s["obs"].items())
        counts.append((rows, n, s))
        rows += n
    ncols = 12 + 6 * len(shots)
    M = lil_matrix((rows, ncols), dtype=int)
    base = 12
    for k, (r0, n, s) in enumerate(counts):
        r = r0
        for c, (_, img) in s["obs"].items():
            m = 2 * len(img)
            # Board-Pose dieses Shots
            M[r:r+m, base + 6*k: base + 6*k + 6] = 1
            # Extrinsics der jeweiligen Kamera
            if c == 2:
                M[r:r+m, 0:6] = 1
            elif c == 3:
                M[r:r+m, 6:12] = 1
            r += m
    return M


def decompose(R, label):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return ang


def main():
    board = cc.make_board()
    detector = cc.make_detector(board)

    print("Sammle Aufnahmen ...")
    shots, K, D = build_shots(board, detector)
    n2 = sum(1 for s in shots if 2 in s["obs"])
    n3 = sum(1 for s in shots if 3 in s["obs"])
    print(f"  {len(shots)} Shots (mit Cam2: {n2}, mit Cam3: {n3})")
    if len(shots) < 5:
        print("Zu wenige Shots fuer BA."); return

    # Startwerte aus Stereo
    s12 = load("stereo_cam1_cam2.pkl")
    s13 = load("stereo_cam1_cam3.pkl")
    rvec12 = cv2.Rodrigues(s12["R"])[0].ravel()
    rvec13 = cv2.Rodrigues(s13["R"])[0].ravel()
    ext0 = list(rvec12) + list(s12["T"].ravel()) + list(rvec13) + list(s13["T"].ravel())
    p0 = pack(ext0, shots)

    r0 = residuals(p0, shots, K, D)
    rms0 = np.sqrt(np.mean(r0 ** 2))
    print(f"\nStart-Reprojektion (global) RMS: {rms0:.4f} px  "
          f"({len(r0)//2} Punktbeobachtungen)")

    J = sparsity(shots)
    print("Optimiere (Levenberg-Marquardt, sparse) ...")
    sol = least_squares(
        residuals, p0, jac_sparsity=J, method="trf",
        x_scale="jac", loss="huber", f_scale=1.0,
        args=(shots, K, D), verbose=2, max_nfev=80,
    )
    r1 = residuals(sol.x, shots, K, D)
    rms1 = np.sqrt(np.mean(r1 ** 2))
    print(f"\nEnd-Reprojektion (global) RMS:   {rms1:.4f} px  "
          f"(Start {rms0:.4f})")

    # Ergebnis-Extrinsics
    R12 = cv2.Rodrigues(sol.x[0:3])[0]; T12 = sol.x[3:6].reshape(3, 1)
    R13 = cv2.Rodrigues(sol.x[6:9])[0]; T13 = sol.x[9:12].reshape(3, 1)
    R23 = R13 @ R12.T
    T23 = T13 - R23 @ T12

    print("\n=== Global optimierte Posen (Cam1 = Ursprung) ===")
    print(f"Cam1->Cam2: Baseline {np.linalg.norm(T12):.4f} m  "
          f"Winkel {decompose(R12,'12'):.2f}°")
    print(f"Cam1->Cam3: Baseline {np.linalg.norm(T13):.4f} m  "
          f"Winkel {decompose(R13,'13'):.2f}°")
    print(f"Cam2->Cam3: Baseline {np.linalg.norm(T23):.4f} m  "
          f"Winkel {decompose(R23,'23'):.2f}°  (verkettet)")

    out = dict(
        K1=K[1], D1=D[1], K2=K[2], D2=D[2], K3=K[3], D3=D[3],
        R12=R12, T12=T12, R13=R13, T13=T13, R23=R23, T23=T23,
        rms_start=float(rms0), rms_end=float(rms1),
        num_shots=len(shots), image_size=tuple(load("mono_cam1.pkl")["image_size"])
        if os.path.exists("mono_cam1.pkl") else None,
    )
    with open("multicam_bundle.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\ngespeichert -> multicam_bundle.pkl")


if __name__ == "__main__":
    main()
