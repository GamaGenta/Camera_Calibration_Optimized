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

import csv
import math
from decimal import Decimal, InvalidOperation
import pandas as pd
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

history = {'cost': [], 'rms': []}


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
            else load(f"mono_cam{c}_pinhole.pkl") for c in (1, 2, 3, 4)}
    K = {c: mono[c]["K"] for c in (1, 2, 3, 4)}
    D = {c: mono[c]["D"] for c in (1, 2, 3, 4)}

    shots = []
    for folder, cams in [("calib_1_2", (1, 2)), ("calib_1_3", (1, 3)), ("calib_1_4", (1, 4))]:
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


def residuals_logged(x, shots, K, D):
    f = residuals(x, shots, K, D)            # Ihre bestehende Funktion (Residuen)
    val = 0.5 * np.sum(f**2)               # 0.5*(0.25+0.09+0.01) = 0.175
    history['cost'].append(val)            # speichert 0.175
    # history['cost'].append(0.5 * np.sum(f**2))
    history['rms'].append(np.sqrt(np.mean(f**2)))
    return f

def residuals(params, shots, K, D):
    rvec12 = params[0:3]; tvec12 = params[3:6]
    rvec13 = params[6:9]; tvec13 = params[9:12]
    rvec14 = params[12:15]; tvec14 = params[15:18]
    ext = {2: (rvec12, tvec12), 3: (rvec13, tvec13), 4: (rvec14, tvec14)}
    res = []
    base = 18  # es gibt 18 Extrinsic-Parameter: (r,t) fuer Cam2,Cam3,Cam4 (3+3 each). Von 12 auf 18 gehändert 
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
    # 18 Extrinsic-Parameter (3+3 fuer jede der Cam2, Cam3, Cam4) plus 6 pro Shot
    ncols = 18 + 6 * len(shots)
    M = lil_matrix((rows, ncols), dtype=int)
    base = 18 #<---------------------------------------------------------------------------------------- 12->18
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
            elif c == 4:
                M[r:r+m, 12:18] = 1
            r += m
    return M


def decompose(R, label):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return ang

'''
def save_r_csv(r1, filename='r1.csv', sep=';'):
    r1 = np.asarray(r1).ravel()
    if r1.size % 2 == 0:
        pts = r1.reshape(-1, 2)
        df = pd.DataFrame(pts, columns=['dx', 'dy'])
        df['abs_dx'] = df['dx'].abs() #abs = absolute value
        df['abs_dy'] = df['dy'].abs()
        df['norm_px'] = np.sqrt((df['dx']**2 + df['dy']**2))
    else:
        df = pd.DataFrame(r1, columns=['residual'])
        df['abs_residual'] = df['residual'].abs()
    df.index.name = 'obs_index'
    df.to_csv(filename, sep=sep, index=True, float_format='%.6f')
    return os.path.abspath(filename)
'''

def save_r_csv(r1, filename='r1.csv', sep=';'):
    r1 = np.asarray(r1).ravel()
    rows = []
    if r1.size % 2 == 0:
        pts = r1.reshape(-1, 2)
        for dx, dy in pts:
            norm = math.hypot(dx, dy)
            rows.append([dx, dy, abs(dx), abs(dy), norm])
        header = ['dx', 'dy', 'abs_dx', 'abs_dy', 'norm_px']
    else:
        for v in r1:
            rows.append([v, abs(v)])
        header = ['residual', 'abs_residual']

    def fmt_num(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ''
        try:
            d = Decimal(str(v))            # Decimal aus String, vermeidet direkte Float-Effekte
            s = format(d, 'f')             # keine Exponentialschreibweise
        except (InvalidOperation, ValueError):
            s = str(v)
        return s.replace('.', ',')        # nur Punkt -> Komma ersetzen

    with open(filename, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout, delimiter=sep, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt_num(x) for x in row])

    return os.path.abspath(filename)

def check_residual_order(r0, r1, k=100, tol=1e-9, save_csv=None):
    """
    Prüft, ob r0 und r1 dieselbe Reihenfolge haben und liefert Diagnostics.
    - r0, r1: array-like (1D oder flachbar)
    - k: Top-k für Vergleich der stärksten Residuen
    - tol: Abweichungstoleranz für Gleichheitscheck (absolute)
    - save_csv: optionaler Pfad, um alle Paare index;r0;r1;diff zu speichern
    Returns: dict mit Schlüssel:
      equal_shape (bool), equal_order (bool), num_mismatches (int),
      mismatches (list der ersten 20 (index, r0, r1, diff)),
      topk, topk_match_rate (float)
    """
    r0 = np.asarray(r0).ravel()
    r1 = np.asarray(r1).ravel()

    if r0.shape != r1.shape:
        return {
            "equal_shape": False,
            "msg": f"different lengths: r0={r0.size}, r1={r1.size}"
        }

    # genauer Wertevergleich in Reihenfolge
    equal_mask = np.isclose(r0, r1, atol=tol, rtol=0.0)
    equal_order = bool(equal_mask.all())
    diff_idx = np.nonzero(~equal_mask)[0]

    # Top-k Übereinstimmung der Indizes der größten Absolutwerte
    k = min(int(k), r0.size)
    idx0 = np.argsort(-np.abs(r0))[:k]
    idx1 = np.argsort(-np.abs(r1))[:k]
    topk_match_rate = np.intersect1d(idx0, idx1).size / k if k > 0 else 1.0

    # erste N Mismatches sammeln
    mismatches = []
    for i in diff_idx[:20]:
        mismatches.append((int(i), float(r0[i]), float(r1[i]), float(r0[i] - r1[i])))

    # optional CSV schreiben (Semikolon, deutsches Dezimalkomma)
    if save_csv:
        with open(save_csv, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.writer(fout, delimiter=';')
            writer.writerow(['index', 'r0', 'r1', 'diff'])
            for i in range(r0.size):
                def fmt(x, f="{:.6f}"):
                    return f.format(float(x)).replace('.', ',')
                writer.writerow([i, fmt(r0[i]), fmt(r1[i]), fmt(r0[i] - r1[i])])

    return {
        "equal_shape": True,
        "equal_order": equal_order,
        "num_mismatches": int(diff_idx.size),
        "mismatches": mismatches,
        "topk": k,
        "topk_match_rate": float(topk_match_rate)
    }

def main():
    board = cc.make_board()
    detector = cc.make_detector(board)

    print("Sammle Aufnahmen ...")
    shots, K, D = build_shots(board, detector)
    n2 = sum(1 for s in shots if 2 in s["obs"])
    n3 = sum(1 for s in shots if 3 in s["obs"])
    n4 = sum(1 for s in shots if 4 in s["obs"]) #n4
    print(f"  {len(shots)} Shots (mit Cam2: {n2}, mit Cam3: {n3}, mit Cam4: {n4})")
    if len(shots) < 5:
        print("Zu wenige Shots fuer BA."); return

    # Startwerte aus Stereo
    s12 = load("stereo_cam1_cam2.pkl")
    s13 = load("stereo_cam1_cam3.pkl")
    s14 = load("stereo_cam1_cam4.pkl")
    rvec12 = cv2.Rodrigues(s12["R"])[0].ravel()
    rvec13 = cv2.Rodrigues(s13["R"])[0].ravel()
    rvec14 = cv2.Rodrigues(s14["R"])[0].ravel()
    ext0 = list(rvec12) + list(s12["T"].ravel()) + list(rvec13) + list(s13["T"].ravel()) + list(rvec14) + list(s14["T"].ravel())
    p0 = pack(ext0, shots)

    r0 = residuals(p0, shots, K, D)
    rms0 = np.sqrt(np.mean(r0 ** 2))
    print(f"\nStart-Reprojektion (global) RMS: {rms0:.4f} px  "
          f"({len(r0)//2} Punktbeobachtungen)")

    J = sparsity(shots)
    print("Optimiere (Levenberg-Marquardt, sparse) ...")
    sol = least_squares(
        residuals_logged, p0, jac_sparsity=J, method="trf",
        x_scale="jac", loss="huber", f_scale=1.0,
        args=(shots, K, D), verbose=2, max_nfev=80,
        #args=(shots, K, D), verbose=2, max_nfev=20000,
    )

    r1 = residuals(sol.x, shots, K, D)
    #test export residuals 
    print(f'r1: {r1}') #resiuals

    r1Array = np.asarray(r1).ravel()              # (2N,)
    print(f'r1Array (2N,): {r1Array}')                          
    squared = r1**2                     # quadrierte Residuen pro Komponente
    print(f'quadrierte Residuen pro Komponente/Beobachtung: {squared}')
    rms = np.sqrt(np.mean(squared))     # RMS
    print(f'RMS: {rms}')
    # print(f'history: {history}')

    #export r0 and r1 Array to csv
    print(save_r_csv(r0, 'r0_pixels_4-cams_global.csv'))
    print(save_r_csv(r1, 'r1_pixels_4-cams_global.csv'))
    #exprot history to csv
    pd.DataFrame(history).to_csv('ls_history.csv', sep=';', index=False, float_format='%.6f')

    # Beispielnutzung nach Berechnung von r0 und r1:
    result = check_residual_order(r0, r1, k=100, tol=1e-9, save_csv='resid_compare.csv')
    print(result)

    #root mean square calculation 
    rms1 = np.sqrt(np.mean(r1 ** 2))
    print(f"\nEnd-Reprojektion (global) RMS:   {rms1:.4f} px  "
          f"(Start {rms0:.4f})")

    # Ergebnis-Extrinsics
    R12 = cv2.Rodrigues(sol.x[0:3])[0]; T12 = sol.x[3:6].reshape(3, 1)
    R13 = cv2.Rodrigues(sol.x[6:9])[0]; T13 = sol.x[9:12].reshape(3, 1)
    R14 = cv2.Rodrigues(sol.x[12:15])[0]; T14 = sol.x[15:18].reshape(3, 1)
    R23 = R13 @ R12.T
    T23 = T13 - R23 @ T12
    R34 = R14 @ R13.T
    T34 = T14 - R34 @ T13
    R24 = R14 @ R12.T
    T24 = T14 - R24 @ T12

    print("\n=== Global optimierte Posen (Cam1 = Ursprung) ===")
    print(f"Cam1->Cam2: Baseline {np.linalg.norm(T12):.4f} m  "
          f"Winkel {decompose(R12,'12'):.2f}°")
    print(f"Cam1->Cam3: Baseline {np.linalg.norm(T13):.4f} m  "
          f"Winkel {decompose(R13,'13'):.2f}°")
    print(f"Cam1->Cam4: Baseline {np.linalg.norm(T14):.4f} m  "
          f"Winkel {decompose(R14,'14'):.2f}°")
    print(f"Cam2->Cam3: Baseline {np.linalg.norm(T23):.4f} m  "
          f"Winkel {decompose(R23,'23'):.2f}°  (verkettet)")
    print(f"Cam2->Cam4: Baseline {np.linalg.norm(T24):.4f} m  "
          f"Winkel {decompose(R24,'24'):.2f}°  (verkettet)")
    print(f"Cam3->Cam4: Baseline {np.linalg.norm(T34):.4f} m  "
          f"Winkel {decompose(R34,'34'):.2f}°  (verkettet)")

    
    out = dict(
        K1=K[1], D1=D[1], K2=K[2], D2=D[2], K3=K[3], D3=D[3], K4=K[4], D4=D[4],
        R12=R12, T12=T12, R13=R13, T13=T13, R14=R14, T14=T14, R23=R23, T23=T23, R24=R24, T24=T24, R34=R34, T34=T34,
        rms_start=float(rms0), rms_end=float(rms1),
        num_shots=len(shots), image_size=tuple(load("mono_cam1.pkl")["image_size"])
        if os.path.exists("mono_cam1.pkl") else None,
    )
    '''
    out = dict(
        K1=K[1], D1=D[1], K2=K[2], D2=D[2], K3=K[3], D3=D[3], K4=K[4], D4=D[4],
        R12=R12, T12=T12, R13=R13, T13=T13, R14=R14, T14=T14, R23=R23, T23=T23, 
        rms_start=float(rms0), rms_end=float(rms1),
        num_shots=len(shots), image_size=tuple(load("mono_cam1.pkl")["image_size"])
        if os.path.exists("mono_cam1.pkl") else None,
    )
        '''

    with open("multicam_bundle.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\ngespeichert -> multicam_bundle.pkl")


if __name__ == "__main__":
    main()
