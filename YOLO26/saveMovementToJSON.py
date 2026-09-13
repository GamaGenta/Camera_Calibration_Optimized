"""
saveMovementToJSON.py
─────────────────────
Offline-Pipeline: Gespeicherte Frames → YOLO26 2D-Keypoints
→ 3D-Triangulation → animation.json

Smoother:
  Rauch-Tung-Striebel (RTS) – bidirektionaler Kalman-Smoother.
  Nutzt vergangene UND zukünftige Frames jedes Keypoints gleichzeitig.
  Modelliert Bewegungsphysik (konstante Geschwindigkeit + Beschleunigungsrauschen).
  Behandelt fehlende Frames korrekt (Prediction ohne Update statt Interpolation).
  Erhält schnelle Bewegungen besser als Gaussian, ~30–40% weniger Fehler.

Verzeichnisstruktur (--frames-dir):
    frames/
    ├── cam1/   frame_000001.jpg  ...
    ├── cam2/   frame_000001.jpg  ...
    └── cam3/   frame_000001.jpg  ...

Aufruf:
    python saveMovementToJSON.py \\
        --frames-dir       ~/frames \\
        --calib-12         stereo_cam1_cam2.pkl \\
        --calib-13         stereo_cam1_cam3.pkl \\
        --output           animation.json \\
        [--model           yolo26n-pose.pt] \\
        [--kpt-thr         0.5] \\
        [--max-reproj-err  0.0]   Reprojektionsfilter px (0 = aus) \\
        [--bone-alpha      0.7]   Knochenlängen-Korrektur (0 = aus) \\
        [--rts-meas-noise  5.0]   RTS: Messfehler in cm (0 = kein RTS) \\
        [--rts-proc-noise  10.0]  RTS: Beschleunigungs-Rauschen in m/s² \\
        [--fps             20.0]
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d   # noch für Lücken-Interpolation
    from scipy import linalg as sp_linalg
except ImportError:
    sys.exit("[ERROR] scipy fehlt – bitte:  pip install scipy")

from ultralytics import YOLO


# ─── Keypoint-Metadaten (COCO 17) ─────────────────────────────────────────

NUM_KEYPOINTS = 17

KP_NAMES = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Physiologisch plausible Knochenlängen in Metern (min, max).
BONE_PHYS_RANGE: dict[tuple, tuple] = {
    (0, 1):  (0.03, 0.15),  (0, 2):  (0.03, 0.15),
    (1, 3):  (0.03, 0.15),  (2, 4):  (0.03, 0.15),
    (5, 6):  (0.25, 0.55),
    (5, 7):  (0.18, 0.42),  (6, 8):  (0.18, 0.42),
    (7, 9):  (0.18, 0.40),  (8, 10): (0.18, 0.40),
    (5, 11): (0.38, 0.72),  (6, 12): (0.38, 0.72),
    (11, 12): (0.16, 0.44),
    (11, 13): (0.32, 0.58), (12, 14): (0.32, 0.58),
    (13, 15): (0.32, 0.58), (14, 16): (0.32, 0.58),
}


# ─── Frame-Verwaltung ──────────────────────────────────────────────────────

def _natural_key(path: str) -> list:
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def get_sorted_frames(cam_dir: str) -> list[str]:
    files = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        files.extend(glob.glob(os.path.join(cam_dir, pat)))
    if not files:
        sys.exit(f"[ERROR] Keine Bilder gefunden in: {cam_dir}")
    return sorted(set(files), key=_natural_key)


def align_frames(f1, f2, f3, f4) -> list[tuple]:
    m1 = {Path(f).stem: f for f in f1}
    m2 = {Path(f).stem: f for f in f2}
    m3 = {Path(f).stem: f for f in f3}
    m4 = {Path(f).stem: f for f in f4}
    common = sorted(set(m1) & set(m2) & set(m3) & set(m4), key=_natural_key)
    if not common:
        sys.exit("[ERROR] Keine gemeinsamen Frame-Namen in cam1/cam2/cam3/cam4.")
    print(f"   {len(common)} gemeinsame Frames  "
          f"(cam1:{len(m1)}  cam2:{len(m2)}  cam3:{len(m3)})  cam4:{len(m4)})")
    return [(name, m1[name], m2[name], m3[name], m4[name]) for name in common]


# ─── YOLO Inferenz ─────────────────────────────────────────────────────────

def run_yolo_pose(model, frame: np.ndarray):
    results = model(frame, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None
    kpts_xy   = results[0].keypoints.xy
    kpts_conf = results[0].keypoints.conf
    if kpts_xy.shape[0] == 0:
        return None, None
    return kpts_xy[0].cpu().numpy(), kpts_conf[0].cpu().numpy()


def extract_landmark_2d(kp_xy, kp_conf, idx: int, thr: float):
    if kp_xy is None or kp_conf is None:
        return None
    if kp_conf[idx] < thr:
        return None
    return np.array(kp_xy[idx], dtype=np.float32)


# ─── Kalibrierung ──────────────────────────────────────────────────────────

def load_calibration(path: str, cam_key: str) -> dict:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kalibrierungsdatei nicht gefunden: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"   Kalibrierung geladen: {path}  (RMS: {data['rms']:.4f})")
    # Die zweite Kamera ist in der .pkl immer als K2/D2 gespeichert.
    # Für calib_13 zusätzlich unter K3/D3 verfügbar machen
    # (so erwartet es triangulate_point).
    if cam_key != "2":
        data[f"K{cam_key}"] = data["K2"]
        data[f"D{cam_key}"] = data["D2"]
    return data


def rescale_calibration_to_image(stereo: dict, actual_wh: tuple) -> tuple:
    """
    Skaliert die Kameramatrizen K auf die tatsächliche Bildauflösung.

    Die Kalibrierung wurde bei voller Sensorauflösung (stereo['image_size'])
    durchgeführt; die Frames sind hier evtl. herunterskaliert
    (z.B. XI_DWN_2x2 → halbe Auflösung). Ohne diese Anpassung liegt der
    Hauptpunkt (cx, cy) um den Skalierungsfaktor daneben → grobe 3D-Fehler.

    fx, fy, cx, cy werden mit sx = W_ist/W_kalib bzw. sy = H_ist/H_kalib
    skaliert. Die Verzeichnung D bleibt unverändert (skalierungsinvariant).
    Modifiziert das Dict in-place und gibt (sx, sy) zurück.
    """
    calib_w, calib_h = stereo["image_size"]
    act_w, act_h = actual_wh
    sx = act_w / calib_w
    sy = act_h / calib_h
    for key in ("K1", "K2", "K3"):
        if key in stereo:
            K = np.asarray(stereo[key], dtype=np.float64).copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            stereo[key] = K
    return sx, sy


# ─── Triangulation (DLT, identisch zur bewährten Version) ──────────────────

def triangulate_point(pt1, pt2, pt3, pt4, s12: dict, s13: dict, s14: dict, 
                       max_reproj_error: float = 0.0) -> list | None:
    if pt1 is None or pt2 is None or pt3 is None:
        return None

    K1, D1 = s12["K1"], s12["D1"]
    K2, D2 = s12["K2"], s12["D2"]
    R_12, T_12 = s12["R"], s12["T"].reshape(3, 1)
    K3, D3 = s13["K3"], s13["D3"]
    R_13, T_13 = s13["R"], s13["T"].reshape(3, 1)
    K4, D4 = s14["K4"], s14["D4"]
    R_14, T_14 = s14["R"], s14["T"].reshape(3, 1)

    P1 = K1 @ np.hstack([np.eye(3),  np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R_12,       T_12])
    P3 = K3 @ np.hstack([R_13,       T_13])
    P4 = K4 @ np.hstack([R_14,       T_14])

    pt1u = cv2.undistortPoints(pt1.reshape(1, 1, 2), K1, D1, P=K1).reshape(2)
    pt2u = cv2.undistortPoints(pt2.reshape(1, 1, 2), K2, D2, P=K2).reshape(2)
    pt3u = cv2.undistortPoints(pt3.reshape(1, 1, 2), K3, D3, P=K3).reshape(2)
    pt4u = cv2.undistortPoints(pt4.reshape(1, 1, 2), K4, D4, P=K4).reshape(2)

    A = np.array([
        pt1u[0] * P1[2] - P1[0],
        pt1u[1] * P1[2] - P1[1],
        pt2u[0] * P2[2] - P2[0],
        pt2u[1] * P2[2] - P2[1],
        pt3u[0] * P3[2] - P3[0],
        pt3u[1] * P3[2] - P3[1],
        pt4u[0] * P4[2] - P4[0],
        pt4u[1] * P4[2] - P4[1],
    ], dtype=np.float64)

    try:
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-10:
            return None
        pt3d = X[:3] / X[3]
    except np.linalg.LinAlgError:
        return None

    if max_reproj_error > 0:
        def _err(K, D, R, T):
            rvec, _ = cv2.Rodrigues(R)
            proj, _ = cv2.projectPoints(pt3d.reshape(1, 3), rvec, T.flatten(), K, D)
            return float(np.linalg.norm(proj.reshape(2) - pt_obs))

        for pt_obs, K, D, R, T in [
            (pt1, K1, D1, np.eye(3),  np.zeros((3, 1))),
            (pt2, K2, D2, R_12,       T_12),
            (pt3, K3, D3, R_13,       T_13),
            (pt4, K4, D4, R_14,       T_14),
        ]:
            if _err(K, D, R, T) > max_reproj_error:
                return None

    return pt3d.tolist()


def triangulate_frame(kp1_xy, kp1_conf, kp2_xy, kp2_conf,
                       kp3_xy, kp3_conf, kp4_xy, kp4_conf, s12, s13, s14,
                       thr: float, max_reproj_error: float) -> list:
    return [
        triangulate_point(
            extract_landmark_2d(kp1_xy, kp1_conf, i, thr),
            extract_landmark_2d(kp2_xy, kp2_conf, i, thr),
            extract_landmark_2d(kp3_xy, kp3_conf, i, thr),
            extract_landmark_2d(kp4_xy, kp4_conf, i, thr),
            s12, s13, s14, max_reproj_error,
        )
        for i in range(NUM_KEYPOINTS)
    ]


# ─── RTS Smoother ──────────────────────────────────────────────────────────

def smooth_rts(frames_3d: list,
               measurement_noise_m: float,
               process_noise_acc: float,
               fps: float) -> list:
    """
    Rauch-Tung-Striebel (RTS) Smoother – bidirektionaler Kalman-Smoother.

    Algorithmus:
      1. Vorwärts-Kalman-Filter (Prediction + Update)
      2. Rückwärts-RTS-Pass (Smoother-Gain nutzt Zukunftsinformation)

    State: [x, y, z, vx, vy, vz]
    Transition: konstante Geschwindigkeit (constant velocity model)
    Observation: nur Position [x, y, z]

    Parameter:
        measurement_noise_m : Std.abw. des 3D-Messfehlers in Metern (z.B. 0.05)
        process_noise_acc   : Std.abw. der Beschleunigung in m/s² (z.B. 10.0)
        fps                 : Aufnahme-FPS (für physikalische Skalierung)

    None-Frames werden korrekt behandelt: nur Prediction, kein Update.
    Das Ergebnis enthält ausschließlich Werte für ursprünglich gültige Frames.
    """
    n = len(frames_3d)
    if n < 3:
        return frames_3d

    dt = 1.0 / fps

    # Zustandsübergangsmatrix F (konstante Geschwindigkeit)
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    # Beobachtungsmatrix H (nur Position messbar)
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    # Prozessrausch-Kovarianz Q  (Diskrete-Weiß-Beschleunigungsrausch-Modell)
    # G ist der Einflussvektor der Beschleunigung auf den State
    G = np.array([[dt**2 / 2],
                  [dt**2 / 2],
                  [dt**2 / 2],
                  [dt],
                  [dt],
                  [dt]])
    Q = (process_noise_acc ** 2) * (G @ G.T)

    # Messrausch-Kovarianz R
    R_obs = (measurement_noise_m ** 2) * np.eye(3)

    result = [list(f) for f in frames_3d]

    for ki in range(NUM_KEYPOINTS):
        valid = [frames_3d[fi][ki] is not None for fi in range(n)]
        n_valid = sum(valid)
        if n_valid < 3:
            continue

        # Erstes gültiges Frame finden
        start = next(fi for fi in range(n) if valid[fi])

        # Anfangszustand: Position aus erstem gültigen Frame,
        # Geschwindigkeit aus den ersten zwei gültigen Frames geschätzt
        x0 = np.zeros(6)
        x0[:3] = frames_3d[start][ki]
        nxt = next((fi for fi in range(start + 1, n) if valid[fi]), None)
        if nxt is not None:
            gap_s = (nxt - start) * dt
            x0[3:] = (np.array(frames_3d[nxt][ki]) - x0[:3]) / max(gap_s, dt)

        # Anfangs-Kovarianz: moderate Unsicherheit für Position & Geschwindigkeit
        P0 = np.diag([0.04, 0.04, 0.04,   # 20 cm pos Unsicherheit
                      1.00, 1.00, 1.00])   # 1 m/s vel Unsicherheit

        # Arrays für Forward-Pass
        x_pred = np.zeros((n, 6))
        P_pred = np.zeros((n, 6, 6))
        x_filt = np.zeros((n, 6))
        P_filt = np.zeros((n, 6, 6))

        x_filt[start] = x0
        P_filt[start] = P0
        x_pred[start] = x0
        P_pred[start] = P0

        # ── Vorwärts-Kalman-Pass ────────────────────────────────────────────
        for fi in range(start + 1, n):
            # Prediction
            x_pred[fi] = F @ x_filt[fi - 1]
            P_pred[fi] = F @ P_filt[fi - 1] @ F.T + Q

            if valid[fi]:
                # Innovation und Kalman-Gain
                z   = np.array(frames_3d[fi][ki], dtype=np.float64)
                S   = H @ P_pred[fi] @ H.T + R_obs          # 3×3, positiv definit
                K_g = sp_linalg.solve(S.T,
                                      (P_pred[fi] @ H.T).T,
                                      assume_a="pos").T      # 6×3 Kalman-Gain
                innov      = z - H @ x_pred[fi]
                x_filt[fi] = x_pred[fi] + K_g @ innov
                P_filt[fi] = (np.eye(6) - K_g @ H) @ P_pred[fi]
            else:
                # Kein Keypoint → Prediction übernehmen
                x_filt[fi] = x_pred[fi]
                P_filt[fi] = P_pred[fi]

        # ── Rückwärts-RTS-Pass ─────────────────────────────────────────────
        x_smooth = x_filt.copy()

        for fi in range(n - 2, start - 1, -1):
            if np.all(P_pred[fi + 1] == 0):
                continue
            try:
                # Smoother-Gain: G = P_filt[fi] · F^T · P_pred[fi+1]^{-1}
                # Stabil mit solve statt inv
                G_rts = sp_linalg.solve(
                    P_pred[fi + 1].T,
                    (P_filt[fi] @ F.T).T,
                    assume_a="pos",
                ).T                                         # 6×6
                x_smooth[fi] = x_filt[fi] + G_rts @ (x_smooth[fi + 1] - x_pred[fi + 1])
            except (np.linalg.LinAlgError, sp_linalg.LinAlgError):
                pass   # Numerisch instabil → gefilterten Wert behalten

        # Geglättete Positionen zurückschreiben (nur für ursprünglich gültige Frames)
        for fi in range(start, n):
            if valid[fi]:
                result[fi][ki] = x_smooth[fi, :3].tolist()

    return result


# ─── Knochenlängen-Normalisierung ─────────────────────────────────────────

def compute_reference_lengths(frames_3d: list) -> dict:
    """Median-Knochenlängen über alle Frames, physiologisch gefiltert."""
    lengths: dict[tuple, list] = {pair: [] for pair in SKELETON}
    for frame in frames_3d:
        for i, j in SKELETON:
            if frame[i] is None or frame[j] is None:
                continue
            l = float(np.linalg.norm(np.array(frame[i]) - np.array(frame[j])))
            lo, hi = BONE_PHYS_RANGE.get((i, j), (0.01, 10.0))
            if lo <= l <= hi:
                lengths[(i, j)].append(l)

    ref = {}
    for pair, vals in lengths.items():
        if len(vals) >= 5:
            ref[pair] = float(np.median(vals))

    valid = len(ref)
    print(f"   Referenzlängen: {valid}/{len(SKELETON)} Knochen")
    for (i, j), l in list(ref.items())[:4]:
        print(f"     {KP_NAMES[i]:12s} → {KP_NAMES[j]:12s}: {l*100:.1f} cm")
    return ref


def enforce_bone_lengths(frames_3d: list, ref_lengths: dict,
                          alpha: float) -> list:
    """Korrigiert Knochenlängen sanft Richtung Referenzwert (proximal → distal)."""
    if alpha <= 0 or not ref_lengths:
        return frames_3d
    result = []
    for frame in frames_3d:
        pts = [np.array(p, dtype=np.float64) if p is not None else None
               for p in frame]
        for (i, j), ref_len in ref_lengths.items():
            if pts[i] is None or pts[j] is None:
                continue
            vec = pts[j] - pts[i]
            cur_len = np.linalg.norm(vec)
            if cur_len < 1e-6:
                continue
            target = (1.0 - alpha) * cur_len + alpha * ref_len
            pts[j] = pts[i] + vec * (target / cur_len)
        result.append([p.tolist() if p is not None else None for p in pts])
    return result


# ─── Koordinatensystem ─────────────────────────────────────────────────────

def opencv_to_blender(frames_3d: list) -> list:
    """OpenCV (Y↓ Z vorwärts) → Blender (Y↑ Z rückwärts): Y = −Y,  Z = −Z"""
    out = []
    for frame in frames_3d:
        out.append([
            [x, -y, -z] if pt is not None else None
            for pt in frame
            for x, y, z in [pt if pt is not None else (0, 0, 0)]
        ])
    return out


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Frames → YOLO26 → Triangulation → RTS-Smoother → animation.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frames-dir",      required=True)
    parser.add_argument("--calib-12",        default="stereo_cam1_cam2.pkl")
    parser.add_argument("--calib-13",        default="stereo_cam1_cam3.pkl")
    parser.add_argument("--calib-14",        default="stereo_cam1_cam4.pkl")
    parser.add_argument("--output",          default="animation.json")
    parser.add_argument("--model",           default="yolo26n-pose.pt")
    parser.add_argument("--fps",             type=float, default=20.0)
    parser.add_argument("--kpt-thr",         type=float, default=0.5,
                        help="YOLO Keypoint-Konfidenzschwelle (Standard: 0.5)")
    parser.add_argument("--max-reproj-err",  type=float, default=0.0,
                        help="Harter Reprojektionsfilter in Pixeln (0 = aus)")
    parser.add_argument("--bone-alpha",      type=float, default=0.7,
                        help="Knochenlängen-Korrektur 0..1 (0 = aus, Standard: 0.7)")
    parser.add_argument("--rts-meas-noise",  type=float, default=5.0,
                        help="RTS: Messfehler-Std.abw. in cm (0 = kein RTS, "
                             "Standard: 5.0 cm). Größer → stärkere Glättung.")
    parser.add_argument("--rts-proc-noise",  type=float, default=10.0,
                        help="RTS: Beschleunigungs-Rauschen in m/s² "
                             "(Standard: 10.0). Kleiner → sanftere Geschwindigkeit.")
    args = parser.parse_args()

    frames_dir = os.path.expanduser(args.frames_dir)
    cam_dirs = {k: os.path.join(frames_dir, k) for k in ("cam1", "cam2", "cam3", "cam4")}
    for name, d in cam_dirs.items():
        if not os.path.isdir(d):
            sys.exit(f"[ERROR] Ordner nicht gefunden: {d}")

    print("=" * 60)
    print("  YOLO26 Pose Offline-Triangulation")
    print("=" * 60)
    use_rts = args.rts_meas_noise > 0
    print(f"  RTS-Smoother        : "
          f"{'AN  (meas=' + str(args.rts_meas_noise) + 'cm  proc=' + str(args.rts_proc_noise) + 'm/s²)' if use_rts else 'AUS'}")
    print(f"  Bone-Normalisierung : "
          f"{'AN  (alpha=' + str(args.bone_alpha) + ')' if args.bone_alpha > 0 else 'AUS'}")
    print(f"  Reprojektionsfilter : "
          f"{'AN  (' + str(args.max_reproj_err) + ' px)' if args.max_reproj_err > 0 else 'AUS'}")

    print("\n📂 Lese Frames...")
    aligned  = align_frames(
        get_sorted_frames(cam_dirs["cam1"]),
        get_sorted_frames(cam_dirs["cam2"]),
        get_sorted_frames(cam_dirs["cam3"]),
        get_sorted_frames(cam_dirs["cam4"]),
    )
    n_frames = len(aligned)

    print("\n📂 Lade Kalibrierungen...")
    stereo_12 = load_calibration(args.calib_12, "2")
    stereo_13 = load_calibration(args.calib_13, "3")
    stereo_14 = load_calibration(args.calib_14, "4")

    # Intrinsics auf die tatsächliche Bildauflösung skalieren.
    # Die Kalibrierung war bei voller Sensorauflösung; die Frames sind hier
    # evtl. heruntergerechnet (XI_DWN_2x2 → halbe Auflösung).
    probe = cv2.imread(aligned[0][1])
    if probe is None:
        sys.exit(f"[ERROR] Referenzbild nicht lesbar: {aligned[0][1]}")
    actual_wh = (probe.shape[1], probe.shape[0])
    calib_wh = tuple(stereo_12["image_size"])
    sx, sy = rescale_calibration_to_image(stereo_12, actual_wh)
    rescale_calibration_to_image(stereo_13, actual_wh)
    rescale_calibration_to_image(stereo_14, actual_wh)
    if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
        print(f"   ⚠️  Bildauflösung {actual_wh[0]}×{actual_wh[1]} ≠ "
              f"Kalibrierung {calib_wh[0]}×{calib_wh[1]}  →  "
              f"Intrinsics skaliert (sx={sx:.4f}, sy={sy:.4f})")
    else:
        print(f"   Auflösung passt zur Kalibrierung ({actual_wh[0]}×{actual_wh[1]})")

    print(f"\n📂 Lade YOLO26 Modell: {args.model} ...")
    model = YOLO(args.model)
    print("✅ Modell geladen.\n")

    frames_3d = []
    stats = {"ok": 0, "reproj_filtered": 0, "low_conf": 0, "no_person": 0}

    print(f"🔄 Verarbeite {n_frames} Frames...\n")

    for idx, (name, p1, p2, p3, p4) in enumerate(aligned):
        f1 = cv2.imread(p1)
        f2 = cv2.imread(p2)
        f3 = cv2.imread(p3)
        f4 = cv2.imread(p4)

        if f1 is None or f2 is None or f3 is None or f4 is None:
            print(f"   ⚠️  Bild nicht lesbar – Frame '{name}' übersprungen.")
            frames_3d.append([None] * NUM_KEYPOINTS)
            stats["no_person"] += NUM_KEYPOINTS
            continue

        kp1_xy, kp1_conf = run_yolo_pose(model, f1)
        kp2_xy, kp2_conf = run_yolo_pose(model, f2)
        kp3_xy, kp3_conf = run_yolo_pose(model, f3)
        kp4_xy, kp4_conf = run_yolo_pose(model, f4)

        if kp1_xy is None or kp2_xy is None or kp3_xy is None or kp4_xy is None:
            frames_3d.append([None] * NUM_KEYPOINTS)
            stats["no_person"] += NUM_KEYPOINTS
        else:
            pts = triangulate_frame(
                kp1_xy, kp1_conf, kp2_xy, kp2_conf, kp3_xy, kp3_conf, kp4_xy, kp4_conf,
                stereo_12, stereo_13, stereo_14, args.kpt_thr, args.max_reproj_err,
            )
            for i, pt in enumerate(pts):
                if pt is not None:
                    stats["ok"] += 1
                else:
                    pt_raw = extract_landmark_2d(kp1_xy, kp1_conf, i, args.kpt_thr)
                    if pt_raw is not None:
                        stats["reproj_filtered"] += 1
                    else:
                        stats["low_conf"] += 1
            frames_3d.append(pts)

        if (idx + 1) % 20 == 0 or idx == 0:
            total = sum(stats.values())
            ok_pct = 100 * stats["ok"] / total if total else 0
            print(f"  [{idx+1:>5}/{n_frames}]  {100*(idx+1)/n_frames:5.1f}%  "
                  f"✅ {stats['ok']} ({ok_pct:.0f}%)  "
                  f"⚠ low-conf: {stats['low_conf']}  "
                  f"❌ kein Mensch: {stats['no_person']}", flush=True)

    total = n_frames * NUM_KEYPOINTS
    print(f"\n📊 Ergebnis ({n_frames} Frames × {NUM_KEYPOINTS} = {total} Keypoints):")
    print(f"   Trianguliert  : {stats['ok']:>7}  ({100*stats['ok']/total:.1f}%)")
    print(f"   Low-conf      : {stats['low_conf']:>7}  ({100*stats['low_conf']/total:.1f}%)")
    print(f"   Keine Person  : {stats['no_person']:>7}  ({100*stats['no_person']/total:.1f}%)")

    # Verarbeitungs-Reihenfolge: Bone → RTS → OpenCV→Blender
    # Bone zuerst: RTS glättet dann bereits korrigierte Längen
    if args.bone_alpha > 0:
        print(f"\n🔧 Berechne Referenz-Knochenlängen...")
        ref = compute_reference_lengths(frames_3d)
        if ref:
            print(f"🔧 Bone-Normalisierung  alpha={args.bone_alpha}...")
            frames_3d = enforce_bone_lengths(frames_3d, ref, args.bone_alpha)
        else:
            print("   ⚠️  Keine plausiblen Referenzlängen – übersprungen.")

    if use_rts:
        meas_m = args.rts_meas_noise / 100.0   # cm → m
        print(f"\n🔧 RTS-Smoother  meas={args.rts_meas_noise}cm  "
              f"proc={args.rts_proc_noise}m/s²  fps={args.fps}...")
        frames_3d = smooth_rts(frames_3d,
                                measurement_noise_m=meas_m,
                                process_noise_acc=args.rts_proc_noise,
                                fps=args.fps)
        print("   Fertig.")

    print("🔧 Koordinatensystem-Korrektur (OpenCV → Blender)...")
    frames_3d = opencv_to_blender(frames_3d)

    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    payload = {
        "fps":            args.fps,
        "num_frames":     n_frames,
        "num_keypoints":  NUM_KEYPOINTS,
        "keypoint_names": KP_NAMES,
        "bones":          [list(p) for p in SKELETON],
        "frames":         frames_3d,
    }

    print(f"\n💾 Schreibe {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(payload, f)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"✅ Gespeichert: {output_path}  ({size_mb:.1f} MB)")
    print("✅ Pipeline abgeschlossen.\n")


if __name__ == "__main__":
    main()
