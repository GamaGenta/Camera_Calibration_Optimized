import numpy as np
import cv2
import pickle
from ultralytics import YOLO
from ximea import xiapi
import os
import time
import threading, pose_server

# =======================
# Configuration
# =======================

STEREO_CALIB_FILE_1_2 = "./stereo_cam1_cam2.pkl"
STEREO_CALIB_FILE_1_3 = "./stereo_cam1_cam3.pkl"
STEREO_CALIB_FILE_1_4 = "./stereo_cam1_cam4.pkl"

# Kamera-Einstellungen
EXPOSURE_US = 30000
GAIN = 5.0
WB_RED = 1.75
WB_GREEN = 1.0
WB_BLUE = 2.25

CAM_SERIALS = {
    "CUCAU1829019": 0,  # Cam0 — aktuell Index 0
    "CUCAU1829041": 1,  # Cam1 — aktuell Index 1
    "CUCAU1829031": 2,  # Cam2 — aktuell Index 2
}

# Visualisierung
DISPLAY_SIZE = (800, 600)   # Größe jedes einzelnen Kamerabildes im Fenster
TARGET_FPS = 20
FRAME_SKIP = 1

# YOLO26 Pose Modell
# Verfügbare Modelle: yolo26n-pose.pt, yolo26s-pose.pt, yolo26m-pose.pt,
#                     yolo26l-pose.pt, yolo26x-pose.pt  (n=nano, x=extra-large)
YOLO_MODEL_PATH = "yolo26n-pose.pt"

# YOLO26 / COCO Keypoint-Indizes (17 Punkte)
# 0: Nose       1: Left Eye    2: Right Eye   3: Left Ear    4: Right Ear
# 5: L-Shoulder 6: R-Shoulder  7: L-Elbow     8: R-Elbow
# 9: L-Wrist   10: R-Wrist    11: L-Hip      12: R-Hip
# 13: L-Knee   14: R-Knee     15: L-Ankle    16: R-Ankle
NUM_KEYPOINTS = 17


# Keypoint-Namen für Labels
KP_NAMES = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
]

# COCO Skeleton-Verbindungen
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # Kopf
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Arme
    (5, 11), (6, 12), (11, 12),                  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)       # Beine
]

# Konfidenz-Schwellwert
KEYPOINT_CONF_THRESHOLD = 0.5


# Farben (BGR)
COLOR_SKELETON   = (0, 255, 0)
COLOR_KEYPOINT   = (0, 0, 255)
COLOR_HAND_LEFT  = (255, 100, 0)
COLOR_HAND_RIGHT = (0, 100, 255)
COLOR_LABEL      = (255, 255, 255)
COLOR_BG         = (50, 50, 50)
COLOR_INFO       = (255, 200, 0)
COLOR_DIST       = (0, 255, 255)

font = cv2.FONT_HERSHEY_SIMPLEX


# =======================
# YOLO Modell
# =======================

def load_yolo_model():
    """Lädt das YOLO26 Pose-Modell"""
    print(f"📂 Lade YOLO26 Pose-Modell: {YOLO_MODEL_PATH} ...")
    model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO26 Pose-Modell geladen")
    return model


# =======================
# Kamera Setup
# =======================

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

        try:
            cam.set_downsampling('XI_DWN_1x1')
        except Exception:
            pass

        cam.set_exposure(EXPOSURE_US)
        cam.set_gain(GAIN)
        cam.disable_auto_wb()
        cam.set_wb_kr(WB_RED)
        cam.set_wb_kg(WB_GREEN)
        cam.set_wb_kb(WB_BLUE)
        cam.start_acquisition()

        # Warmup
        for _ in range(3):
            img = xiapi.Image()
            cam.get_image(img)

        print(f"✅ {name} bereit | SN: {sn} | Exposure: {cam.get_exposure()} µs | Gain: {cam.get_gain()}")
        return sn

    except Exception as e:
        print(f"❌ Fehler bei {name}: {e}")
        return False


def capture_frame(cam):
    img = xiapi.Image()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    return frame


# =======================
# Stereo Kalibrierung
# =======================

def load_stereo_calibration_1_2():
    """Lädt die Stereokalibrierung Cam1–Cam2"""
    print("📂 Lade Stereokalibrierung Cam1–Cam2...")
    if not os.path.exists(STEREO_CALIB_FILE_1_2):
        raise FileNotFoundError(f"❌ {STEREO_CALIB_FILE_1_2} nicht gefunden!")
    with open(STEREO_CALIB_FILE_1_2, "rb") as f:
        stereo_data = pickle.load(f)
    print(f"✅ Kalibrierung geladen (RMS: {stereo_data['rms']:.4f})")

    K1, D1 = stereo_data['K1'], stereo_data['D1']
    K2, D2 = stereo_data['K2'], stereo_data['D2']
    R1, R2 = stereo_data['R1'], stereo_data['R2']
    P1, P2 = stereo_data['P1'], stereo_data['P2']
    image_size = stereo_data['image_size']

    stereo_data['map1x'], stereo_data['map1y'] = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    stereo_data['map2x'], stereo_data['map2y'] = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    return stereo_data


def load_stereo_calibration_1_3():
    """Lädt die Stereokalibrierung Cam1–Cam3"""
    print("📂 Lade Stereokalibrierung Cam1–Cam3...")
    if not os.path.exists(STEREO_CALIB_FILE_1_3):
        raise FileNotFoundError(f"❌ {STEREO_CALIB_FILE_1_3} nicht gefunden!")
    with open(STEREO_CALIB_FILE_1_3, "rb") as f:
        stereo_data = pickle.load(f)
    print(f"✅ Kalibrierung geladen (RMS: {stereo_data['rms']:.4f})")

    K1, D1 = stereo_data['K1'], stereo_data['D1']
    K3, D3 = stereo_data['K3'], stereo_data['D3']
    R1, R2 = stereo_data['R1'], stereo_data['R2']
    P1, P2 = stereo_data['P1'], stereo_data['P2']
    image_size = stereo_data['image_size']

    stereo_data['map1x'], stereo_data['map1y'] = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    stereo_data['map3x'], stereo_data['map3y'] = cv2.initUndistortRectifyMap(
        K3, D3, R2, P2, image_size, cv2.CV_32FC1)
    return stereo_data
    
#new
def load_stereo_calibration_1_4():
    """Lädt die Stereokalibrierung Cam1–Cam4"""
    print("📂 Lade Stereokalibrierung Cam1–Cam4...")
    if not os.path.exists(STEREO_CALIB_FILE_1_4):
        raise FileNotFoundError(f"❌ {STEREO_CALIB_FILE_1_4} nicht gefunden!")
    with open(STEREO_CALIB_FILE_1_4, "rb") as f:
        stereo_data = pickle.load(f)
    print(f"✅ Kalibrierung geladen (RMS: {stereo_data['rms']:.4f})")

    K1, D1 = stereo_data['K1'], stereo_data['D1']
    K4, D4 = stereo_data['K4'], stereo_data['D4']
    R1, R2 = stereo_data['R1'], stereo_data['R2']
    P1, P2 = stereo_data['P1'], stereo_data['P2']
    image_size = stereo_data['image_size']

    stereo_data['map1x'], stereo_data['map1y'] = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    stereo_data['map4x'], stereo_data['map4y'] = cv2.initUndistortRectifyMap(
        K4, D4, R2, P2, image_size, cv2.CV_32FC1)
    return stereo_data


# =======================
# YOLO26 Inferenz
# =======================

def run_yolo_pose(model, frame):
    """
    Führt YOLO26 Pose-Inferenz auf einem Frame aus.

    Returns:
        keypoints_xy  : np.ndarray (17, 2) – Pixelkoordinaten der ersten Person
        keypoints_conf: np.ndarray (17,)   – Konfidenzwerte
        oder (None, None) falls keine Person erkannt
    """
    results = model(frame, verbose=False)

    if not results or results[0].keypoints is None:
        return None, None

    kpts_xy   = results[0].keypoints.xy
    kpts_conf = results[0].keypoints.conf

    if kpts_xy.shape[0] == 0:
        return None, None

    return kpts_xy[0].cpu().numpy(), kpts_conf[0].cpu().numpy()


def extract_landmark_2d(keypoints_xy, keypoints_conf, idx):
    """
    Extrahiert 2D-Koordinaten eines einzelnen Keypoints.

    Returns:
        np.ndarray([x, y], float32) oder None wenn unter Konfidenzschwelle
    """
    if keypoints_xy is None or keypoints_conf is None:
        return None
    if keypoints_conf[idx] < KEYPOINT_CONF_THRESHOLD:
        return None
    x, y = keypoints_xy[idx]
    return np.array([x, y], dtype=np.float32)


# =======================
# Triangulation (3D)
# =======================

def triangulate_points(pt2d_1, pt2d_2, pt2d_3, pt2d_4, stereo_data_1_2, stereo_data_1_3, stereo_data_1_4):
    """
    Trianguliert einen 3D-Punkt aus 2D-Koordinaten aller drei Kameras (DLT).

    Returns:
        np.ndarray([X, Y, Z]) in Metern oder None
    """
    if pt2d_1 is None or pt2d_2 is None or pt2d_3 is None:
        return None

    K1, D1 = stereo_data_1_2['K1'], stereo_data_1_2['D1']
    K2, D2 = stereo_data_1_2['K2'], stereo_data_1_2['D2']
    R_12   = stereo_data_1_2['R']
    T_12   = stereo_data_1_2['T'].reshape(3, 1)

    K3, D3 = stereo_data_1_3['K3'], stereo_data_1_3['D3']
    R_13   = stereo_data_1_3['R']
    T_13   = stereo_data_1_3['T'].reshape(3, 1)
    
    K4, D4 = stereo_data_1_4['K4'], stereo_data_1_4['D4']
    R_13   = stereo_data_1_4['R']
    T_13   = stereo_data_1_4['T'].reshape(3, 1)


    P1 = K1 @ np.hstack([np.eye(3),  np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R_12,       T_12])
    P3 = K3 @ np.hstack([R_13,       T_13])
    P3 = K4 @ np.hstack([R_14,       T_14])

    pt1 = cv2.undistortPoints(pt2d_1.reshape(1, 1, 2), K1, D1, P=K1).reshape(2)
    pt2 = cv2.undistortPoints(pt2d_2.reshape(1, 1, 2), K2, D2, P=K2).reshape(2)
    pt3 = cv2.undistortPoints(pt2d_3.reshape(1, 1, 2), K3, D3, P=K3).reshape(2)
    pt4 = cv2.undistortPoints(pt2d_4.reshape(1, 1, 2), K4, D4, P=K4).reshape(2)

    A = np.array([
        pt1[0] * P1[2, :] - P1[0, :],
        pt1[1] * P1[2, :] - P1[1, :],
        pt2[0] * P2[2, :] - P2[0, :],
        pt2[1] * P2[2, :] - P2[1, :],
        pt3[0] * P3[2, :] - P3[0, :],
        pt3[1] * P3[2, :] - P3[1, :],
    ], dtype=np.float64)

    try:
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-10:
            return None
        return X[:3] / X[3]
    except np.linalg.LinAlgError:
        return None


def triangulate_all_keypoints(kp1_xy, kp1_conf, kp2_xy, kp2_conf,
                               kp3_xy, kp3_conf, kp4_xy, kp4_conf, stereo_12, stereo_13, stereo_14):
    """
    Trianguliert alle 17 COCO-Keypoints zu 3D-Punkten.

    Returns:
        list[np.ndarray | None] – 17 Einträge, None wenn nicht sichtbar
    """
    return [
        triangulate_points(
            extract_landmark_2d(kp1_xy, kp1_conf, idx),
            extract_landmark_2d(kp2_xy, kp2_conf, idx),
            extract_landmark_2d(kp3_xy, kp3_conf, idx),
            extract_landmark_2d(kp4_xy, kp4_conf, idx),
            stereo_12, stereo_13, stereo_14
        )
        for idx in range(NUM_KEYPOINTS)
    ]


# =======================
# Visualisierung
# =======================

def draw_pose_yolo(frame, keypoints_xy, keypoints_conf, show_labels=False):
    """
    Zeichnet das YOLO26 Skeleton auf den Frame.
    Hände (Wrists) werden farblich hervorgehoben.

    Args:
        show_labels: Keypoint-Namen neben jedem Punkt anzeigen
    """
    if keypoints_xy is None:
        return frame

    # Skeleton-Linien
    for i, j in SKELETON:
        if (keypoints_conf[i] >= KEYPOINT_CONF_THRESHOLD and
                keypoints_conf[j] >= KEYPOINT_CONF_THRESHOLD):
            pt_i = tuple(keypoints_xy[i].astype(int))
            pt_j = tuple(keypoints_xy[j].astype(int))
            cv2.line(frame, pt_i, pt_j, COLOR_SKELETON, 2, cv2.LINE_AA)

    # Keypoints
    for idx in range(NUM_KEYPOINTS):
        if keypoints_conf[idx] < KEYPOINT_CONF_THRESHOLD:
            continue

        pt = tuple(keypoints_xy[idx].astype(int))


        cv2.circle(frame, pt, 4, COLOR_KEYPOINT, -1, cv2.LINE_AA)

        if show_labels:
            cv2.putText(frame, KP_NAMES[idx],
                        (pt[0] + 6, pt[1] - 4),
                        font, 0.32, COLOR_LABEL, 1, cv2.LINE_AA)

    return frame


def draw_overlay(frame, cam_label):
    """
    Fügt Kamera-Label (oben links) und Handabstand-Box (unten) ein.
    """
    h, w = frame.shape[:2]

    # Kamera-Label
    cv2.rectangle(frame, (0, 0), (170, 30), COLOR_BG, -1)
    cv2.putText(frame, cam_label, (8, 22), font, 0.65, COLOR_INFO, 2, cv2.LINE_AA)

    return frame


def build_display_grid(frames_labels):
    """
    Setzt drei annotierte Kamerabilder horizontal nebeneinander.

    Args:
        frames_labels: [(frame, label), ...]
        dist_m       : berechneter 3D-Handabstand in Metern oder None

    Returns:
        Kombiniertes BGR-Bild
    """
    tiles = []
    for frame, label in frames_labels:
        tile = cv2.resize(frame, DISPLAY_SIZE)
        draw_overlay(tile, label)
        tiles.append(tile)
    return np.hstack(tiles)


# =======================
# Main Loop
# =======================

def main():
    print("=" * 55)
    print("  YOLO26 Pose Tracking – 4-Kamera Stereo Setup")
    print("=" * 55)

    # Modell
    model = load_yolo_model()

    # Kalibrierung
    stereo_12 = load_stereo_calibration_1_2()
    stereo_13 = load_stereo_calibration_1_3()
    stereo_14 = load_stereo_calibration_1_4()

    # Kameras öffnen
    cam1 = xiapi.Camera(0)
    cam2 = xiapi.Camera(1)
    cam3 = xiapi.Camera(2)
    cam4 = xiapi.Camera(3)

    ok1 = setup_camera(cam1, "Kamera 1 (Referenz)")
    ok2 = setup_camera(cam2, "Kamera 2")
    ok3 = setup_camera(cam3, "Kamera 3")
    ok4 = setup_camera(cam4, "Kamera 4")

    if not (ok1 and ok2 and ok3 and ok4):
        print("❌ Konnte nicht alle Kameras öffnen. Abbruch.")
        return

    print("\n▶  Drücke  [Q]  zum Beenden")
    print("▶  Drücke  [L]  Keypoint-Labels ein/ausschalten\n")

    show_labels = False
    frame_count = 0
    t_last = time.time()

    cv2.namedWindow("YOLO26 Pose – 4 Kameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO26 Pose – 4 Kameras",
                     DISPLAY_SIZE[0] * 3, DISPLAY_SIZE[1]) #3->4?

    server_thread = threading.Thread(target=pose_server.run_server, daemon=True)
    server_thread.start()
    try:
        while True:
            frame_count += 1

            # Frames aufnehmen
            frame1 = capture_frame(cam1)
            frame2 = capture_frame(cam2)
            frame3 = capture_frame(cam3)
            frame4 = capture_frame(cam4)

            if frame_count % FRAME_SKIP != 0:
                continue

            # YOLO26 Pose-Inferenz auf allen Kameras
            kp1_xy, kp1_conf = run_yolo_pose(model, frame1)
            kp2_xy, kp2_conf = run_yolo_pose(model, frame2)
            kp3_xy, kp3_conf = run_yolo_pose(model, frame3)
            kp4_xy, kp4_conf = run_yolo_pose(model, frame4)
                        

            t_now = time.time()
            fps = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now

            # 3D-Triangulation aller 17 Keypoints
            points_3d = triangulate_all_keypoints(
                kp1_xy, kp1_conf,
                kp2_xy, kp2_conf,
                kp3_xy, kp3_conf,
                kp4_xy, kp4_conf,
                stereo_12, stereo_13, stereo_14
            )

            pose_server.send_pose_data(points_3d, fps=fps)

            # Skeleton + Hand-Linie auf jeden Frame zeichnen
            draw_pose_yolo(frame1, kp1_xy, kp1_conf, show_labels)

            draw_pose_yolo(frame2, kp2_xy, kp2_conf, show_labels)

            draw_pose_yolo(frame3, kp3_xy, kp3_conf, show_labels)

            draw_pose_yolo(frame4, kp4_xy, kp4_conf, show_labels)

            # 3-Kamera-Grid anzeigen
            grid = build_display_grid(
                [(frame1, "Kamera 1"),
                 (frame2, "Kamera 2"),
                 (frame3, "Kamera 3"),
                 (frame4, "Kamera 4")]
            )
            cv2.imshow("YOLO26 Pose – 4 Kameras", grid)


            # Tastatureingabe
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\n\n⏹  Beende...")
                break
            elif key in (ord('l'), ord('L')):
                show_labels = not show_labels
                print(f"\n  Labels: {'AN' if show_labels else 'AUS'}")

    except KeyboardInterrupt:
        print("\n\n⏹  KeyboardInterrupt – Beende...")

    finally:
        print("🔒 Kameras werden geschlossen...")
        for cam in (cam1, cam2, cam3, cam4):
            try:
                cam.stop_acquisition()
                cam.close_device()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("✅ Fertig.")


if __name__ == "__main__":
    main()
