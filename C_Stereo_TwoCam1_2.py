import numpy as np
import cv2
import os
import glob
import pickle
from tqdm import tqdm

# =======================
# ChArUco Board Definition
# =======================

SQUARES_X = 11
SQUARES_Y = 8

SQUARE_LENGTH = 0.060   # meters
MARKER_LENGTH = 0.045   # meters

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

BOARD = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    DICTIONARY
)
BOARD.setLegacyPattern(True)

# =======================
# Detector Setup
# =======================

CHARUCO_PARAMS = cv2.aruco.CharucoParameters()
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.CharucoDetector(
    BOARD,
    CHARUCO_PARAMS,
    DETECTOR_PARAMS
)

# =======================
# Paths
# =======================

STEREO_IMAGE_DIR = "calib_1_2"
MONO_CAM1_FILE = "mono_cam1_pinhole.pkl"
MONO_CAM2_FILE = "mono_cam2_pinhole.pkl"
OUTPUT_FILE = "stereo_cam1_cam2.pkl"

# =======================
# Load Mono Calibrations
# =======================

def load_mono_calibrations():
    """Lädt die Monokalibierungen von Cam1 und Cam2"""
    print("📂 Lade Monokalibierungen...")
    
    if not os.path.exists(MONO_CAM1_FILE):
        raise FileNotFoundError(f"❌ {MONO_CAM1_FILE} nicht gefunden!")
    if not os.path.exists(MONO_CAM2_FILE):
        raise FileNotFoundError(f"❌ {MONO_CAM2_FILE} nicht gefunden!")
    
    with open(MONO_CAM1_FILE, "rb") as f:
        cam1_data = pickle.load(f)
    
    with open(MONO_CAM2_FILE, "rb") as f:
        cam2_data = pickle.load(f)
    
    print(f"✅ Cam1 geladen: RMS={cam1_data['rms']:.4f}")
    print(f"✅ Cam2 geladen: RMS={cam2_data['rms']:.4f}")
    
    return cam1_data, cam2_data

# =======================
# Load Stereo Image Pairs
# =======================

def load_stereo_pairs():
    """Lädt synchronisierte Bildpaare aus dem Stereo-Ordner"""
    print(f"\n📂 Suche Stereo-Bildpaare in {STEREO_IMAGE_DIR}...")
    
    if not os.path.exists(STEREO_IMAGE_DIR):
        raise RuntimeError(f"❌ Ordner {STEREO_IMAGE_DIR} existiert nicht!")
    
    cam1_files = sorted(glob.glob(os.path.join(STEREO_IMAGE_DIR, "cam1_*.png")))
    cam2_files = sorted(glob.glob(os.path.join(STEREO_IMAGE_DIR, "cam2_*.png")))
    
    # Paare anhand des Timestamps matchen
    pairs = []
    for cam1_path in cam1_files:
        timestamp = os.path.basename(cam1_path).replace("cam1_", "").replace(".png", "")
        cam2_path = os.path.join(STEREO_IMAGE_DIR, f"cam2_{timestamp}.png")
        
        if os.path.exists(cam2_path):
            pairs.append((cam1_path, cam2_path))
        else:
            print(f"⚠️ Kein Match für {os.path.basename(cam1_path)}")
    
    if len(pairs) == 0:
        raise RuntimeError(f"❌ Keine passenden Stereo-Paare gefunden!")
    
    print(f"✅ {len(pairs)} Stereo-Bildpaare gefunden")
    return pairs

# =======================
# Detect Corresponding Points
# =======================

def detect_corresponding_points(pairs):
    """Detektiert ChArUco-Ecken in beiden Kameras und findet gemeinsame IDs"""
    print("\n🔍 Detektiere korrespondierende Punkte...")
    
    valid_pairs = []
    board_corners = BOARD.getChessboardCorners().astype(np.float32)
    
    for cam1_path, cam2_path in tqdm(pairs):
        img1 = cv2.imread(cam1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(cam2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            continue
        
        # Detektiere in beiden Bildern
        corners1, ids1, _, _ = DETECTOR.detectBoard(img1)
        corners2, ids2, _, _ = DETECTOR.detectBoard(img2)
        
        if ids1 is None or ids2 is None:
            continue
        
        if len(ids1) < 6 or len(ids2) < 6:
            continue
        
        # Finde gemeinsame IDs
        ids1_flat = ids1.flatten()
        ids2_flat = ids2.flatten()
        common_ids = np.intersect1d(ids1_flat, ids2_flat)
        
        if len(common_ids) < 6:
            continue

        name1 = os.path.basename(cam1_path)
        name2 = os.path.basename(cam2_path)
        print(f"   ⭐ Paar gefunden: {name1} & {name2} ({len(common_ids)} gemeinsame Punkte)")
        
        # Extrahiere nur die gemeinsamen Punkte
        mask1 = np.isin(ids1_flat, common_ids)
        mask2 = np.isin(ids2_flat, common_ids)
        
        # Sortiere nach IDs um sicherzustellen, dass die Reihenfolge übereinstimmt
        sort_idx1 = np.argsort(ids1_flat[mask1])
        sort_idx2 = np.argsort(ids2_flat[mask2])
        
        corners1_common = corners1[mask1][sort_idx1]
        corners2_common = corners2[mask2][sort_idx2]
        common_ids_sorted = ids1_flat[mask1][sort_idx1]
        
        # 3D-Objektpunkte
        obj_points = board_corners[common_ids_sorted]
        
        valid_pairs.append({
            'obj_points': obj_points,
            'img_points1': corners1_common.reshape(-1, 2).astype(np.float32),
            'img_points2': corners2_common.reshape(-1, 2).astype(np.float32),
            'num_points': len(common_ids_sorted),
            'cam1_path': cam1_path,
            'cam2_path': cam2_path
        })
    
    print(f"✅ {len(valid_pairs)} Bildpaare mit gemeinsamen Detektionen")
    
    if len(valid_pairs) < 5:
        print("⚠️ Warnung: Wenige gültige Paare! Verbesserung empfohlen.")
    
    return valid_pairs

# =======================
# Stereo Calibration
# =======================

def stereo_calibrate(valid_pairs, cam1_data, cam2_data):
    """Führt die Stereokalibrierung durch"""
    print("\n🔧 Starte Stereokalibrierung...")
    
    # Extrahiere Daten aus validen Paaren
    obj_points = [pair['obj_points'] for pair in valid_pairs]
    img_points1 = [pair['img_points1'] for pair in valid_pairs]
    img_points2 = [pair['img_points2'] for pair in valid_pairs]
    
    K1 = cam1_data['K']
    D1 = cam1_data['D']
    K2 = cam2_data['K']
    D2 = cam2_data['D']
    image_size = cam1_data['image_size']
    
    # Stereokalibrierung mit fixen intrinsischen Parametern
    flags = cv2.CALIB_FIX_INTRINSIC
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    print(f"   Anzahl Bildpaare: {len(obj_points)}")
    print(f"   Bildgröße: {image_size}")
    
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points1,
        img_points2,
        K1, D1,
        K2, D2,
        image_size,
        criteria=criteria,
        flags=flags
    )
    
    print(f"\n📊 Stereokalibrierung abgeschlossen!")
    print(f"   RMS Fehler: {rms:.4f} Pixel")
    print(f"\n🔄 Rotation Matrix (R):")
    print(R)
    print(f"\n📏 Translation Vector (T) [Meter]:")
    print(T.T)
    print(f"   Baseline: {np.linalg.norm(T):.4f} m")
    
    # Berechne Stereo-Rektifizierung
    print("\n🔧 Berechne Rektifizierung...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1,
        K2, D2,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 0 = nur valider Bereich, 1 = alle Pixel
    )
    
    print("✅ Rektifizierung berechnet")
    
    return {
        'K1': K1,
        'D1': D1,
        'K2': K2,
        'D2': D2,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'roi1': roi1,
        'roi2': roi2,
        'image_size': image_size,
        'rms': rms,
        'baseline': np.linalg.norm(T)
    }

# =======================
# Save Results
# =======================

def save_stereo_calibration(stereo_data):
    """Speichert die Stereokalibrierung"""
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(stereo_data, f)
    
    print(f"\n✅ Stereokalibrierung gespeichert in: {OUTPUT_FILE}")

# =======================
# Main
# =======================

def main():
    try:
        # 1. Lade Monokalibierungen
        cam1_data, cam2_data = load_mono_calibrations()
        
        # 2. Lade Stereo-Bildpaare
        pairs = load_stereo_pairs()
        
        # 3. Detektiere korrespondierende Punkte
        valid_pairs = detect_corresponding_points(pairs)
        
        if len(valid_pairs) < 3:
            print("❌ Zu wenige gültige Bildpaare für Stereokalibrierung!")
            return
        
        # 4. Führe Stereokalibrierung durch
        stereo_data = stereo_calibrate(valid_pairs, cam1_data, cam2_data)
        
        # 5. Speichere Ergebnisse
        save_stereo_calibration(stereo_data)
        
        print("\n" + "="*60)
        print("🎉 STEREOKALIBRIERUNG ERFOLGREICH ABGESCHLOSSEN!")
        print("="*60)
        print(f"\nDu kannst nun die Datei '{OUTPUT_FILE}' für")
        print("3D-Triangulation verwenden.")
        
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()