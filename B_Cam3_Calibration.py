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
# Detector Setup (STABLE)
# =======================

CHARUCO_PARAMS = cv2.aruco.CharucoParameters()
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()

DETECTOR = cv2.aruco.CharucoDetector(
    BOARD,
    CHARUCO_PARAMS,
    DETECTOR_PARAMS
)

# =======================
# Paths (ANGEPASST FÜR ALLE cam3 QUELLEN)
# =======================

# Hier alle Ordner auflisten, in denen cam3 Bilder liegen könnten
IMAGE_DIRS = [ "calib_1_3", "calib_single_3"]
OUTPUT_FILE = "mono_cam3_pinhole.pkl"

# =======================
# Load Images (ANGEPASST)
# =======================

def load_images(folders):
    all_files = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"[HINWEIS] Ordner {folder} existiert nicht, überspringe...")
            continue
            
        # Suche gezielt nach cam3_*.png und cam3_*.jpg
        # Das verhindert, dass versehentlich cam2-Bilder aus den Stereo-Ordnern geladen werden
        pngs = glob.glob(os.path.join(folder, "cam3_*.png"))
        jpgs = glob.glob(os.path.join(folder, "cam3_*.jpg"))
        all_files.extend(pngs + jpgs)
        
    files = sorted(all_files)
    if not files:
        raise RuntimeError(f"❌ Keine cam3-Bilder in den Verzeichnissen {folders} gefunden.")
    
    print(f"✅ Insgesamt {len(files)} Bilder für cam3 gefunden.")
    return files

# =======================
# Calibration
# =======================

def calculate_sharpness(gray):
    """Berechnet Bildschärfe mit Laplacian Varianz"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_coverage(corners, image_size):
    """Berechnet wie gut das Bild abgedeckt ist"""
    if len(corners) == 0:
        return 0.0
    points = corners.reshape(-1, 2)
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    area = (x_range * y_range) / (image_size[0] * image_size[1])
    return area

def calibrate(images):
    # Erste Pass: Sammle ALLE Detektionen mit Qualitätsmetriken
    candidates = []
    image_size = None
    
    print("🔍 Pass 1: Detecting ChArUco corners...")
    for path in tqdm(images):
        img = cv2.imread(path)
        if img is None:
            continue
        
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]
        
        corners, ids, _, _ = DETECTOR.detectBoard(gray)
        
        if ids is not None and len(ids) >= 6:
            # Qualitätsmetriken berechnen
            sharpness = calculate_sharpness(gray)
            coverage = calculate_coverage(corners, image_size)
            num_corners = len(ids)
            
            candidates.append({
                'corners': corners,
                'ids': ids,
                'sharpness': sharpness,
                'coverage': coverage,
                'num_corners': num_corners,
                'path': path
            })
    
    print(f"✅ {len(candidates)} Bilder mit Detektion gefunden")
    
    # Zweiter Pass: Qualitätsfilterung
    print("\n📊 Qualitätsfilterung...")
    
    # Schwellwerte definieren
    sharpness_threshold = np.percentile([c['sharpness'] for c in candidates], 25)  # untere 25% raus
    min_corners = 20  # mindestens 20 Corners
    min_coverage = 0.15  # mindestens 15% Bildabdeckung
    
    filtered = [
        c for c in candidates
        if c['sharpness'] > sharpness_threshold
        and c['num_corners'] >= min_corners
        and c['coverage'] >= min_coverage
    ]
    
    print(f"   Sharpness-Filter: {sharpness_threshold:.1f}")
    print(f"   Min Corners: {min_corners}")
    print(f"   Min Coverage: {min_coverage:.2%}")
    print(f"✅ {len(filtered)}/{len(candidates)} Bilder nach Filterung")
    
    if len(filtered) < 10:
        print("⚠️ Warnung: Weniger als 10 hochwertige Bilder! Verwende lockere Kriterien...")
        filtered = sorted(candidates, key=lambda x: x['num_corners'], reverse=True)[:15]
    
    # Diversität: Vermeide zu ähnliche Perspektiven
    # (Optional: Clustering basierend auf rvec/tvec)
    
    # Extrahiere finale Daten
    all_corners = [c['corners'] for c in filtered]
    all_ids = [c['ids'] for c in filtered]
    
    # Zeige Statistiken
    print("\n📈 Verwendete Bilder:")
    for i, c in enumerate(filtered[:5]):  # erste 5 anzeigen
        print(f"   {os.path.basename(c['path'])}: "
              f"{c['num_corners']} corners, "
              f"sharpness={c['sharpness']:.0f}, "
              f"coverage={c['coverage']:.1%}")
    if len(filtered) > 5:
        print(f"   ... und {len(filtered)-5} weitere")
    
    # Rest der Kalibrierung wie gehabt
    board_corners = BOARD.getChessboardCorners().astype(np.float32)
    obj_points = []
    img_points = []
    
    for corners, ids in zip(all_corners, all_ids):
        ids_1d = ids.flatten().astype(np.int32)
        obj = board_corners[ids_1d]
        img = corners.reshape(-1, 2)
        obj_points.append(obj)
        img_points.append(img.astype(np.float32))
    
    print(f"\n📐 Starte Pinhole-Kalibrierung mit {len(obj_points)} Datensätzen...")
    
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, K, D
    )
    
    # Dritter Pass: Per-Image Reprojection Error
    print("\n📊 Per-Image Reprojection Errors:")
    for i, (obj, img, rvec, tvec) in enumerate(zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        error = cv2.norm(img, projected.reshape(-1, 2), cv2.NORM_L2) / len(img)
        if error > rms * 1.5:  # Outliers markieren
            print(f"   ⚠️ Bild {i}: {error:.4f} (Outlier!)")
        elif i < 5:
            print(f"   ✓ Bild {i}: {error:.4f}")
    
    print(f"\n📊 RMS Fehler: {rms:.4f}")
    print("K (Intrinsics):\n", K)
    print("D (Distortion):\n", D.T)
    
    # ... Speichern wie gehabt
    data = {
        "K": K,
        "D": D,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "image_size": image_size,
        "rms": rms
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print("✅ Kalibrierung gespeichert in:", OUTPUT_FILE)

# =======================
# Main
# =======================

if __name__ == "__main__":
    try:
        images = load_images(IMAGE_DIRS)
        calibrate(images)
    except Exception as e:
        print(f"Fehler: {e}")