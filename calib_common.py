"""
calib_common.py
================
Gemeinsames Fundament fuer die gesamte Kalibrierungskette.

Wissenschaftlicher Hintergrund
------------------------------
Die urspruengliche Pipeline hat ueber die Haelfte aller Kalibrierbilder
*stillschweigend verworfen*, weil die Default-DetectorParameters von OpenCV
fuer 4112x3008-Bilder eines teils weit entfernten / gekippten ChArUco-Boards
voellig unzureichend sind. Folge: Cam2 wurde mit nur 3 Ansichten kalibriert,
die Distortion-Koeffizienten waren reiner Overfit (k2=-0.54, k3=+1.17), und die
Intrinsics waren trotz RMS<0.3px um hunderte Pixel falsch (cy-Fehler ~190px).

Dieses Modul zentralisiert:
  * die exakte Board-Definition (eine einzige Quelle der Wahrheit),
  * getunte Detektor-Parameter mit Subpixel-Refinement,
  * robuste Detektion + Objekt-/Bildpunkt-Extraktion via matchImagePoints,
  * Qualitaets-/Coverage-Diagnostik.

Alle Kalibrierungsskripte importieren ausschliesslich von hier, damit Mono-,
Stereo- und Multi-Cam-Stufen garantiert dieselbe Geometrie verwenden.
"""

import cv2
import numpy as np

# ======================================================================
# 1. BOARD-GEOMETRIE  (Single Source of Truth)
# ======================================================================
# WICHTIG: Diese Werte MUESSEN exakt dem physisch gedruckten Board
# entsprechen. SQUARE_LENGTH/MARKER_LENGTH definieren den metrischen
# Massstab der gesamten Rekonstruktion -- ein 2%-Druckfehler wird zu
# 2% Skalenfehler in jeder spaeteren 3D-Position.
SQUARES_X = 11
SQUARES_Y = 8
SQUARE_LENGTH = 0.060   # m  -> VOR Gebrauch mit Messschieber am Druck pruefen!
MARKER_LENGTH = 0.045   # m

# DICT_4X4_100 + Legacy-Pattern wurde empirisch bestaetigt:
# legacy=True liefert ~3x mehr Detektionen als legacy=False -> das
# physische Board ist ein Legacy-Board. Nicht aendern, ohne ein neues
# Board zu drucken.
ARUCO_DICT = cv2.aruco.DICT_4X4_100
USE_LEGACY_PATTERN = True

# Maximale Anzahl innerer Schachbrett-Ecken (zur Coverage-Normierung)
MAX_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)   # = 70


def make_board():
    """Erzeugt das ChArUco-Board (eine konsistente Instanz pro Aufruf)."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    board.setLegacyPattern(USE_LEGACY_PATTERN)
    return board


# ======================================================================
# 2. GETUNTE DETEKTOR-PARAMETER  (der eigentliche Fix)
# ======================================================================
def make_detector(board=None):
    """
    CharucoDetector mit fuer hochaufloesende, weit entfernte Boards
    optimierten Parametern.

    Schluessel-Aenderungen gegenueber den OpenCV-Defaults:
      * cornerRefinementMethod = CORNER_REFINE_SUBPIX
            -> Subpixel-genaue Marker-Ecken (kritisch fuer Praezision).
      * adaptiveThreshWinSizeMax 23 -> 53, feinere Steps
            -> robuste Binarisierung bei ungleichmaessiger Decken-
               beleuchtung und grossen Markern.
      * minMarkerPerimeterRate 0.03 -> 0.01
            -> erlaubt die KLEINEN Marker eines weit entfernten Boards,
               die der Default verwirft (Hauptursache der verworfenen Bilder).
      * minCornerDistanceRate gelockert
            -> volle, nahe Boards mit dicht stehenden Markern.

    Empirisch: Detektionsrate auf Cam2 von 10/30 -> 27/30.
    """
    if board is None:
        board = make_board()

    dp = cv2.aruco.DetectorParameters()
    # --- Subpixel-Refinement der ArUco-Markerecken ---
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.cornerRefinementWinSize = 5
    dp.cornerRefinementMaxIterations = 50
    dp.cornerRefinementMinAccuracy = 0.01
    # --- Adaptive Schwellwertbildung fuer grosse Bilder ---
    dp.adaptiveThreshWinSizeMin = 3
    dp.adaptiveThreshWinSizeMax = 53
    dp.adaptiveThreshWinSizeStep = 10
    # --- Marker-Groessenbereich weit oeffnen ---
    dp.minMarkerPerimeterRate = 0.01
    dp.maxMarkerPerimeterRate = 4.0
    dp.minCornerDistanceRate = 0.03
    # --- Fehlertoleranz beim Bit-Auslesen leicht erhoehen ---
    dp.maxErroneousBitsInBorderRate = 0.35
    dp.errorCorrectionRate = 0.6

    cp = cv2.aruco.CharucoParameters()
    # Interne Marker-Verfeinerung anhand der bekannten Board-Geometrie
    try:
        cp.tryRefineMarkers = True
    except AttributeError:
        pass

    return cv2.aruco.CharucoDetector(board, cp, dp)


# ======================================================================
# 3. DETEKTION + PUNKTEXTRAKTION
# ======================================================================
def detect_charuco(gray, detector, board, min_corners=6):
    """
    Detektiert das Board in einem Graustufenbild.

    Returns dict mit:
        charuco_corners (N,1,2), charuco_ids (N,1),
        obj_points (N,1,3), img_points (N,1,2)
    oder None, wenn weniger als `min_corners` Ecken gefunden wurden.

    obj/img-Punkte werden ueber board.matchImagePoints() erzeugt -- die
    von OpenCV empfohlene, geometrisch korrekte Methode (statt manuellem
    Indexieren von getChessboardCorners()).
    """
    cc, cids, mc, mids = detector.detectBoard(gray)
    if cids is None or len(cids) < min_corners:
        return None

    obj, img = board.matchImagePoints(cc, cids)
    if obj is None or len(obj) < min_corners:
        return None

    return {
        "charuco_corners": cc,
        "charuco_ids": cids,
        "obj_points": obj.reshape(-1, 1, 3).astype(np.float32),
        "img_points": img.reshape(-1, 1, 2).astype(np.float32),
        "ids_flat": cids.flatten().astype(np.int32),
        "num_corners": int(len(cids)),
    }


def calculate_sharpness(gray):
    """Bildschaerfe ueber Laplacian-Varianz (hoeher = schaerfer)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def corner_coverage_fraction(charuco_corners, image_size):
    """
    Anteil der Bildflaeche, der von der Bounding-Box der erkannten Ecken
    abgedeckt wird. NUR als grobe Per-Bild-Metrik; die fuer die
    Kalibrierqualitaet entscheidende Groesse ist die AGGREGIERTE
    Eckenverteilung ueber ALLE Bilder (siehe coverage_heatmap_score).
    """
    pts = charuco_corners.reshape(-1, 2)
    if len(pts) == 0:
        return 0.0
    dx = pts[:, 0].max() - pts[:, 0].min()
    dy = pts[:, 1].max() - pts[:, 1].min()
    return float((dx * dy) / (image_size[0] * image_size[1]))


def coverage_heatmap_score(all_img_points, image_size, grid=(8, 6)):
    """
    Aggregierte Coverage: Wie viele Zellen eines grid-Rasters ueber dem
    gesamten Bild werden von mindestens einer erkannten Ecke getroffen?

    Das ist die wissenschaftlich relevante Coverage-Metrik fuer
    Intrinsics-Kalibrierung: Distortion und Hauptpunkt werden nur dort
    zuverlaessig geschaetzt, wo auch Messpunkte liegen -- besonders an den
    Bildraendern/-ecken. Rueckgabe in [0,1].
    """
    gx, gy = grid
    hit = np.zeros((gy, gx), dtype=bool)
    W, H = image_size
    for ip in all_img_points:
        pts = ip.reshape(-1, 2)
        cx = np.clip((pts[:, 0] / W * gx).astype(int), 0, gx - 1)
        cy = np.clip((pts[:, 1] / H * gy).astype(int), 0, gy - 1)
        hit[cy, cx] = True
    return float(hit.sum() / (gx * gy))


# ======================================================================
# 4. FEHLERMETRIKEN
# ======================================================================
def per_view_reprojection_errors(obj_points, img_points, rvecs, tvecs, K, D):
    """RMS-Reprojektionsfehler pro Ansicht (Pixel)."""
    errs = []
    for o, i, r, t in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        proj = proj.reshape(-1, 2)
        meas = i.reshape(-1, 2)
        e = np.sqrt(np.mean(np.sum((proj - meas) ** 2, axis=1)))
        errs.append(float(e))
    return np.array(errs)
