# Multi-Kamera-Kalibrierung – Dokumentation

Kalibrierungs-Pipeline für ein Drei-Kamera-System (XIMEA MC124CG-SY-UB) bestehend aus geführter Bildaufnahme, intrinsischer Monokalibrierung, paarweiser Stereokalibrierung, Validierung und globalem Bundle Adjustment.

---

## Hardware

| Eigenschaft | Wert |
|-------------|------|
| Kameramodell | XIMEA MC124CG-SY-UB |
| Auflösung | 4112 × 3008 px (Bildmitte: 2056 × 1504 px) |
| API | xiAPI V4.30.00.00 |
| CAM1 SN | CUCAU1829019 |
| CAM2 SN | CUCAU1829041 |
| CAM3 SN | CUCAU1829031 |
| Kalibrierboard | ChArUco |

---

## Pipeline-Übersicht

```
A2_capture_guided.py   →  Geführte Bildaufnahme (synchron)
B_mono_calib_all.py    →  Intrinsische Monokalibrierung (je Kamera)
C_stereo_all.py        →  Paarweise Stereokalibrierung
D_validate.py          →  Triangulationstest & Konsistenzprüfung
E_bundle_adjust.py     →  Globales Bundle Adjustment
```

---

## Schritt 1 – Bildaufnahme (`A2_capture_guided.py`)

Synchrone, geführte Aufnahme aller drei Kameras mit Coverage-Tracking.

**Steuerung:**

| Taste | Aktion |
|-------|--------|
| `p` | Alle Kameras aufnehmen |
| `6` / `7` / `8` | Einzelne Kamera aufnehmen |
| `2` | Paar CAM1 + CAM2 |
| `3` | Paar CAM1 + CAM3 |
| `q` | Beenden |

**Ziel:** Möglichst gleichmäßige Abdeckung des Bildbereichs (Coverage ≥ 55 % empfohlen). Aufgenommene Bildanzahl pro Kamera:

| Kamera | Aufnahmen |
|--------|-----------|
| CAM1 | 148 |
| CAM2 | 69 |
| CAM3 | 79 |

---

## Schritt 2 – Monokalibrierung (`B_mono_calib_all.py`)

Intrinsische Kalibrierung jeder Kamera einzeln mit dem Pinhole-Modell (5 Verzerrungsparameter: k₁, k₂, p₁, p₂, k₃). Iterative Ausreißerelimination in 3 Runden. Ergebnisse werden als `.pkl`-Dateien gespeichert.

**Ergebnisse:**

| Kamera | Views | RMS (px) | Coverage | fₓ (px) | fᵧ (px) | cₓ (px) | cᵧ (px) | Ausgabe |
|--------|-------|----------|----------|---------|---------|---------|---------|---------|
| CAM1 | 117 | 0,2500 | 96 % | 2299,8 | 2301,6 | 2086,3 | 1520,3 | `mono_cam1.pkl` |
| CAM2 | 61 | 0,2824 | 79 % | 2273,9 | 2273,3 | 2045,4 | 1503,8 | `mono_cam2.pkl` |
| CAM3 | 67 | 0,2697 | 75 % | 2286,5 | 2291,7 | 2109,8 | 1458,8 | `mono_cam3.pkl` |

**Verzerrungskoeffizienten D = [k₁, k₂, p₁, p₂, k₃]:**

| Kamera | k₁ | k₂ | p₁ | p₂ | k₃ |
|--------|----|----|----|----|-----|
| CAM1 | −0,00234 | −0,05508 | −0,00082 | 0,00041 | 0,01770 |
| CAM2 | −0,01370 | −0,04536 | −0,00028 | 0,00002 | 0,01451 |
| CAM3 | −0,01984 | −0,03215 | −0,00135 | 0,00112 | 0,00330 |

> **Hinweis:** Niedriger RMS allein ist kein Qualitätsbeweis. Auf ausreichende View-Zahl (≥ 12) und Coverage (≥ 55 %) achten. Alle drei Kameras erfüllen diese Kriterien.

---

## Schritt 3 – Stereokalibrierung (`C_stereo_all.py`)

Paarweise extrinsische Kalibrierung via `stereoCalibrate` auf Basis synchroner Bildpaare mit gemeinsamen ChArUco-Eckpunkten. CAM1 dient als Referenzkamera.

**Ergebnisse:**

| Paar | Paare gesamt | Paare genutzt | RMS (px) | Epipolarfehler Ø (px) | Epipolarfehler Median (px) | Baseline (m) | Drehwinkel | Ausgabe |
|------|-------------|---------------|----------|-----------------------|---------------------------|--------------|------------|---------|
| CAM1–CAM2 | 69 | 58 | 0,981 | 9,672 | 3,442 | 3,727 | 135,96° | `stereo_cam1_cam2.pkl` |
| CAM1–CAM3 | 79 | 64 | 0,547 | 1,604 | 0,810 | 2,307 | 71,11° | `stereo_cam1_cam3.pkl` |

**Translationsvektoren (CAM1-Ursprung):**

```
T(CAM1→CAM2) = [ 1.9016, -2.0370,  2.4745 ] m
T(CAM1→CAM3) = [-1.2905, -1.1566,  1.5218 ] m
```

> **Hinweis:** Der erhöhte mittlere Epipolarfehler bei CAM1–CAM2 (9,67 px vs. Median 3,44 px) deutet auf einzelne Ausreißer-Paare hin. Ursache ist vermutlich die große Baseline (3,73 m) in Kombination mit dem extremen Drehwinkel (136°).

---

## Schritt 4 – Validierung (`D_validate.py`)

Triangulation bekannter 3D-Punkte (ChArUco-Eckpunkte) und Vergleich mit Referenzmaßen. Da keine Aufnahmen existieren, in denen das Board gleichzeitig von CAM2 **und** CAM3 sichtbar ist, kann der indirekte CAM2–CAM3-Pfad nicht direkt validiert werden (→ siehe Bundle Adjustment).

**Triangulationstest:**

| Paar | Frames | Skalenfehler | Residuum Ø (mm) | Residuum Median (mm) | Residuum Max (mm) | Tiefe Z Ø (m) | Tiefenbereich (m) |
|------|--------|-------------|-----------------|----------------------|-------------------|---------------|-------------------|
| CAM1–CAM2 | 57 | 0,04 % | 0,80 | 0,74 | 1,66 | 2,80 | 0,91 – 4,36 |
| CAM1–CAM3 | 63 | 0,03 % | 0,81 | 0,69 | 2,74 | 2,60 | 0,80 – 3,70 |

**Mehrkamera-Konsistenz (abgeleitet, CAM1 als Hub):**

| Paar | Baseline (m) | Drehwinkel |
|------|-------------|------------|
| CAM1–CAM2 | 3,727 | 136,0° |
| CAM1–CAM3 | 2,306 | 71,1° |
| CAM2–CAM3 | 4,362 | 154,2° (verkettet) |

> Skalenfehler < 0,05 % und mittlere 3D-Residuen von ~0,8 mm bei ~2,7 m Arbeitstiefe entsprechen einer relativen Genauigkeit von ca. **1 : 3300**.

---

## Schritt 5 – Bundle Adjustment (`E_bundle_adjust.py`)

Gemeinsame Optimierung aller Kameraposen und Punktbeobachtungen (Levenberg-Marquardt, sparse). CAM1 als fester Ursprung.

**Eingabe:** 120 Aufnahmen, 15.412 Punktbeobachtungen (CAM2: 57, CAM3: 63 Frames)

| Metrik | Vorher | Nachher |
|--------|--------|---------|
| Globaler RMS | 2,6700 px | 0,6317 px |
| Kostenfunktion | 2,99 × 10⁴ | 4,82 × 10³ |
| Iterationen | – | 76 |

**Global optimierte Kamerageometrie:**

| Paar | Baseline (m) | Drehwinkel |
|------|-------------|------------|
| CAM1–CAM2 | 3,7277 | 135,97° |
| CAM1–CAM3 | 2,3074 | 71,12° |
| CAM2–CAM3 | 4,3632 | 154,14° (verkettet) |

> Die Optimierung wurde durch Erreichen des Iterationslimits (80 Funktionsauswertungen) beendet, nicht durch Divergenz. Der monotone Kostenabfall ohne Instabilitäten sowie die Übereinstimmung der optimierten Baselines mit den Stereokalibrierwerten (Abweichung < 0,3 %) bestätigen die globale Konsistenz des Kameranetzes.

---

## Ausgabedateien

| Datei | Inhalt |
|-------|--------|
| `mono_cam1.pkl` | Intrinsische Parameter CAM1 |
| `mono_cam2.pkl` | Intrinsische Parameter CAM2 |
| `mono_cam3.pkl` | Intrinsische Parameter CAM3 |
| `stereo_cam1_cam2.pkl` | Extrinsische Parameter CAM1–CAM2 |
| `stereo_cam1_cam3.pkl` | Extrinsische Parameter CAM1–CAM3 |

---

## Abhängigkeiten

- Python 3.11
- OpenCV (`cv2`)
- XIMEA xiAPI (`ximea`)
- NumPy, SciPy
- tqdm

Conda-Umgebung: `masterprojekt`

```bash
conda activate masterprojekt
```