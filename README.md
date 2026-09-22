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
| `6` / `7` / `8` / `9` | Einzelne Kamera aufnehmen |
| `2` | Paar CAM1 + CAM2 |
| `3` | Paar CAM1 + CAM3 |
| `4` | Paar CAM1 + CAM4 |

| `q` | Beenden |

**Ziel:** Möglichst gleichmäßige Abdeckung des Bildbereichs (Coverage ≥ 55 % empfohlen). Aufgenommene Bildanzahl pro Kamera:

| Kamera | Aufnahmen | End-Coverrage |
|--------|-----------|---------------|
| CAM1 | 59 | 100% |
| CAM2 | 23 | 100% |
| CAM3 | 28 | 96% |
| CAM4 | 29 | 77% |

---

## Schritt 2 – Monokalibrierung (`B_mono_calib_all.py`)

Intrinsische Kalibrierung jeder Kamera einzeln mit dem Pinhole-Modell (5 Verzerrungsparameter: k₁, k₂, p₁, p₂, k₃). Iterative Ausreißerelimination in 3 Runden. Ergebnisse werden als `.pkl`-Dateien gespeichert.

**Ergebnisse:**

| Kamera | Views | RMS (px) | Coverage | fₓ (px) | fᵧ (px) | cₓ (px) | cᵧ (px) | Ausgabe |
|--------|-------|----------|----------|---------|---------|---------|---------|---------|
| CAM1 | 37 | 0,2944 | 96 % | 2291,9 | 2294,3 | 2074,9 | 1528.4 | `mono_cam1.pkl` |
| CAM2 | 19 | 0,2678 | 83 % | 2286,9 | 2278,5 | 2102,9 | 1501,3 | `mono_cam2.pkl` |
| CAM3 | 26 | 0,4187 | 96 % | 2264,4 | 2264,6 | 2056,2 | 1505,5 | `mono_cam3.pkl` |
| CAM4 | 29 | 0,2218 | 77 % | 1750,6 | 1748,4 | 2056,7 | 1508,7 | `mono_cam4.pkl` |

**Verzerrungskoeffizienten D = [k₁, k₂, p₁, p₂, k₃]:**

| Kamera | k₁ | k₂ | p₁ | p₂ | k₃ |
|--------|----|----|----|----|-----|
| CAM1 | -0,01233 | −0,0435 | 0,00059 | 0,0009 | 0,01221 |
| CAM2 | −0,01663 | −0,04106 | 0,00027 | 0,00088 | 0,01097 |
| CAM3 | −0,01811 | −0,04185 | 0,00005 | -0,00046 | 0,01282 |
| CAM4 | −0,2147 | −0,1327 | −0,00091 | -0,00024 | -0,02823 |

> **Hinweis:** Niedriger RMS allein ist kein Qualitätsbeweis. Auf ausreichende View-Zahl (≥ 12) und Coverage (≥ 55 %) achten. Alle drei Kameras erfüllen diese Kriterien.

---

## Schritt 3 – Stereokalibrierung (`C_stereo_all.py`)

Paarweise extrinsische Kalibrierung via `stereoCalibrate` auf Basis synchroner Bildpaare mit gemeinsamen ChArUco-Eckpunkten. CAM1 dient als Referenzkamera.

**Ergebnisse:**

| Paar | Paare gesamt | Paare genutzt | RMS (px) | Epipolarfehler Ø (px) | Epipolarfehler Median (px) | Baseline (m) | Drehwinkel | Ausgabe |
|------|-------------|---------------|----------|-----------------------|---------------------------|--------------|------------|---------|
| CAM1–CAM2 | 17 | 14 | 0,7388 | 5,6138 | 3,1087 | 2,3112 | 71,04° | `stereo_cam1_cam2.pkl` |
| CAM1–CAM3 | 23 | 14 | 1,0350 | 7,0738 | 3,6435 | 3,6986 | 136,34° | `stereo_cam1_cam3.pkl` | 
| CAM1–CAM4 | 16 | 7 | 0,4936 | 23,9960 | 17,6794 | 3,3718 | 128,86° | `stereo_cam1_cam4.pkl` |

**Translationsvektoren (CAM1-Ursprung):**

```
T(CAM1→CAM2) = [ 1.9016, -2.0370,  2.4745 ] m
T(CAM1→CAM3) = [-1.2905, -1.1566,  1.5218 ] m 
T(CAM1→CAM4) = [-1.1901, -1.7608,  2.6176 ] m
```


> **Hinweis:** Der erhöhte mittlere Epipolarfehler bei CAM1–CAM2 (9,67 px vs. Median 3,44 px) deutet auf einzelne Ausreißer-Paare hin. Ursache ist vermutlich die große Baseline (3,73 m) in Kombination mit dem extremen Drehwinkel (136°).

---

## Schritt 4 – Validierung (`D_validate.py`)

Triangulation bekannter 3D-Punkte (ChArUco-Eckpunkte) und Vergleich mit Referenzmaßen. Da keine Aufnahmen existieren, in denen das Board gleichzeitig von CAM2 **und** CAM3 sichtbar ist, kann der indirekte CAM2–CAM3-Pfad nicht direkt validiert werden (→ siehe Bundle Adjustment).

**Triangulationstest:**

| Paar | Frames | Skalenfehler | Residuum Ø (mm) | Residuum Median (mm) | Residuum Max (mm) | Tiefe Z Ø (m) | Tiefenbereich (m) |
|------|--------|-------------|-----------------|----------------------|-------------------|---------------|-------------------|
| CAM1–CAM2 | 14 | 0,04 % | 0,64 | 0,56 | 1,10 | 2,39 | 0,58 – 3,34 |
| CAM1–CAM3 | 14 | 0,01 % | 0,93 | 0,81 | 1,81 | 1,91 | 0,89 – 4,58 |
| CAM1–CAM4 | 7 | 0,02 % | 0,82 | 0,81 | 1,20 | 3,23 | 1,98 – 4,31 |

**Mehrkamera-Konsistenz (abgeleitet, CAM1 als Hub):**

| Paar | Baseline (m) | Drehwinkel |
|------|-------------|------------|
| CAM1–CAM2 | 3,727 | 71,0° |
| CAM1–CAM3 | 2,306 | 136,3° |
| CAM1–CAM4 | 3,372 | 128,9° |
| CAM2–CAM3 | 4,362 | 153,7° (verkettet) |
| CAM2–CAM4 | 4,362 | 58,6° (verkettet) |
| CAM3–CAM4 | 4,362 | 95,3° (verkettet) |

> Skalenfehler < 0,03 % und mittlere 3D-Residuen von ~0,8 mm bei ~2,5 m Arbeitstiefe entsprechen einer relativen Genauigkeit von ca. **1 : 3125** (0,8 : 2500).

---

## Schritt 5 – Bundle Adjustment (`E_bundle_adjust.py`)

Gemeinsame Optimierung aller Kameraposen und Punktbeobachtungen (Levenberg-Marquardt, sparse). CAM1 als fester Ursprung.

**Eingabe:** 35 Aufnahmen, 4.624 Punktbeobachtungen (CAM2: 14, CAM3: 14, CAM4: 7 Frames)

| Metrik | Vorher | Nachher |
|--------|--------|---------|
| Globaler RMS | 2,2723 px | 0,6131 px |
| Kostenfunktion | 8,6327 × 10³ | 1,3876 × 10³ |
| Iterationen | – | 78 |

**Global optimierte Kamerageometrie:**

| Paar | Baseline (m) | Drehwinkel |
|------|-------------|------------|
| CAM1–CAM2 | 3,7277 | 71,04° |
| CAM1–CAM3 | 2,3074 | 136,3° |
| CAM1–CAM4 | 3,3728 | 128,90° |
| CAM2–CAM3 | 4,3632 | 154,14° (verkettet) |
| CAM2–CAM4 | 4,3632 | 58,60° (verkettet) |
| CAM3–CAM4 | 4,3632 | 95,30° (verkettet) |

> Die Optimierung wurde durch Erreichen des Iterationslimits (80 Funktionsauswertungen) beendet, nicht durch Divergenz. Der monotone Kostenabfall ohne Instabilitäten sowie die Übereinstimmung der optimierten Baselines mit den Stereokalibrierwerten (Abweichung < 0,3 %) bestätigen die globale Konsistenz des Kameranetzes.

---

## Ausgabedateien

| Datei | Inhalt |
|-------|--------|
| `mono_cam1.pkl` | Intrinsische Parameter CAM1 |
| `mono_cam2.pkl` | Intrinsische Parameter CAM2 |
| `mono_cam3.pkl` | Intrinsische Parameter CAM3 |
| `mono_cam4.pkl` | Intrinsische Parameter CAM4 |
| `stereo_cam1_cam2.pkl` | Extrinsische Parameter CAM1–CAM2 |
| `stereo_cam1_cam3.pkl` | Extrinsische Parameter CAM1–CAM3 |
| `stereo_cam1_cam4.pkl` | Extrinsische Parameter CAM1–CAM4 |

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