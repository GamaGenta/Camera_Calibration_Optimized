# Wissenschaftliche Analyse der Kalibrierungskette

Ziel: hochpräzises Markerless-Motion-Capture mit 3 fest montierten Deckenkameras.
Bewertet wird die gesamte Kette: Bildaufnahme → ChArUco-Detektion → Mono → Stereo →
Multi-Cam-Konsistenz → Triangulation.

---

## 0. Kernbefund (Executive Summary)

**Die Hauptfehlerquelle liegt in der ChArUco-Detektion, nicht in der Kalibrierungsmathematik.**

Die OpenCV-**Default-`DetectorParameters` sind für 4112×3008-Bilder eines teils weit
entfernten/gekippten Boards völlig unzureichend.** Dadurch wurden über die Hälfte aller
Kalibrierbilder **stillschweigend verworfen**:

| Kamera | Bilder verfügbar | nutzbar (alt) | nutzbar (neu) |
|--------|------------------|---------------|---------------|
| Cam1   | 55               | 35            | 54            |
| Cam2   | 55               | **3** ⚠️       | 49            |
| Cam3   | 59               | 12            | 57            |

Folgen der alten Pipeline:
- **Cam2 wurde mit nur 3 Ansichten kalibriert.** Distortion-Koeffizienten waren reiner
  Overfit: `k2 = −0.54, k3 = +1.17` (physikalisch unplausibel; Cam1/Cam3 ≈ ±0.03).
- Die Intrinsics waren **trotz RMS < 0.3 px um hunderte Pixel falsch**
  (z. B. Cam1 `cy`: 1539 → 1349, Δ ≈ 190 px; `fx`: 2275 → 2435, Δ ≈ 160 px).

> **Zentrale Lektion:** Ein niedriger RMS-Reprojektionsfehler ist **kein** Qualitätsbeweis.
> Bei zu wenigen oder geometrisch einseitigen Ansichten ist er ein Overfitting-Artefakt.

---

## 1. Bildaufnahme (`A_ThreeCameraStreamTakePics.py`)

**Befunde:**
1. **Keine Hardware-Synchronisation.** Die drei Kameras werden sequentiell in der Loop
   gegrabbt. Für ein *statisches* Kalibrierboard tolerierbar, für späteres MoCap bewegter
   Personen **kritisch** – hier sind Hardware-Trigger/Genlock zwingend.
2. **Sekunden-Zeitstempel** (`%Y%m%d_%H%M%S`): zwei Aufnahmen in derselben Sekunde
   überschreiben sich.
3. **Es werden nur die Paare 1-2 und 1-3 aufgenommen, nie 2-3 und nie alle drei
   gleichzeitig.** → Der Kalibriergraph hat keine direkte 2-3-Kante (siehe §5).
4. Belichtung 30 ms + handgeführtes Board ⇒ Bewegungsunschärfe ist ein Mit-Grund für die
   vielen Null-Detektionen. Empfehlung: Board still halten oder Belichtung senken/Blitz.

---

## 2. ChArUco-Detektion (Wurzel des Problems)

`legacy=True` ist **korrekt** (empirisch bestätigt: liefert ~3× mehr Detektionen als
`legacy=False`; das physische Board ist ein Legacy-Board).

Der Fix liegt in den Detektor-Parametern (siehe `calib_common.make_detector`):
- `cornerRefinementMethod = CORNER_REFINE_SUBPIX` (Subpixel-Genauigkeit, vorher aus)
- `adaptiveThreshWinSizeMax 23 → 53` (große Bilder, ungleichmäßiges Deckenlicht)
- `minMarkerPerimeterRate 0.03 → 0.01` (erlaubt die kleinen Marker entfernter Boards)

Empirisch (Cam2, 30 Bilder): Detektion **10/30 → 27/30**.

---

## 3. Monokalibrierung (`B_*` → ersetzt durch `B_mono_calib_all.py`)

**Befunde am alten Code:**
- Stilles Verwerfen ohne Warnung bei zu wenigen Views.
- Fallback „<10 Views ⇒ nimm die 15 mit den meisten Ecken" wählt die **frontalsten,
  nächsten** Boards → **minimale Pose-Diversität** = schlechteste Wahl für Kalibrierung.
- Coverage pro Einzelbild gemessen; relevant ist aber die **aggregierte** Eckenverteilung
  über alle Bilder (besonders Randabdeckung für Distortion/Hauptpunkt).
- Doppelte Funktionsdefinition in `B_Cam2` (harmlos, aber Codequalität).
- `getChessboardCorners()[ids]` statt `board.matchImagePoints()`.

**Distortion-Modell:** Rational/Thin-Prism/Tilted bringen **nichts** (RMS 0.1250 → 0.1248)
und erhöhen nur das Overfitting-Risiko. **Das Standard-5-Parameter-Pinhole-Modell ist
korrekt.** Mehr Parameter sind erst sinnvoll, wenn Datenmenge/-abdeckung deutlich größer ist.

**Neue Mono-Pipeline:**
- Getunte Detektion, `matchImagePoints`.
- Harte Mindest-View-Zahl (≥12) + aggregierte Coverage-Heatmap **mit Warnungen**.
- Iterative robuste Re-Kalibrierung (3σ-Ausreißer-Entfernung).
- Datengestützter Modellvergleich (5-param vs. rational).
- Speichert Roh-Detektionen für das Bundle Adjustment.

---

## 4. Stereokalibrierung (`C_*` → ersetzt durch `C_stereo_all.py`)

**Befunde am alten Code:**
- `CALIB_FIX_INTRINSIC` ist **richtig** – aber es fixierte die *falschen* (overfit)
  Intrinsics. Mit sauberen Mono-Werten wird dieser Schritt erst valide.
- Bewertung nur über `stereoCalibrate`-RMS. Aussagekräftiger ist der
  **symmetrische Epipolarfehler** (physikalisch relevant für Triangulation) – jetzt ergänzt.
- `stereoRectify` mit `alpha=0` ist für Triangulation unnötig (Rektifizierung wird für
  Dense-Disparity gebraucht, nicht für die spätere Punkt-Triangulation). Kommentar im
  Originalcode („ULI FRAGEN") bestätigt die Unsicherheit – für MoCap nicht erforderlich.

---

## 5. Multi-Cam-Konsistenz (`D_validate.py`)

Gefordert: Vergleich `A→C` direkt vs. `A→B→C` verkettet.

**Strukturelles Problem:** Es gibt **keine direkte 2-3-Beobachtung** und **keine Aufnahme mit
allen drei Kameras gleichzeitig**. Cam2↔Cam3 kann daher nur über Cam1 *verkettet*, aber
**nicht unabhängig validiert** werden. Das ist eine Lücke im Aufnahmeprotokoll, kein
Rechenfehler.

→ **Dringende Empfehlung:** Aufnahmeserie nachholen, in der das Board gleichzeitig von
**Cam2 und Cam3** (idealerweise allen dreien) gesehen wird. Erst dann ist eine echte
Loop-Closure-Konsistenzprüfung und ein vollwertiges 3-View-Bundle-Adjustment möglich.

---

## 6. Globale Optimierung (`E_bundle_adjust.py`)

Paarweise Stereokalibrierung ist für 3 Kameras suboptimal (jedes Paar isoliert minimiert).
Implementiert ist ein **globales Bundle Adjustment** (Cam1 als Anker), das den
Gesamt-Reprojektionsfehler über alle Kameras und Aufnahmen gemeinsam minimiert
(scipy `least_squares`, sparse Jacobian, Huber-Loss). Cam1 verankert das System, weil es in
jeder Aufnahme sichtbar ist.

Grenze (s. §5): Ohne 2-3-Beobachtungen bleibt Cam2↔Cam3 eine konsistent gemachte
Verkettung. Mit nachgeholten Daten wird das BA das gesamte Rig echt global lösen.

---

## 7. Triangulations-Validierung (`D_validate.py`)

Für MoCap entscheidend, nicht der Reprojektionsfehler. Der Test trianguliert die in beiden
Kameras sichtbaren ChArUco-Ecken und vergleicht mit der bekannten Board-Geometrie:
- **Skalenfehler (%)** → metrische Korrektheit (hängt direkt an `SQUARE_LENGTH`!)
- **3D-Residuum (mm)** nach starrer Ausrichtung → Triangulationsgenauigkeit
- **rekonstruierte Tiefe Z (m)** → Arbeitsbereich

> Prüfe `SQUARE_LENGTH`/`MARKER_LENGTH` mit dem Messschieber am realen Druck. Ein 2%-iger
> Skalenfehler im Board überträgt sich 1:1 in jede 3D-Position.

---

## Empfohlene Ausführungsreihenfolge

```bash
python B_mono_calib_all.py     # Mono, alle 3 Kameras (mit Diagnostik)
python C_stereo_all.py         # Stereo 1-2 und 1-3
python D_validate.py           # Triangulation + Konsistenz
python E_bundle_adjust.py      # globale Optimierung
```

## Wichtigste Maßnahmen (Priorität)

1. **Sofort:** neue Detektor-Parameter verwenden, alles neu kalibrieren. (größter Effekt)
2. **Daten:** Aufnahmen mit Board gleichzeitig in Cam2+Cam3 / allen dreien nachholen;
   mehr Tilt-/Rand-Abdeckung für Distortion & Hauptpunkt.
3. **`SQUARE_LENGTH` am Druck nachmessen** (metrische Skala).
4. **MoCap-Betrieb:** Hardware-Synchronisation der Kameras.
5. Global per Bundle Adjustment lösen statt rein paarweise.
