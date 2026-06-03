import sys
import time
import os
import cv2
from ximea import xiapi

# --- Globale Kamera-Einstellungen ---
EXPOSURE_US = 30000      
GAIN = 5.0
WB_RED = 1.75
WB_GREEN = 1.0
WB_BLUE = 2.25

# Anzeigegröße für OpenCV (NUR für das Live-Fenster)
DISPLAY_SIZE = (800, 600) 

def setup_camera_full_res(cam, name):
    try:
        print(f"\n[INFO] Öffne {name}...")
        cam.open_device()
        device_sn = cam.get_device_sn().decode()
        print(f"[OK] {name} geöffnet. SN: {device_sn}")

        # Bildformat RGB24 für volle Farbtiefe
        try:
            cam.set_imgdataformat('XI_IMG_FORMAT_RGB24')
        except Exception:
            cam.set_param('imgdataformat', 'XI_RGB24')

        # --- Volle Auflösung sicherstellen ---
        cam.set_offsetX(0)
        cam.set_offsetY(0)
        
        # Downsampling explizit deaktivieren (1x1)
        try:
            cam.set_downsampling('XI_DWN_1x1')
            print(f"[{name}] Full Resolution aktiv.")
        except Exception:
            print(f"[INFO] {name}: Nutze Standardauflösung.")

        cam.set_exposure(EXPOSURE_US)
        cam.set_gain(GAIN)
        cam.disable_auto_wb()
        cam.set_wb_kr(WB_RED)
        cam.set_wb_kg(WB_GREEN)
        cam.set_wb_kb(WB_BLUE)

        cam.start_acquisition()
        return True, device_sn

    except Exception as e:
        print(f"[FEHLER] Konnte {name} nicht initialisieren: {e}")
        return False, None

def main():
    cam1, cam2, cam3 = xiapi.Camera(0), xiapi.Camera(1), xiapi.Camera(2)
    img1, img2, img3 = xiapi.Image(), xiapi.Image(), xiapi.Image()
    
    success1, sn1 = setup_camera_full_res(cam1, "Cam1")
    success2, sn2 = setup_camera_full_res(cam2, "Cam2")
    success3, sn3 = setup_camera_full_res(cam3, "Cam3")

    if not any([success1, success2, success3]):
        print("\n[FEHLER] Keine Kameras gefunden.")
        return
    
    dirs = ["calib_1_2", "calib_1_3", "calib_single_1", "calib_single_2", "calib_single_3"]
    for d in dirs: os.makedirs(d, exist_ok=True)

    print("\n[STEUERUNG] 'p': Alle speichern | '6','7','8': Einzeln | '2','3': Stereo | 'q': Beenden")

    while True:
        frames = {1: None, 2: None, 3: None}
        
        for i, (cam, img, success) in enumerate(zip([cam1, cam2, cam3], [img1, img2, img3], [success1, success2, success3]), 1):
            if success:
                try:
                    cam.get_image(img)
                    # Rohdaten in Originalgröße holen
                    raw_frame = img.get_image_data_numpy()
                    
                    frames[i] = raw_frame
                    
                    # Hier kannst du rotieren, ohne dass es das gespeicherte Bild beeinflusst
                    preview_frame = raw_frame.copy()
                    
                    preview_res = cv2.resize(preview_frame, DISPLAY_SIZE)
                    curr_sn = sn1 if i==1 else (sn2 if i==2 else sn3)
                    cv2.imshow(f"Cam {i} (SN: {curr_sn})", preview_res)
                    
                except Exception as e:
                    print(f"Fehler Cam {i}: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # --- Speichern (Nutzt frames[i], was das unveränderte Original-Numpy-Array ist) ---
        if key == ord('p'):
            for i in [1,2,3]:
                if frames[i] is not None:
                    cv2.imwrite(f"calib_single_{i}/cam{i}_{timestamp}.png", frames[i])
            print(f"[OK] Alle Originalbilder gespeichert ({timestamp})")

        elif key == ord('6'):
            if frames[1] is not None:
                cv2.imwrite(f"calib_single_1/cam1_{timestamp}.png", frames[1])
                print("[OK] Cam 1 Original gespeichert")
        
        elif key == ord('7'):
            if frames[2] is not None:
                cv2.imwrite(f"calib_single_2/cam2_{timestamp}.png", frames[2])
                print("[OK] Cam 2 Original gespeichert")
        
        elif key == ord('8'):
            if frames[3] is not None:
                cv2.imwrite(f"calib_single_3/cam3_{timestamp}.png", frames[3])
                print("[OK] Cam 3 Original gespeichert")

        elif key == ord('2'):
            if frames[1] is not None and frames[2] is not None:
                cv2.imwrite(f"calib_1_2/cam1_{timestamp}.png", frames[1])
                cv2.imwrite(f"calib_1_2/cam2_{timestamp}.png", frames[2])
                print("[OK] Stereo 1+2 Original gespeichert")

        elif key == ord('3'):
            if frames[1] is not None and frames[3] is not None:
                cv2.imwrite(f"calib_1_3/cam1_{timestamp}.png", frames[1])
                cv2.imwrite(f"calib_1_3/cam3_{timestamp}.png", frames[3])
                print("[OK] Stereo 1+3 Original gespeichert")

    for c, s in zip([cam1, cam2, cam3], [success1, success2, success3]):
        if s: 
            c.stop_acquisition()
            c.close_device()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()