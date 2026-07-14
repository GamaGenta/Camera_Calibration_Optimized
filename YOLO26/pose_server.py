"""
pose_server.py  –  WebSocket-Server für 3D-Pose-Daten
------------------------------------------------------
Startet einen WebSocket-Server auf ws://localhost:8765

Integration in Pose_Estimation.py:
  1. Dieses Skript parallel starten: python pose_server.py
  2. In Pose_Estimation.py die Funktion send_pose_data() aufrufen
     (siehe Kommentar "INTEGRATION" weiter unten)
"""

import asyncio
import json
import websockets
from websockets.server import serve

CLIENTS: set = set()


async def handler(websocket):
    """Neue Browser-Verbindung registrieren."""
    CLIENTS.add(websocket)
    print(f"[+] Client verbunden  ({len(CLIENTS)} gesamt)")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"[-] Client getrennt  ({len(CLIENTS)} gesamt)")


async def broadcast(message: str):
    """Nachricht an alle verbundenen Clients senden."""
    if CLIENTS:
        await asyncio.gather(
            *[ws.send(message) for ws in CLIENTS],
            return_exceptions=True,
        )


# ── Globale Event-Loop-Referenz (wird von send_pose_data genutzt) ──────────────
_loop: asyncio.AbstractEventLoop | None = None


def send_pose_data(points_3d: list, fps: float = 0.0):
    """
    Thread-sichere Hilfsfunktion – aus dem Haupt-Thread von Pose_Estimation.py
    aufrufen.

    points_3d : Liste mit 17 Einträgen (np.ndarray([X,Y,Z]) oder None)
    fps       : aktuelle Framerate (optional, für das HUD)

    INTEGRATION – in Pose_Estimation.py:
    ─────────────────────────────────────
    # Am Anfang importieren:
    import threading, pose_server

    # Vor der main()-Schleife einmalig starten:
    server_thread = threading.Thread(target=pose_server.run_server, daemon=True)
    server_thread.start()

    # In der Hauptschleife nach der Triangulation:
    pose_server.send_pose_data(points_3d, fps=fps)
    """
    if _loop is None:
        return
    payload = {
        "fps": round(fps, 1),
        "keypoints": [
            pt.tolist() if pt is not None else None
            for pt in points_3d
        ],
    }
    asyncio.run_coroutine_threadsafe(
        broadcast(json.dumps(payload)), _loop
    )


def run_server(host: str = "localhost", port: int = 8765):
    """
    Startet den WebSocket-Server (blockierend – in eigenem Thread aufrufen).
    """
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    async def _start():
        print(f"[WebSocket] Server läuft auf ws://{host}:{port}")
        async with serve(handler, host, port):
            await asyncio.Future()   # läuft bis zum Prozessende

    _loop.run_until_complete(_start())


# ── Demo-Modus: Skript direkt starten ──────────────────────────────────────────
if __name__ == "__main__":
    import math, time, threading, numpy as np

    # Synthetische Skeleton-Animation für Tests ohne echte Kameras
    def demo_loop():
        t = 0.0
        while True:
            t += 0.05
            # Einfaches Pendel-Skeleton (Y = oben)
            pts = [None] * 17
            # Nase
            pts[0]  = np.array([0.0,      1.70 + 0.02 * math.sin(t),       0.0])
            # Schultern
            pts[5]  = np.array([-0.20,    1.45,                              0.0])
            pts[6]  = np.array([ 0.20,    1.45,                              0.0])
            # Ellbogen
            pts[7]  = np.array([-0.35,    1.20 + 0.10 * math.sin(t),        0.0])
            pts[8]  = np.array([ 0.35,    1.20 + 0.10 * math.sin(t + 1.0),  0.0])
            # Handgelenke
            pts[9]  = np.array([-0.40,    0.95 + 0.18 * math.sin(t + 0.5),  0.0])
            pts[10] = np.array([ 0.40,    0.95 + 0.18 * math.sin(t + 1.5),  0.0])
            # Hüften
            pts[11] = np.array([-0.12,    1.00,                              0.0])
            pts[12] = np.array([ 0.12,    1.00,                              0.0])
            # Knie
            pts[13] = np.array([-0.13,    0.55 + 0.05 * math.sin(t * 0.7),  0.0])
            pts[14] = np.array([ 0.13,    0.55 + 0.05 * math.sin(t * 0.7 + 1.0), 0.0])
            # Knöchel
            pts[15] = np.array([-0.14,    0.10,                              0.0])
            pts[16] = np.array([ 0.14,    0.10,                              0.0])

            send_pose_data(pts, fps=20.0)
            time.sleep(0.05)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(0.3)  # kurz warten bis Loop bereit

    print("[Demo] Synthetisches Skeleton wird gesendet – öffne pose_viewer.html")
    demo_loop()
