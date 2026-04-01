"""
dual_realsense.py
-----------------
Simultaneously captures and displays RGB + Depth streams from two
Intel RealSense cameras in real time.

Requirements:
    pip install pyrealsense2 opencv-python numpy

Usage:
    python dual_realsense.py

Controls:
    Q / ESC  →  quit
    S        →  save a snapshot of both cameras
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 640, 480
FPS = 30
DEPTH_COLORMAP = cv2.COLORMAP_JET
WINDOW_NAME = "Dual RealSense — Q to quit | S to snapshot"
SNAPSHOT_DIR = Path("snapshots")
# ─────────────────────────────────────────────────────────────────────────────


class RealSenseCamera:
    """Wraps a single RealSense pipeline and grabs frames in a background thread."""

    def __init__(self, serial: str, label: str):
        self.serial = serial
        self.label = label

        self.color_frame: np.ndarray | None = None
        self.depth_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None          # expose thread errors to main

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8,  FPS)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,   FPS)

        profile = self.pipeline.start(config)
        self._align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        print(f"[{label}] Started  serial={serial}  depth_scale={self.depth_scale:.6f}")

    # ── Background capture thread ─────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name=f"capture-{self.label}")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.pipeline.stop()
        print(f"[{self.label}] Stopped")

    def _capture_loop(self):
        """
        Capture loop running on a dedicated thread.

        Key fixes vs original:
          * np.array(...) instead of np.asanyarray(...)
            np.asanyarray returns a *view* into the SDK's buffer; once the
            frameset leaves scope the buffer is reclaimed and the array points
            at garbage — causing the frozen-first-frame symptom.
            np.array() always makes an owned copy.
          * try_wait_for_frames instead of wait_for_frames so a slow camera
            doesn't block its own thread indefinitely.
          * Full try/except so errors are logged and the thread keeps retrying
            rather than dying silently after the first frame.
        """
        consecutive_misses = 0
        while self._running:
            try:
                success, frames = self.pipeline.try_wait_for_frames(timeout_ms=200)
                if not success:
                    consecutive_misses += 1
                    if consecutive_misses % 20 == 0:        # log every ~4 s
                        print(f"[{self.label}] WARNING: no frame for "
                              f"{consecutive_misses * 0.2:.1f}s — check USB port")
                    continue

                consecutive_misses = 0
                aligned = self._align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # ── CRITICAL FIX: np.array() copies the buffer ──────────────
                # np.asanyarray() only creates a *view*.  When `aligned` goes
                # out of scope the SDK reclaims the buffer and cam-B freezes on
                # its first frame while cam-A coincidentally keeps working.
                color     = np.array(color_frame.get_data())      # (H, W, 3) uint8
                depth_raw = np.array(depth_frame.get_data())      # (H, W)    uint16

                # Colourize depth for display
                depth_m    = depth_raw * self.depth_scale
                depth_norm = cv2.normalize(depth_m, None, 0, 255,
                                           cv2.NORM_MINMAX, cv2.CV_8U)
                depth_col  = cv2.applyColorMap(depth_norm, DEPTH_COLORMAP)

                with self._lock:
                    self.color_frame = color
                    self.depth_frame  = depth_col
                    self.error = None

            except Exception as exc:
                print(f"[{self.label}] ERROR in capture loop: {exc}")
                with self._lock:
                    self.error = str(exc)
                time.sleep(0.1)

    # ── Thread-safe frame getters ─────────────────────────────────────────────

    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            c = self.color_frame.copy() if self.color_frame is not None else None
            d = self.depth_frame.copy() if self.depth_frame is not None else None
        return c, d

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ── Helpers ───────────────────────────────────────────────────────────────────

def discover_serials() -> list[str]:
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]


def overlay_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def waiting_placeholder(label: str) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(img, f"{label}: waiting for frames...",
                (20, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def build_grid(cams: list[RealSenseCamera]) -> np.ndarray:
    rows = []
    for cam in cams:
        color, depth = cam.get_frames()
        color = overlay_label(color, f"{cam.label} — RGB")   if color is not None \
                else waiting_placeholder(f"{cam.label} RGB")
        depth = overlay_label(depth, f"{cam.label} — Depth") if depth is not None \
                else waiting_placeholder(f"{cam.label} Depth")
        rows.append(np.hstack([color, depth]))
    return np.vstack(rows)


def save_snapshot(cams: list[RealSenseCamera]):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for cam in cams:
        color, depth = cam.get_frames()
        if color is not None:
            cv2.imwrite(str(SNAPSHOT_DIR / f"{ts}_{cam.label}_color.png"), color)
        if depth is not None:
            cv2.imwrite(str(SNAPSHOT_DIR / f"{ts}_{cam.label}_depth.png"), depth)
    print(f"[snapshot] Saved to {SNAPSHOT_DIR}/ with prefix {ts}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    serials = discover_serials()
    print(f"Found {len(serials)} RealSense device(s): {serials}")

    if len(serials) < 2:
        raise RuntimeError(
            f"Need at least 2 RealSense cameras, found {len(serials)}.\n"
            "Tip: each camera needs its own USB 3.x port on a separate "
            "host controller for reliable dual streaming."
        )

    # Stagger initialisation slightly — prevents USB enumeration conflicts
    cams: list[RealSenseCamera] = []
    for i, serial in enumerate(serials[0:2]):
        cam = RealSenseCamera(serial, f"CAM-{'AB'[i]}")
        cams.append(cam)
        time.sleep(0.5)

    for cam in cams:
        cam.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("Streaming … press Q or ESC to quit, S to snapshot.")

    try:
        while True:
            cv2.imshow(WINDOW_NAME, build_grid(cams))

            for cam in cams:
                if not cam.is_alive():
                    print(f"[{cam.label}] WARNING: capture thread has died!")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                save_snapshot(cams)

    finally:
        for cam in cams:
            cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()