"""
Load a colored PLY (e.g. from da3_pointcloud.py), rasterize a pinhole depth map, and visualize
in Rerun using the same entities as render_3d.py (world/camera, world/camera/image,
world/object/points1).

**Coordinates:** same assumptions as ``render_3d.py`` (pinhole + Z band). Fix upside-down clouds by
exporting with ``da3_pointcloud`` ``--flip-z`` / ``--no-flip-z`` (coordinate flip only), not by
disabling depth clipping here.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import numpy as np
import rerun as rr

from utils import K as K_DEFAULT


# Match render_3d.py
IMG_PLANE_DIST = 0.25
K_NOMINAL_W = 1920
K_NOMINAL_H = 1080


def scale_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Assume K is calibrated for K_NOMINAL_W x K_NOMINAL_H (see utils.py)."""
    Ks = K.astype(np.float64).copy()
    sx = width / float(K_NOMINAL_W)
    sy = height / float(K_NOMINAL_H)
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks


def load_ply_xyz_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """ASCII PLY with x,y,z and uchar r,g,b (same layout as da3_pointcloud.save_ply_ascii)."""
    positions: list = []
    colors: list = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
        if "ply" not in line.lower():
            raise ValueError(f"Not an ASCII PLY header: {path}")
        n_vert = 0
        props: list[str] = []
        while True:
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if parts and parts[0] == "element" and len(parts) >= 3 and parts[1] == "vertex":
                n_vert = int(parts[2])
            elif parts and parts[0] == "property":
                props.append(parts[-1])
            elif line.strip() == "end_header":
                break
        need = {"x", "y", "z", "red", "green", "blue"}
        if not need.issubset(set(props)):
            raise ValueError(
                f"PLY must have properties x,y,z,red,green,blue; got {props}. "
                "Export with da3_pointcloud.py or equivalent."
            )
        for _ in range(n_vert):
            line = f.readline()
            if not line:
                break
            vals = line.split()
            if len(vals) < 6:
                continue
            positions.append([float(vals[0]), float(vals[1]), float(vals[2])])
            colors.append([int(vals[3]), int(vals[4]), int(vals[5])])
    pos = np.asarray(positions, dtype=np.float32)
    col = np.asarray(colors, dtype=np.uint8)
    return pos, col


def rescale_depth_vis(d: np.ndarray) -> np.ndarray:
    """Same idea as render_3d rescale: normalize to 0–255 for display."""
    x = d.copy().astype(np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(d, dtype=np.uint8)
    lo = np.nanmin(x[finite])
    hi = np.nanmax(x[finite])
    if hi <= lo:
        return np.zeros_like(d, dtype=np.uint8)
    x = (x - lo) / (hi - lo) * 255.0
    x[~finite] = 0
    return np.clip(x, 0, 255).astype(np.uint8)


def rasterize_depth_map(
    points_cam: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pinhole camera at origin, OpenCV convention (x right, y down, z forward).
    points_cam: (N, 3). Returns depth (H,W) nan where empty, rgb (H,W,3) uint8.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = (
        np.isfinite(z)
        & (z > 1e-6)
        & (z >= z_min)
        & (z <= z_max)
    )
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    depth = np.full((height, width), np.nan, dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    order = np.argsort(z)
    for i in order:
        if not valid[i]:
            continue
        vi, ui = int(v[i]), int(u[i])
        if np.isnan(depth[vi, ui]):
            depth[vi, ui] = z[i]
            rgb[vi, ui] = colors[i]

    return depth, rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="PLY → depth raster + Rerun (same layout as render_3d.py)")
    parser.add_argument("ply", help="Path to ASCII PLY with x,y,z,r,g,b")
    parser.add_argument(
        "--ref-image",
        default=None,
        help="Optional RGB image (e.g. room1.jpg); sets resolution and camera image like render_3d.py",
    )
    parser.add_argument("--width", type=int, default=None, help="Raster width (default: ref image or 1920)")
    parser.add_argument("--height", type=int, default=None, help="Raster height (default: ref image or 1080)")
    parser.add_argument("--z-min", type=float, default=0.3, help="Clip points on PLY Z (same idea as render_3d)")
    parser.add_argument("--z-max", type=float, default=50.0, help="Clip points on PLY Z")
    parser.add_argument(
        "--app-id",
        default="SLAM_Visualization",
        help="Rerun app id (default matches render_3d.py)",
    )
    args = parser.parse_args()

    positions, colors = load_ply_xyz_rgb(args.ply)

    if args.ref_image:
        img = cv2.imread(args.ref_image, cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise SystemExit(f"Could not read image: {args.ref_image}")
        height, width = img.shape[0], img.shape[1]
    else:
        width = args.width or K_NOMINAL_W
        height = args.height or K_NOMINAL_H
        img = None

    K = scale_intrinsics(K_DEFAULT, width, height)

    z = positions[:, 2]
    valid = (z >= args.z_min) & (z <= args.z_max) & np.isfinite(z)
    positions_f = positions[valid]
    colors_f = colors[valid]

    depth_map, _rgb_raster = rasterize_depth_map(
        positions_f, colors_f, K, width, height, args.z_min, args.z_max
    )

    # --- Rerun: mirror render_3d.py ---
    rr.init(args.app_id)
    rr.spawn()

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    rr.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=K.tolist(),
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.LEFT_HAND_Y_UP,
            image_plane_distance=IMG_PLANE_DIST + 0.1,
        ),
    )

    if img is not None:
        rr.log("world/camera/image", rr.Image(cv2.flip(img, 1)))
    else:
        vis = rescale_depth_vis(depth_map)
        vis_bgr = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        rr.log("world/camera/image", rr.Image(cv2.flip(vis_rgb, 1)))

    rr.log(
        "world/object/points1",
        rr.Points3D(positions=positions_f, colors=colors_f),
    )

    print(
        f"Logged PLY {args.ply} ({positions_f.shape[0]} / {positions.shape[0]} points after z filter). "
        f"Depth map {width}x{height} (finite pixels: {np.sum(np.isfinite(depth_map))})."
    )


if __name__ == "__main__":
    main()
