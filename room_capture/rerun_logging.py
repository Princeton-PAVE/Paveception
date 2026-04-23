"""Rerun logging helpers for the room_capture pipeline.

Same conventions as Paveception/render_video.py and DA3-video.py:
  - World frame: right-handed, Y up
  - Each camera logged as a Pinhole under world/camera_{i}
  - Camera image plane uses RIGHT_HAND_Y_DOWN (OpenCV / DA3 convention)
  - Point cloud logged as one merged world/points entity
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import rerun as rr


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def depth_to_world_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Back-project a [H, W] depth map to [H*W, 3] world-space points.

    Matches the implementation in Paveception/render_video.py and DA3-video.py.
    """
    H, W = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x_cam = (uu - cx) * depth / fx
    y_cam = (vv - cy) * depth / fy
    z_cam = depth

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
    pts_world = (c2w @ pts_cam.reshape(-1, 4).T).T[:, :3]
    return pts_world


def w2c_to_c2w(extrinsic_3x4: np.ndarray) -> np.ndarray:
    """Lift a (3, 4) world-to-camera matrix to a (4, 4) camera-to-world matrix."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = extrinsic_3x4.astype(np.float32)
    return np.linalg.inv(w2c)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Float depth -> BGR uint8 inferno colormap."""
    d_min, d_max = float(np.nanmin(depth)), float(np.nanmax(depth))
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------------------
# Recording setup
# ---------------------------------------------------------------------------

def init_world(
    app_name: str = "RoomCapture",
    save_path: str | None = None,
    spawn: bool = True,
) -> None:
    """Initialize a Rerun recording and log the static world frame.

    Exactly one sink is set up: either a viewer (spawn) or a file (save_path).
    They are mutually exclusive because rerun's `save()` replaces the current
    sink, which silently drops later logs from the spawned viewer.
    """
    if save_path is not None:
        # File-only mode: no viewer.
        rr.init(app_name, spawn=False)
        rr.save(save_path)
    else:
        rr.init(app_name, spawn=spawn)

    # Static world frame so it's visible regardless of the time cursor.
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_camera(
    idx: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_3x4: np.ndarray,
    image_plane_distance: float = 0.35,
    static: bool = True,
    log_depth_image: bool = False,
) -> np.ndarray:
    """Log one camera (Transform3D + Pinhole + RGB image).

    Returns the (4, 4) camera-to-world matrix so callers can reuse it for
    point-cloud back-projection.

    Args:
        static: When True (the default) every component is logged as static so
                it's visible at all time cursors. For a one-shot multi-view
                reconstruction this is what you want; set False only when the
                pipeline is feeding a real timeline.
        log_depth_image: When True, also log the depth map as a DepthImage
                under the pinhole. Rerun auto-backprojects any DepthImage
                that's a child of a Pinhole into a colormapped 3D point
                cloud, which competes with the true-color merged cloud at
                ``world/points``. Default False. If you want the 2D depth
                preview without the 3D heat map, log the depth image from the
                caller to a path that is NOT under the pinhole (e.g.
                ``depth_previews/camera_{idx}``).
    """
    c2w = w2c_to_c2w(extrinsic_3x4)
    H, W = rgb.shape[:2]

    # Cast to plain python lists so rerun doesn't quietly reject odd ndarray
    # dtypes (e.g. non-contiguous float64 views).
    translation = np.asarray(c2w[:3, 3], dtype=np.float32).reshape(3).tolist()
    rotation = np.asarray(c2w[:3, :3], dtype=np.float32).reshape(3, 3).tolist()
    K_list = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3).tolist()

    rr.log(
        f"world/camera_{idx}",
        rr.Transform3D(translation=translation, mat3x3=rotation),
        static=static,
    )
    rr.log(
        f"world/camera_{idx}",
        rr.Pinhole(
            image_from_camera=K_list,
            resolution=[W, H],
            camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
            image_plane_distance=image_plane_distance,
        ),
        static=static,
    )
    rgb_c = np.ascontiguousarray(rgb, dtype=np.uint8)
    rr.log(f"world/camera_{idx}/image", rr.Image(rgb_c), static=static)

    if log_depth_image and depth.shape[:2] == rgb.shape[:2]:
        depth_c = np.ascontiguousarray(depth, dtype=np.float32)
        rr.log(
            f"world/camera_{idx}/depth",
            rr.DepthImage(depth_c, meter=1.0),
            static=static,
        )

    return c2w


def log_merged_point_cloud(
    rgbs: Sequence[np.ndarray],
    depths: Sequence[np.ndarray],
    intrinsics: Sequence[np.ndarray],
    c2ws: Sequence[np.ndarray],
    confs: Sequence[np.ndarray] | None = None,
    conf_percentile: float = 40.0,
    min_depth: float = 0.05,
    max_depth: float = 300.0,
    sample_ratio: float = 0.15,
    entity_path: str = "world/points",
    static: bool = True,
) -> int:
    """Back-project every (rgb, depth, K, c2w) tuple and log a single merged cloud.

    Returns the number of points logged (post-sampling).
    """
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for i, (rgb, depth, K, c2w) in enumerate(zip(rgbs, depths, intrinsics, c2ws)):
        pts_world = depth_to_world_points(depth, K, c2w).astype(np.float32)
        colors = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(-1, 3)

        depth_flat = depth.reshape(-1)
        valid = (
            (depth_flat > min_depth)
            & (depth_flat < max_depth)
            & np.isfinite(depth_flat)
        )
        # NaN/Inf defense on the backprojected coordinates themselves.
        valid = valid & np.isfinite(pts_world).all(axis=1)

        if confs is not None:
            conf_flat = np.asarray(confs[i], dtype=np.float32).reshape(-1)
            if conf_flat.shape[0] == valid.shape[0]:
                if valid.any():
                    thresh = float(np.percentile(conf_flat[valid], conf_percentile))
                else:
                    thresh = 0.0
                valid = valid & (conf_flat >= thresh)

        pts_world = pts_world[valid]
        colors = colors[valid]

        if sample_ratio < 1.0 and len(pts_world) > 0:
            n = max(1, int(len(pts_world) * sample_ratio))
            idx = np.random.choice(len(pts_world), n, replace=False)
            pts_world = pts_world[idx]
            colors = colors[idx]

        print(
            f"    view {i}: {int(valid.sum()):>8,} valid, "
            f"{len(pts_world):>8,} after sample "
            f"(depth min={float(depth_flat[np.isfinite(depth_flat)].min() if np.isfinite(depth_flat).any() else 0):.3f}, "
            f"max={float(depth_flat[np.isfinite(depth_flat)].max() if np.isfinite(depth_flat).any() else 0):.3f})"
        )

        if len(pts_world) > 0:
            all_pts.append(pts_world)
            all_cols.append(colors)

    if not all_pts:
        print("[rerun_logging] No valid points survived filtering; "
              "nothing logged to world/points.")
        return 0

    merged_pts = np.concatenate(all_pts, axis=0)
    merged_cols = np.concatenate(all_cols, axis=0)

    rr.log(
        entity_path,
        rr.Points3D(positions=merged_pts, colors=merged_cols, radii=0.005),
        static=static,
    )
    return int(len(merged_pts))
