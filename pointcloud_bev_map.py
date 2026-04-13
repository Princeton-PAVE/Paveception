"""
Build a 2D bird's-eye-view (BEV) map from a 3D point cloud by segmenting objects
with DBSCAN — the same clustering step used in ``render_video.py`` after YOLO masks
isolate foreground points (see ``orientationfit.cluster_3d_points``).

Inputs: ASCII PLY (xyz + optional rgb, matching ``da3_pointcloud.save_ply_ascii``) or
``.npy`` with shape (N, 3) positions or (N, 6) positions + RGB.

Usage:
  python pointcloud_bev_map.py --input da3_pointcloud.ply --out bev_map.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from orientationfit import project_3d_to_2d


def load_ply_ascii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ASCII PLY with x,y,z and optional uchar r,g,b per vertex."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            lines.append(line.strip())
            if line.strip() == "end_header":
                break

    n_vertices = 0
    props: List[str] = []
    for ln in lines:
        if ln.startswith("element vertex"):
            n_vertices = int(ln.split()[-1])
        if ln.startswith("property"):
            props.append(ln.split()[-1])

    has_rgb = "red" in props and "green" in props and "blue" in props
    positions = np.zeros((n_vertices, 3), dtype=np.float64)
    if has_rgb:
        colors = np.zeros((n_vertices, 3), dtype=np.uint8)
    else:
        colors = np.full((n_vertices, 3), 200, dtype=np.uint8)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # skip header again
        while f.readline().strip() != "end_header":
            pass
        for i in range(n_vertices):
            parts = f.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Bad vertex line {i} in {path}")
            positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            if has_rgb and len(parts) >= 6:
                colors[i] = [int(parts[3]), int(parts[4]), int(parts[5])]

    return positions.astype(np.float32), colors


def load_npy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"Expected .npy shape (N,3) or (N,6), got {arr.shape}")
    pos = arr[:, :3].astype(np.float32)
    if arr.shape[1] == 6:
        col = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    else:
        col = np.full((pos.shape[0], 3), 200, dtype=np.uint8)
    return pos, col


def load_point_cloud(path: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return load_ply_ascii(path)
    if ext == ".npy":
        return load_npy(path)
    raise ValueError(f"Unsupported format {ext}; use .ply or .npy")


def voxel_downsample(
    positions: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        return positions, colors
    q = np.floor(positions.astype(np.float64) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    n_groups = counts.shape[0]
    sum_pos = np.zeros((n_groups, 3), dtype=np.float64)
    sum_col = np.zeros((n_groups, 3), dtype=np.float64)
    np.add.at(sum_pos, inverse, positions.astype(np.float64))
    np.add.at(sum_col, inverse, colors.astype(np.float64))
    cnt = counts.astype(np.float64)[:, None]
    out_p = (sum_pos / cnt).astype(np.float32)
    out_c = np.clip(sum_col / cnt, 0, 255).astype(np.uint8)
    return out_p, out_c


def random_subsample(
    positions: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = positions.shape[0]
    if max_points <= 0 or n <= max_points:
        return positions, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return positions[idx], colors[idx]


def _world_xz_to_pixels(
    xz: np.ndarray,
    xmin: float,
    xmax: float,
    zmin: float,
    zmax: float,
    width: int,
    height: int,
    margin_px: int,
) -> np.ndarray:
    """Map XZ world coords to image (u right, v down) with Z forward as -row (typical BEV)."""
    eff_w = max(width - 2 * margin_px, 1)
    eff_h = max(height - 2 * margin_px, 1)
    xr = xmax - xmin
    zr = zmax - zmin
    if xr < 1e-9:
        xr = 1.0
    if zr < 1e-9:
        zr = 1.0
    u = margin_px + ((xz[:, 0] - xmin) / xr) * (eff_w - 1)
    # Flip Z so increasing world Z maps upward on the image.
    v = margin_px + (1.0 - (xz[:, 1] - zmin) / zr) * (eff_h - 1)
    return np.stack([u, v], axis=1).astype(np.float32)


def render_bev_map(
    clusters: Dict[int, np.ndarray],
    cluster_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    *,
    width: int = 1024,
    height: int = 1024,
    margin_px: int = 16,
    meters_per_pixel: Optional[float] = None,
    point_radius: int = 1,
    draw_hulls: bool = True,
    hull_alpha: float = 0.35,
    bg_color: Tuple[int, int, int] = (24, 24, 28),
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Rasterize clusters into a BEV image (X horizontal, Z vertical on the map).

    Returns:
        BGR uint8 image, meta dict with bounds and scale.
    """
    if not clusters:
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)
        return img, {"xmin": 0.0, "xmax": 1.0, "zmin": 0.0, "zmax": 1.0, "meters_per_pixel": 1.0}

    all_xz = []
    for pts in clusters.values():
        all_xz.append(project_3d_to_2d(pts, plane="xz"))
    xz_cat = np.vstack(all_xz)
    xmin, xmax = float(np.min(xz_cat[:, 0])), float(np.max(xz_cat[:, 0]))
    zmin, zmax = float(np.min(xz_cat[:, 1])), float(np.max(xz_cat[:, 1]))

    pad = 0.05 * max(xmax - xmin, zmax - zmin, 0.5)
    xmin -= pad
    xmax += pad
    zmin -= pad
    zmax += pad

    if meters_per_pixel is not None and meters_per_pixel > 0:
        xr = xmax - xmin
        zr = zmax - zmin
        width = int(np.ceil(xr / meters_per_pixel)) + 2 * margin_px
        height = int(np.ceil(zr / meters_per_pixel)) + 2 * margin_px
        width = max(width, 64)
        height = max(height, 64)

    eff_w = max(width - 2 * margin_px, 1)
    eff_h = max(height - 2 * margin_px, 1)
    mpp_x = (xmax - xmin) / eff_w
    mpp_z = (zmax - zmin) / eff_h
    mpp = float(max(mpp_x, mpp_z))

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    overlay = img.copy()

    cluster_ids = sorted(clusters.keys())

    for i, cid in enumerate(cluster_ids):
        pts3 = clusters[cid]
        xz = project_3d_to_2d(pts3, plane="xz")
        uv = _world_xz_to_pixels(xz, xmin, xmax, zmin, zmax, width, height, margin_px).astype(np.int32)

        if cluster_colors and cid in cluster_colors:
            bgr = cluster_colors[cid][::-1]  # RGB input -> BGR for OpenCV
        else:
            hue = int((i * 47) % 180)
            color_hsv = np.uint8([[[hue, 220, 255]]])
            bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        if point_radius > 0:
            for (u, v) in uv:
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(img, (int(u), int(v)), point_radius, bgr, -1, lineType=cv2.LINE_AA)

        if draw_hulls and len(uv) >= 3:
            try:
                hull = cv2.convexHull(uv.reshape(-1, 1, 2))
                cv2.fillConvexPoly(overlay, hull, bgr, lineType=cv2.LINE_AA)
            except cv2.error:
                pass

    if draw_hulls:
        cv2.addWeighted(overlay, hull_alpha, img, 1.0 - hull_alpha, 0, img)

    meta = {
        "xmin": xmin,
        "xmax": xmax,
        "zmin": zmin,
        "zmax": zmax,
        "meters_per_pixel": mpp,
        "width": float(width),
        "height": float(height),
    }
    return img, meta


def segment_objects_with_colors(
    positions: np.ndarray,
    colors: np.ndarray,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, int, int]]]:
    """
    DBSCAN in 3D with the same hyperparameters as ``orientationfit.cluster_3d_points``
    (used after YOLO masking in ``render_video.py``). Noise label -1 is dropped.
    """
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(positions)
    labels = clustering.labels_
    cluster_colors: Dict[int, Tuple[int, int, int]] = {}
    clusters: Dict[int, np.ndarray] = {}
    for label in sorted(set(labels)):
        if label < 0:
            continue
        mask = labels == label
        pts = positions[mask]
        clusters[int(label)] = pts
        c = colors[mask].astype(np.float64).mean(axis=0)
        cluster_colors[int(label)] = (
            int(np.clip(c[0], 0, 255)),
            int(np.clip(c[1], 0, 255)),
            int(np.clip(c[2], 0, 255)),
        )
    return clusters, cluster_colors


def run_cli(args: argparse.Namespace) -> None:
    positions, colors = load_point_cloud(args.input)
    positions, colors = random_subsample(positions, colors, args.max_points, seed=args.seed)
    positions, colors = voxel_downsample(positions, colors, args.voxel)

    clusters, cluster_rgb = segment_objects_with_colors(
        positions,
        colors,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )

    mpp = args.meters_per_pixel if args.meters_per_pixel > 0 else None
    img, meta = render_bev_map(
        clusters,
        cluster_rgb,
        width=args.width,
        height=args.height,
        meters_per_pixel=mpp,
        point_radius=args.point_radius,
        draw_hulls=not args.no_hulls,
        hull_alpha=args.hull_alpha,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Wrote BEV map: {args.out}")
    print(
        f"Bounds X [{meta['xmin']:.3f}, {meta['xmax']:.3f}]  "
        f"Z [{meta['zmin']:.3f}, {meta['zmax']:.3f}]  "
        f"~{meta['meters_per_pixel']:.4f} m/px  "
        f"{len(clusters)} objects"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Point cloud → bird's-eye map (DBSCAN objects)")
    p.add_argument("--input", required=True, help="Input .ply (ascii) or .npy")
    p.add_argument("--out", default="bev_map.png", help="Output image path")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument(
        "--meters-per-pixel",
        type=float,
        default=0.0,
        help="If >0, image size follows scene extent / this value (overrides width/height)",
    )
    p.add_argument("--dbscan-eps", type=float, default=0.25, help="DBSCAN neighborhood (world units)")
    p.add_argument("--dbscan-min-samples", type=int, default=25)
    p.add_argument("--voxel", type=float, default=0.0, help="Voxel size for downsampling (0=off)")
    p.add_argument("--max-points", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--point-radius", type=int, default=1)
    p.add_argument("--no-hulls", action="store_true", help="Skip translucent convex hull overlay")
    p.add_argument("--hull-alpha", type=float, default=0.35)
    return p


if __name__ == "__main__":
    run_cli(build_arg_parser().parse_args())
