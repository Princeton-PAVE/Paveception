"""
Build a colored 3D point map from Depth Anything 3 (DA3).

Upstream repo: https://github.com/ByteDance-Seed/Depth-Anything-3

------------------------------------------------------------------------------
Linux + NVIDIA (typical)
------------------------------------------------------------------------------
Install PyTorch with CUDA, then ``pip install xformers`` and ``pip install -e .`` from the cloned
repo, as in the upstream README.

------------------------------------------------------------------------------
macOS — skip xFormers and use Metal (MPS)
------------------------------------------------------------------------------
``xformers`` targets CUDA and often fails on Mac; you do **not** need it for this script. DA3
falls back to plain PyTorch for SwiGLU when ``xformers`` is absent (see
``depth_anything_3/model/dinov2/layers/swiglu_ffn.py``).

**xFormers vs MPS (both matter on Mac, but they are different):**

- **Skipping xFormers** means you do not install a CUDA-oriented extra package. DA3 then uses
  ordinary PyTorch implementations for those layers (e.g. SwiGLU). That is about *which code path* runs, not about Apple vs NVIDIA by itself.
- **MPS** is PyTorch’s **device** for Apple’s GPU (Metal), like ``cuda`` is for NVIDIA. Choosing
  ``--device mps`` means tensors are executed on the Apple GPU when available.
- MPS is **not** a replacement for xFormers; it is **where** the model runs. A typical Mac setup
  is: **omit xFormers** (easier install + fallback ops) **and** use **MPS** (or ``--device auto``)
  so inference still uses the Apple GPU instead of CPU-only.

**1. Python environment** (recommended: Python 3.10–3.12, venv or conda).

**2. PyTorch on Mac** — install the **macOS** build from https://pytorch.org (pip), which enables ``mps`` on Apple Silicon. Example::

 pip install torch torchvision

**3. Install Depth Anything 3 without pulling in xFormers** — pick **one** approach:

   **A — Edit ``pyproject.toml`` (simplest):** clone the repo, open ``pyproject.toml``, remove
   the line that lists ``xformers`` under ``dependencies``, save, then from the repo root::

       pip install -e .

   **B — ``--no-deps``:** from the repo root::

       pip install -e . --no-deps

   then install everything upstream lists in ``pyproject.toml`` **except** ``xformers`` (and add
   ``addict`` if import fails — some revisions omit it from the list).

**4. Run this script on Mac** — use Metal or let auto pick it::

    python da3_pointcloud.py room1.jpg --out out.ply --model depth-anything/DA3-SMALL --device mps

   or ``--device auto`` (uses ``mps`` after CUDA on platforms that support it).

**5. Optional:** if Hugging Face is slow or blocked, set ``HF_ENDPOINT`` or log in with
``huggingface-cli login`` per HF docs. First run downloads the model checkpoint.

This module calls ``apply_da3_runtime_compatibility()`` before loading DA3; on macOS it sets
``PYTORCH_ENABLE_MPS_FALLBACK=1`` so MPS can fall back to CPU for a few ops (e.g. some
upsampling). On Windows it sets ``KMP_DUPLICATE_LIB_OK`` to reduce OpenMP clashes.

**Note:** ``mps`` exists **only on macOS**. On Windows or Linux use ``--device cuda`` or ``--device cpu``.

Single or multi-view images are supported; multi-view fusion uses predicted world-to-camera
poses.

**Resolution / mapping quality:** Depth Anything 3 **resizes** inputs before the network using
``process_res`` (CLI ``--process-res``, default **504** on the bound side — see DA3
``upper_bound_resize`` / ``lower_bound_resize``). That **caps** output depth H×W and therefore
point-cloud density in image space; it is **not** full original megapixel resolution unless you
raise this (and have VRAM). This script also subsamples pixels with ``stride`` (default **2**);
use ``--stride 1`` for the densest cloud. Example for sharper mapping (if memory allows)::

    python da3_pointcloud.py room1.jpg --out out.ply --process-res 756 --stride 1 --device auto

Downstream (e.g. ``ply_depth_rerun.py``, ``pointcloud_bev_map.py``) only see what the PLY
contains; they do not recover sub-504 detail by themselves.

**Vertical orientation:** by default world **Y is negated** after fusion (``--flip-y``, Rerun Y-up
alignment). Use ``--no-flip-y`` for raw DA3 axes. Optional ``--flip-z`` negates **Z** as well.

A logged warning about missing ``gsplat`` is normal for depth / PLY workflows; only
Gaussian-splatting export needs it.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required for DA3. Install torch before using this module.") from e


def apply_da3_runtime_compatibility() -> None:
    """
    Call before importing `depth_anything_3` or running inference (safe to call multiple times).
    Improves Apple Silicon / MPS behavior; does not force-disable xformers on Linux.
    """
    if sys.platform == "darwin":
        # Bicubic resize and other ops may be missing on MPS; allow CPU fallback.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform == "win32":
        # Avoid abort when multiple OpenMP runtimes load (MKL + PyTorch, etc.).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def default_inference_device(explicit: Optional[str] = None) -> torch.device:
    """
    Choose CUDA if available, else Apple MPS (macOS only), else CPU. `explicit` can be
    ``\"cuda\"``, ``\"mps\"``, ``\"cpu\"``, or ``\"auto\"`` (default when None).
    """
    if explicit is not None and explicit.lower() not in ("auto", ""):
        name = explicit.lower()
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Requested --device cuda but torch.cuda.is_available() is False. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        if name == "mps":
            if sys.platform != "darwin":
                raise RuntimeError(
                    "MPS (Metal) exists only on macOS with a Metal-capable PyTorch build. "
                    f"This OS is {sys.platform!r} (e.g. Windows/Linux has no MPS). "
                    "Use --device auto, cuda, or cpu."
                )
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError(
                    "Requested --device mps but MPS is not available. "
                    "Use the official macOS wheels from https://pytorch.org on Apple Silicon (or a supported Mac GPU)."
                )
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


ImageInput = Union[str, np.ndarray]


@dataclass
class PointMap:
    """World-space point cloud (OpenCV-style camera / world conventions from DA3)."""

    positions: np.ndarray  # (P, 3) float32
    colors: np.ndarray  # (P, 3) uint8 RGB


def _se3_4x4_from_w2c(extrinsic_3x4: np.ndarray) -> np.ndarray:
    """Expand (3, 4) world-to-camera [R|t] to 4x4."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :] = extrinsic_3x4.astype(np.float64)
    return out


def _unproject_depth_to_camera(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    depth: (H, W), intrinsics: (3, 3) for the same resolution as depth.
    Returns (H, W, 3) camera-frame points (x right, y down, z forward).
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got {depth.shape}")
    h, w = depth.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float64)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _apply_confidence_mask(
    conf: Optional[np.ndarray],
    depth: np.ndarray,
    min_percentile: Optional[float],
) -> np.ndarray:
    """Boolean mask (H, W): valid depth and optional confidence percentile."""
    valid = np.isfinite(depth) & (depth > 0)
    if conf is None or min_percentile is None:
        return valid
    c = conf.astype(np.float64)
    finite_c = c[np.isfinite(c)]
    if finite_c.size == 0:
        return valid
    thr = np.percentile(finite_c, min_percentile)
    return valid & np.isfinite(c) & (c >= thr)


def depth_to_metric_if_needed(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    model_id: str,
) -> np.ndarray:
    """
    DA3METRIC-LARGE outputs a network value; FAQ: metric_depth = focal * out / 300.
    DA3NESTED* models report depth already in meters.
    """
    mid = model_id.lower()
    if "metric" in mid and "nested" not in mid:
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        focal = 0.5 * (fx + fy)
        return (focal * depth / 300.0).astype(np.float32)
    return depth.astype(np.float32)


def prediction_to_pointmap(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_3x4: np.ndarray,
    colors_rgb_uint8: np.ndarray,
    *,
    conf: Optional[np.ndarray] = None,
    conf_min_percentile: Optional[float] = 40.0,
    stride: int = 2,
    model_id: str = "",
    flip_z: bool = False,
    flip_y: bool = True,
) -> PointMap:
    """
    Fuse all views into one world-frame point cloud.

    depth: (N, H, W)
    intrinsics: (N, 3, 3)
    extrinsics_3x4: (N, 3, 4) world-to-camera
    colors_rgb_uint8: (N, H, W, 3)
    conf: optional (N, H, W)
    stride: pixel subsampling (>=1)
    flip_z: if True, negate world Z after fusion (viewer / axis alignment only).
    flip_y: if True, negate world Y after fusion (default on for common viewer conventions).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    n = depth.shape[0]
    all_pos: List[np.ndarray] = []
    all_col: List[np.ndarray] = []

    for i in range(n):
        d = depth_to_metric_if_needed(depth[i], intrinsics[i], model_id)
        K = intrinsics[i]
        pts_cam = _unproject_depth_to_camera(d, K)
        mask = _apply_confidence_mask(conf[i] if conf is not None else None, d, conf_min_percentile)
        if stride > 1:
            mask = mask[::stride, ::stride]
            pts_cam = pts_cam[::stride, ::stride]
            cols = colors_rgb_uint8[i][::stride, ::stride]
        else:
            cols = colors_rgb_uint8[i]

        pts_cam_flat = pts_cam.reshape(-1, 3)
        cols_flat = cols.reshape(-1, 3)
        m_flat = mask.reshape(-1)

        pts_cam_flat = pts_cam_flat[m_flat]
        cols_flat = cols_flat[m_flat]

        if pts_cam_flat.size == 0:
            continue

        w2c = _se3_4x4_from_w2c(extrinsics_3x4[i])
        c2w = np.linalg.inv(w2c)
        ones = np.ones((pts_cam_flat.shape[0], 1), dtype=np.float64)
        hom = np.concatenate([pts_cam_flat.astype(np.float64), ones], axis=1)
        world = (c2w @ hom.T).T[:, :3].astype(np.float32)

        all_pos.append(world)
        all_col.append(cols_flat)

    if not all_pos:
        return PointMap(
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
        )

    positions = np.concatenate(all_pos, axis=0)
    colors = np.concatenate(all_col, axis=0)
    if (flip_y or flip_z) and positions.shape[0] > 0:
        positions = positions.copy()
        if flip_y:
            positions[:, 1] *= -1.0
        if flip_z:
            positions[:, 2] *= -1.0
    return PointMap(positions=positions, colors=colors)


def save_ply_ascii(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
    """Minimal ASCII PLY writer (RGB0-255 uint8)."""
    if positions.shape[0] != colors.shape[0]:
        raise ValueError("positions and colors length mismatch")
    n = positions.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = positions[i]
            r, g, b = colors[i].astype(np.uint8)
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def run_da3_pointcloud(
    images: Sequence[ImageInput],
    *,
    model_id: str = "depth-anything/DA3-BASE",
    device: Optional[str] = None,
    conf_min_percentile: Optional[float] = 40.0,
    stride: int = 2,
    use_ray_pose: bool = False,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    flip_z: bool = False,
    flip_y: bool = True,
) -> Tuple[PointMap, object]:
    """
    Load DA3, run inference, return fused PointMap and raw prediction object.
    """
    apply_da3_runtime_compatibility()
    from depth_anything_3.api import DepthAnything3

    torch_device = default_inference_device(device)
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=torch_device)

    prediction = model.inference(
        list(images),
        use_ray_pose=use_ray_pose,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    depth = prediction.depth
    colors = prediction.processed_images
    intr = prediction.intrinsics
    ext = prediction.extrinsics
    if ext is not None:
        ext = np.asarray(ext)
        if ext.ndim == 3 and ext.shape[-2:] == (4, 4):
            ext = ext[:, :3, :]
    conf = getattr(prediction, "conf", None)

    if intr is None or ext is None:
        raise RuntimeError(
            "DA3 prediction missing intrinsics or extrinsics; try an any-view or nested model, "
            "not monocular-only checkpoints."
        )

    pmap = prediction_to_pointmap(
        depth,
        intr,
        ext,
        colors,
        conf=conf,
        conf_min_percentile=conf_min_percentile,
        stride=stride,
        model_id=model_id,
        flip_z=flip_z,
        flip_y=flip_y,
    )
    return pmap, prediction


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DA3 → fused world point cloud (PLY)")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image paths, or a single directory of images (png/jpg)",
    )
    parser.add_argument(
        "--model",
        default="depth-anything/DA3-BASE",
        help=(
            "Hugging Face model id. There is no 'Medium' name: DA3-BASE (~120M) is the mid any-view "
            "tier between SMALL and LARGE; use DA3-LARGE-1.1 or DA3NESTED-GIANT-LARGE-1.1 for heavier."
        ),
    )
    parser.add_argument("--out", default="da3_pointcloud.ply", help="Output PLY path")
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Pixel subsampling stride (larger = fewer points)",
    )
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=40.0,
        help="Keep pixels with conf >= this percentile (set negative to disable)",
    )
    parser.add_argument("--ray-pose", action="store_true", help="use_ray_pose for inference")
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 internal resize bound (larger = sharper depth / more points, more VRAM)",
    )
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch device: auto, cuda, mps (macOS only), or cpu. "
            "Default auto picks cuda, then mps on Mac, then cpu."
        ),
    )
    parser.add_argument(
        "--flip-z",
        dest="flip_z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Negate world Z in the PLY (coordinate flip only).",
    )
    parser.add_argument(
        "--flip-y",
        dest="flip_y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Negate world Y in the PLY (default on; use --no-flip-y for raw DA3).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths: List[str] = []
    for p in args.inputs:
        if os.path.isdir(p):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                paths.extend(sorted(glob.glob(os.path.join(p, ext))))
        else:
            paths.append(p)
    if not paths:
        raise SystemExit("No images found.")

    conf_pct = args.conf_percentile if args.conf_percentile >= 0 else None

    dev_arg = None if args.device.lower() == "auto" else args.device
    pmap, _pred = run_da3_pointcloud(
        paths,
        model_id=args.model,
        device=dev_arg,
        conf_min_percentile=conf_pct,
        stride=args.stride,
        use_ray_pose=args.ray_pose,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        flip_z=args.flip_z,
        flip_y=args.flip_y,
    )
    save_ply_ascii(args.out, pmap.positions, pmap.colors)
    print(f"Wrote {pmap.positions.shape[0]} points to {args.out}")


if __name__ == "__main__":
    main()
