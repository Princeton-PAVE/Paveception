"""Re-render a previously captured room in Rerun without re-running DA3.

Reads DA3's `mini_npz` export (depths + confs + intrinsics + extrinsics) plus
the original input images, resizes the images to match the stored depth
resolution, and re-logs everything to Rerun using the same helpers as
`capture_room.py`.

Typical layout after a capture run:

    room_capture/
        input/                          <- original photos
            IMG_0001.HEIC ...
        output/
            exports/
                mini_npz/
                    results.npz         <- depth / conf / extrinsics / intrinsics
            room_intrinsics.npz         <- our iPhone K side-car (optional)

Usage:

    # Spawn viewer:
    python -m room_capture.render_room

    # Or save to file:
    python -m room_capture.render_room --rrd room_capture/output/room.rrd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False

# Support both `python -m room_capture.render_room` and `python render_room.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from room_capture.rerun_logging import (  # type: ignore[no-redef]
        init_world,
        log_camera,
        log_merged_point_cloud,
    )
else:
    from .rerun_logging import (
        init_world,
        log_camera,
        log_merged_point_cloud,
    )


SUPPORTED_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def find_input_images(input_dir: Path) -> list[Path]:
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise FileNotFoundError(
            f"No images found in {input_dir}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )
    return files


def load_rgb(path: Path, target_wh: tuple[int, int]) -> np.ndarray:
    """Load an image, normalize EXIF orientation, and resize to (W, H).

    The EXIF-transpose step is critical: DA3 was fed EXIF-normalized pixels
    during capture, so its depths + extrinsics live in the display-upright
    frame. If we logged raw sensor pixels here they would no longer match
    the pinhole intrinsics the npz stored.
    """
    if path.suffix.lower() in {".heic", ".heif"} and not _HEIC_OK:
        raise RuntimeError(
            f"Cannot open {path.name}: pillow-heif is not installed. "
            "Run `pip install pillow-heif` or convert the HEIC to JPG."
        )
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    tw, th = target_wh
    if (img.width, img.height) != (tw, th):
        # DA3 uses upper_bound_resize with a bicubic/area filter under the hood;
        # the aspect ratio is preserved end-to-end, so a plain resize to match
        # the stored depth resolution reproduces the same pixel grid closely
        # enough for visualization.
        img = img.resize((tw, th), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def find_mini_npz(output_dir: Path) -> Path:
    """Locate the DA3 mini_npz results file under the given output directory."""
    candidates = [
        output_dir / "exports" / "mini_npz" / "results.npz",
        output_dir / "mini_npz" / "results.npz",
        output_dir / "results.npz",
    ]
    for c in candidates:
        if c.exists():
            return c

    hits = list(output_dir.rglob("mini_npz/results.npz"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not locate DA3 mini_npz results under {output_dir}. "
        f"Expected one of: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Re-render a captured room in Rerun from DA3's mini_npz output."
    )
    parser.add_argument("--input", type=Path, default=here / "input",
                        help="Folder with the original iPhone photos that were fed to "
                             "capture_room.py (default: room_capture/input)")
    parser.add_argument("--output", type=Path, default=here / "output",
                        help="Folder where DA3 wrote its exports "
                             "(default: room_capture/output)")
    parser.add_argument("--npz", type=Path, default=None,
                        help="Explicit path to results.npz (overrides auto-discovery "
                             "under --output).")
    parser.add_argument("--rrd", type=Path, default=None, metavar="PATH",
                        help="Save to this .rrd file instead of spawning a viewer.")
    parser.add_argument("--sample-ratio", type=float, default=0.15,
                        help="Fraction of valid pixels to log as world points "
                             "(default: 0.15 ~= 15%%).")
    parser.add_argument("--conf-percentile", type=float, default=40.0,
                        help="Drop points below this per-view confidence "
                             "percentile (default: 40.0).")
    parser.add_argument("--min-depth", type=float, default=0.05,
                        help="Ignore depths below this (m). Default: 0.05.")
    parser.add_argument("--max-depth", type=float, default=30.0,
                        help="Ignore depths above this (m). Default: 30.0.")
    parser.add_argument("--image-plane-distance", type=float, default=0.35,
                        help="Rerun pinhole image plane size (world units). "
                             "Default: 0.35.")
    parser.add_argument("--log-depth-image", action="store_true",
                        help="Also log per-view DepthImage to Rerun. WARNING: "
                             "rerun auto-backprojects DepthImages under a "
                             "Pinhole into a colormapped point cloud that "
                             "competes visually with the RGB merged cloud. "
                             "Off by default so you see true image colors in 3D.")

    args = parser.parse_args()

    npz_path = args.npz if args.npz is not None else find_mini_npz(args.output)
    print(f"[render_room] Loading DA3 results from {npz_path}")
    data = np.load(npz_path)

    depths = np.asarray(data["depth"], dtype=np.float32)
    if depths.ndim == 4:
        depths = np.squeeze(depths, axis=1)
    if depths.ndim != 3:
        raise ValueError(f"Unexpected depth shape {depths.shape}; want (N, H, W).")

    extrinsics = np.asarray(data["extrinsics"], dtype=np.float32)  # (N, 3, 4)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)  # (N, 3, 3)
    conf = np.asarray(data["conf"], dtype=np.float32) if "conf" in data.files else None
    if conf is not None and conf.ndim == 4:
        conf = np.squeeze(conf, axis=1)

    N, H, W = depths.shape
    print(f"[render_room] Loaded N={N}, depth resolution={W}x{H}")
    if extrinsics.shape[0] != N or intrinsics.shape[0] != N:
        raise ValueError(
            f"Shape mismatch: depths={depths.shape}, "
            f"extrinsics={extrinsics.shape}, intrinsics={intrinsics.shape}"
        )

    # --- Reload input images at the processed resolution -------------------
    input_paths = find_input_images(args.input)
    if len(input_paths) != N:
        print(
            f"[render_room] WARN: {len(input_paths)} images in {args.input} "
            f"but npz has N={N}. Using the first {min(N, len(input_paths))} "
            "paired by sort order."
        )
    paired = list(zip(input_paths[:N], range(N)))
    if len(paired) < N:
        raise RuntimeError(
            f"Only {len(paired)} input images available, need {N} to re-render."
        )

    rgb_arrays: list[np.ndarray] = []
    for path, i in paired:
        rgb = load_rgb(path, target_wh=(W, H))
        rgb_arrays.append(rgb)
        print(f"  view {i}: {path.name}  resized -> {rgb.shape[1]}x{rgb.shape[0]}")

    # --- Rerun setup (viewer XOR file, same rule as capture_room) ----------
    init_world(
        "RoomCapture",
        save_path=str(args.rrd) if args.rrd is not None else None,
        spawn=args.rrd is None,
    )

    # --- Per-camera logs ---------------------------------------------------
    c2ws: list[np.ndarray] = []
    for i in range(N):
        c2w = log_camera(
            idx=i,
            rgb=rgb_arrays[i],
            depth=depths[i],
            intrinsic=intrinsics[i],
            extrinsic_3x4=extrinsics[i],
            image_plane_distance=args.image_plane_distance,
            static=True,
            log_depth_image=args.log_depth_image,
        )
        c2ws.append(c2w)

    # --- Merged point cloud ------------------------------------------------
    n_pts = log_merged_point_cloud(
        rgbs=rgb_arrays,
        depths=list(depths),
        intrinsics=list(intrinsics),
        c2ws=c2ws,
        confs=list(conf) if conf is not None else None,
        conf_percentile=args.conf_percentile,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        sample_ratio=args.sample_ratio,
        entity_path="world/points",
        static=True,
    )
    print(f"[render_room] Logged {n_pts:,} world points to Rerun.")

    if args.rrd is not None:
        print(f"[render_room] Wrote recording: {args.rrd}")
        print(f"[render_room] Open it with:  rerun {args.rrd}")
    else:
        print("[render_room] Viewer spawned. Close the window when you're done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[render_room] Interrupted.")
        sys.exit(130)
