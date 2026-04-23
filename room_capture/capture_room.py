"""capture_room.py

Full room reconstruction from ~6 iPhone 16 Pro (main lens) photos using
DA3NESTED-GIANT-LARGE.

Outputs (into --output dir):
    room.rrd                 Rerun recording
    room.glb                 Point cloud + camera wireframes
    mini.npz                 Depth + conf + intrinsics + extrinsics
    *_3dgs.ply / *_3dgs.mp4  Gaussian-splat reconstruction + trajectory video
    depth_vis.*              Color-coded depth previews

Usage:
    cd "d:\\My Projects\\PAVE\\Paveception"
    python -m room_capture.capture_room
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageOps

# Optional HEIC support
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False

import rerun as rr
import torch

# Support both `python -m room_capture.capture_room` and `python capture_room.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from room_capture.hf_auth import setup_hf_token  # type: ignore[no-redef]
    from room_capture.iphone_intrinsics import build_K, scale_K  # type: ignore[no-redef]
    from room_capture.rerun_logging import (  # type: ignore[no-redef]
        init_world,
        log_camera,
        log_merged_point_cloud,
    )
else:
    from .hf_auth import setup_hf_token
    from .iphone_intrinsics import build_K, scale_K
    from .rerun_logging import (
        init_world,
        log_camera,
        log_merged_point_cloud,
    )

# NOTE: depth_anything_3 is imported lazily inside main() so hf_auth can
# install the HF token before any HuggingFace client is instantiated.


SUPPORTED_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_input_images(input_dir: Path) -> list[Path]:
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise FileNotFoundError(
            f"No images found in {input_dir}. "
            f"Drop 6 iPhone photos ({sorted(SUPPORTED_EXTS)}) there first."
        )
    return files


def load_image(path: Path) -> tuple[Image.Image, int]:
    """Load an image and normalize its orientation.

    iPhone photos are stored sensor-native (typically 4032x3024 landscape
    pixels) with an EXIF Orientation tag telling viewers how to rotate for
    display. Different shots in the same capture often have different
    Orientation values (portrait vs landscape vs upside down). If we feed
    DA3 the raw sensor pixels, each view has its own notion of "up" and the
    estimated extrinsics end up in incompatible world frames, so the image
    overlays in Rerun don't line up.

    We fix that here once and for all: `ImageOps.exif_transpose` physically
    rotates the pixels to the display orientation and clears the Orientation
    tag. Every downstream stage (intrinsics, DA3, Rerun) then sees a
    consistent, gravity-aligned "up = up" frame.

    Returns the oriented image plus the original EXIF Orientation code (1 if
    missing) so callers can log what was rotated.
    """
    if path.suffix.lower() in {".heic", ".heif"} and not _HEIC_OK:
        raise RuntimeError(
            f"Cannot open {path.name}: pillow-heif is not installed. "
            "Run `pip install pillow-heif` or convert the HEIC to JPG."
        )
    img = Image.open(path)
    try:
        orig_orientation = int(img.getexif().get(0x0112, 1))  # 0x0112 = Orientation
    except Exception:
        orig_orientation = 1
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return img, orig_orientation


# ---------------------------------------------------------------------------
# DA3 inference wrapper
# ---------------------------------------------------------------------------

def run_da3(
    model,  # depth_anything_3.api.DepthAnything3, imported lazily
    rgb_arrays: list[np.ndarray],
    intrinsics: np.ndarray,
    export_dir: Path,
    export_format: str,
    process_res: int,
    infer_gs: bool,
    use_ray_pose: bool,
):
    """Try pose-conditioned inference with known intrinsics, fall back on failure."""
    kwargs = dict(
        image=rgb_arrays,
        process_res=process_res,
        process_res_method="upper_bound_resize",
        use_ray_pose=use_ray_pose,
        ref_view_strategy="saddle_balanced",
        infer_gs=infer_gs,
        export_dir=str(export_dir),
        export_format=export_format,
        conf_thresh_percentile=40.0,
        num_max_points=2_000_000,
        show_cameras=True,
    )

    with torch.no_grad():
        try:
            # First attempt: pass known iPhone intrinsics only.
            return model.inference(intrinsics=intrinsics, **kwargs), "intrinsics_only"
        except TypeError as e:
            # Only TypeError means "API signature rejected intrinsics-only";
            # other errors (e.g. RuntimeError from the GS exporter) happen
            # AFTER the expensive forward pass has already completed, so
            # rerunning inference is just wasted wall-time. Re-raise those.
            warnings.warn(
                f"DA3 API rejected intrinsics-only call ({e}); "
                "retrying without camera args."
            )

        # Fallback: no camera args; DA3 estimates everything.
        return model.inference(**kwargs), "estimated"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Room reconstruction from iPhone photos via DA3")
    here = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=here / "input",
                        help="Folder containing iPhone photos (default: room_capture/input)")
    parser.add_argument("--output", type=Path, default=here / "output",
                        help="Output directory (default: room_capture/output)")
    parser.add_argument("--model", type=str,
                        default="depth-anything/DA3NESTED-GIANT-LARGE",
                        help="HuggingFace model id")
    parser.add_argument("--process-res", type=int, default=504,
                        help="DA3 processing resolution (default: 504)")
    parser.add_argument("--no-gs", action="store_true",
                        help="Skip Gaussian-splatting exports (less VRAM, faster)")
    parser.add_argument("--no-rerun", action="store_true",
                        help="Do not open or stream to any Rerun sink "
                             "(fully headless, no viewer, no .rrd).")
    parser.add_argument("--rrd", type=Path, default=None, metavar="PATH",
                        help="Save Rerun recording to this .rrd file. Mutually "
                             "exclusive with the default spawned viewer (rerun "
                             "allows only one sink). Default: off (viewer only).")
    parser.add_argument("--ray-pose", action="store_true",
                        help="Use ray-based pose head (slightly slower, often more accurate)")
    parser.add_argument("--log-depth-image", action="store_true",
                        help="Also log per-view DepthImage to Rerun. WARNING: "
                             "rerun auto-backprojects DepthImages under a "
                             "Pinhole into a colormapped point cloud that "
                             "competes visually with the RGB merged cloud. "
                             "Off by default so you see true image colors in 3D.")
    parser.add_argument("--lens", choices=("auto", "main", "ultrawide"), default="auto",
                        help="iPhone 16 Pro lens profile: 'main' (1x / 24mm-equiv), "
                             "'ultrawide' (0.5x / 13mm-equiv), or 'auto' (use EXIF, "
                             "default to main if EXIF is missing). Default: auto.")
    parser.add_argument("--sample-ratio", type=float, default=0.15,
                        help="Fraction of points to log to Rerun cloud (default: 0.15)")
    parser.add_argument("--conf-percentile", type=float, default=40.0,
                        help="Drop points below this per-view confidence percentile "
                             "before sampling. 0 keeps all confidence values; "
                             "40 keeps the top 60%% (default: 40.0).")
    parser.add_argument("--min-depth", type=float, default=0.05,
                        help="Ignore depths below this threshold in meters "
                             "(default: 0.05).")
    parser.add_argument("--max-depth", type=float, default=300.0,
                        help="Ignore depths above this threshold in meters "
                             "(default: 300.0).")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace access token. Prefer setting HF_TOKEN "
                             "in the environment or a .env file; passing it on "
                             "the CLI leaks it into shell history.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # --- HuggingFace auth (must happen before depth_anything_3 import) ------
    setup_hf_token(
        explicit_token=args.hf_token,
        dotenv_path=here / ".env",
    )

    from depth_anything_3.api import DepthAnything3  # lazy import, post-auth

    # --- Load images & intrinsics -------------------------------------------
    paths = find_input_images(args.input)
    print(f"[room_capture] Found {len(paths)} image(s) in {args.input}:")
    for p in paths:
        print(f"  - {p.name}")
    if len(paths) != 6:
        warnings.warn(
            f"Expected 6 photos, got {len(paths)}. Pipeline will still run, "
            "but reconstruction quality is tuned for ~6 main-lens captures."
        )

    pil_images: list[Image.Image] = []
    rgb_arrays: list[np.ndarray] = []
    orig_sizes: list[tuple[int, int]] = []  # (W, H)
    Ks_orig: list[np.ndarray] = []
    sources: list[str] = []
    hfovs: list[float] = []
    lenses: list[str] = []

    # EXIF Orientation -> human-readable rotation (used only for diagnostics).
    _ORIENT_LABEL = {
        1: "upright",
        2: "mirror-h",
        3: "rot180",
        4: "mirror-v",
        5: "transpose",
        6: "rot90-cw",
        7: "transverse",
        8: "rot90-ccw",
    }

    print(f"[room_capture] Lens profile: {args.lens}")
    for path in paths:
        img, orig_orientation = load_image(path)
        pil_images.append(img)
        W, H = img.size  # post-EXIF-transpose; "up = up" in pixel space now
        orig_sizes.append((W, H))
        arr = np.asarray(img, dtype=np.uint8)
        rgb_arrays.append(arr)
        K_info = build_K(img, lens=args.lens)
        Ks_orig.append(K_info.K)
        sources.append(K_info.source)
        hfovs.append(K_info.hfov_deg)
        lenses.append(K_info.lens)
        orient_label = _ORIENT_LABEL.get(orig_orientation, f"orient={orig_orientation}")
        print(
            f"  [K] {path.name}: {W}x{H} ({orient_label}), lens={K_info.lens}, "
            f"source={K_info.source}, HFoV={K_info.hfov_deg:.2f} deg, "
            f"fx={K_info.K[0,0]:.1f} fy={K_info.K[1,1]:.1f}"
        )

    # Sanity: after EXIF-transpose every view should share the same orientation
    # class (all portrait or all landscape). Warn if not, because that usually
    # means pose alignment across views will be degraded.
    orient_sigs = {"portrait" if h > w else "landscape" for (w, h) in orig_sizes}
    if len(orient_sigs) > 1:
        warnings.warn(
            f"Mixed orientations across views: {orient_sigs}. DA3 can still "
            "reconstruct, but keeping the phone in a single orientation (all "
            "landscape OR all portrait) gives cleaner pose alignment."
        )

    # If the user pinned a lens but EXIF says otherwise on any photo, warn.
    if args.lens != "auto":
        detected = {
            s.split("(")[-1].rstrip(")") for s in sources
            if s.startswith("exif_")
        }
        if detected and detected != {args.lens}:
            warnings.warn(
                f"--lens={args.lens} pinned, but EXIF suggests {detected}. "
                "Using the pinned profile anyway; intrinsics may be wrong if "
                "the photos were actually captured on a different lens."
            )

    intrinsics_in = np.stack(Ks_orig, axis=0).astype(np.float32)  # (N, 3, 3)

    # --- Model --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[room_capture] Loading {args.model} on {device} ...")
    model = DepthAnything3.from_pretrained(args.model).to(device)
    model.eval()

    # --- Export formats -----------------------------------------------------
    fmt_parts = ["mini_npz", "glb", "depth_vis"]
    gsplat_available = importlib.util.find_spec("gsplat") is not None
    infer_gs = (not args.no_gs) and gsplat_available
    if (not args.no_gs) and (not gsplat_available):
        warnings.warn(
            "gsplat is not installed; disabling gs_ply/gs_video exports. "
            "Install it to re-enable full 3DGS rendering."
        )
    if infer_gs:
        fmt_parts += ["gs_ply", "gs_video"]
    export_format = "-".join(fmt_parts)
    print(f"[room_capture] Export format: {export_format}")

    # --- Inference ----------------------------------------------------------
    print(f"[room_capture] Running DA3 inference on {len(rgb_arrays)} images ...")
    prediction, used_intrinsics_mode = run_da3(
        model=model,
        rgb_arrays=rgb_arrays,
        intrinsics=intrinsics_in,
        export_dir=args.output,
        export_format=export_format,
        process_res=args.process_res,
        infer_gs=infer_gs,
        use_ray_pose=args.ray_pose,
    )
    print(f"[room_capture] DA3 path: intrinsics={used_intrinsics_mode}")

    # --- Unpack predictions -------------------------------------------------
    depths = np.asarray(prediction.depth)
    if depths.ndim == 4:
        depths = np.squeeze(depths, axis=1)  # (N, H, W)
    depths = depths.astype(np.float32)

    conf = getattr(prediction, "conf", None)
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.ndim == 4:
            conf = np.squeeze(conf, axis=1)

    ext_pred = np.asarray(prediction.extrinsics, dtype=np.float32)   # (N, 3, 4)
    int_pred = np.asarray(prediction.intrinsics, dtype=np.float32)   # (N, 3, 3)
    proc_imgs = np.asarray(prediction.processed_images, dtype=np.uint8)  # (N, H, W, 3)

    N, out_h, out_w = depths.shape
    print(f"[room_capture] DA3 output: N={N}, depth resolution={out_w}x{out_h}")

    # --- Diagnostic: compare iPhone K vs DA3 predicted K --------------------
    print("[room_capture] Intrinsics comparison (iPhone EXIF -> DA3 output res "
          f"vs predicted):")
    Ks_iphone_out: list[np.ndarray] = []
    for i in range(N):
        K_scaled = scale_K(Ks_orig[i], orig_sizes[i], (out_w, out_h))
        Ks_iphone_out.append(K_scaled)
        fx_i, fy_i = K_scaled[0, 0], K_scaled[1, 1]
        fx_p, fy_p = int_pred[i, 0, 0], int_pred[i, 1, 1]
        print(
            f"  view {i}: iPhone fx={fx_i:.2f} fy={fy_i:.2f} "
            f"| DA3 fx={fx_p:.2f} fy={fy_p:.2f} "
            f"| drift fx={100.0*(fx_p-fx_i)/max(fx_i,1e-6):+.1f}%"
        )

    # --- Rerun logging ------------------------------------------------------
    # Rerun allows only one sink at a time, so viewer vs .rrd is mutually
    # exclusive (see DA3-video.py lines 207-210 for the same pattern).
    rrd_path: Path | None = None
    if not args.no_rerun:
        rrd_path = args.rrd
        init_world(
            "RoomCapture",
            save_path=str(rrd_path) if rrd_path is not None else None,
            spawn=rrd_path is None,
        )

    c2ws: list[np.ndarray] = []
    for i in range(N):
        c2w = log_camera(
            idx=i,
            rgb=proc_imgs[i],
            depth=depths[i],
            intrinsic=int_pred[i],
            extrinsic_3x4=ext_pred[i],
            static=True,
            log_depth_image=args.log_depth_image,
        )
        c2ws.append(c2w)

        if not args.no_rerun:
            # Diagnostic sibling pinhole using our iPhone-EXIF intrinsics.
            K_list = np.asarray(Ks_iphone_out[i], dtype=np.float32).reshape(3, 3).tolist()
            translation = np.asarray(c2w[:3, 3], dtype=np.float32).reshape(3).tolist()
            rotation = np.asarray(c2w[:3, :3], dtype=np.float32).reshape(3, 3).tolist()
            rr.log(
                f"world/diag/camera_{i}_iphoneK",
                rr.Transform3D(translation=translation, mat3x3=rotation),
                static=True,
            )
            rr.log(
                f"world/diag/camera_{i}_iphoneK",
                rr.Pinhole(
                    image_from_camera=K_list,
                    resolution=[out_w, out_h],
                    camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
                    image_plane_distance=0.2,
                ),
                static=True,
            )

    n_pts = 0
    if not args.no_rerun:
        print(
            f"[room_capture] Cloud filters: conf>={args.conf_percentile:.1f}th pct, "
            f"depth in [{args.min_depth:.3f}, {args.max_depth:.3f}] m, "
            f"sample_ratio={args.sample_ratio:.3f}"
        )
        n_pts = log_merged_point_cloud(
            rgbs=list(proc_imgs),
            depths=list(depths),
            intrinsics=list(int_pred),
            c2ws=c2ws,
            confs=list(conf) if conf is not None else None,
            conf_percentile=args.conf_percentile,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            sample_ratio=args.sample_ratio,
            entity_path="world/points",
            static=True,
        )
    print(f"[room_capture] Logged {n_pts:,} world points to Rerun.")

    # --- Save npz side-car with our own iPhone intrinsics -------------------
    np.savez_compressed(
        args.output / "room_intrinsics.npz",
        K_iphone_orig=np.stack(Ks_orig, axis=0),
        K_iphone_out=np.stack(Ks_iphone_out, axis=0),
        K_da3=int_pred,
        ext_da3=ext_pred,
        orig_sizes=np.asarray(orig_sizes, dtype=np.int32),
        sources=np.asarray(sources),
        lenses=np.asarray(lenses),
    )

    print(f"[room_capture] Done. Artifacts in {args.output}")
    if rrd_path is not None:
        print(f"  - Rerun recording : {rrd_path}")
    elif not args.no_rerun:
        print("  - Rerun viewer    : spawned (pass --rrd PATH to save a file instead)")
    print(f"  - Exported by DA3 : see files matching '{export_format}' in {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[room_capture] Interrupted.")
        sys.exit(130)
