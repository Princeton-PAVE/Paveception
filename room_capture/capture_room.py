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
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image

# Optional HEIC support
try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False

import rerun as rr
import torch

from .hf_auth import setup_hf_token
from .iphone_intrinsics import build_K, scale_K
from .rerun_logging import (
    colorize_depth,
    init_world,
    log_camera,
    log_merged_point_cloud,
    w2c_to_c2w,
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


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".heic", ".heif"} and not _HEIC_OK:
        raise RuntimeError(
            f"Cannot open {path.name}: pillow-heif is not installed. "
            "Run `pip install pillow-heif` or convert the HEIC to JPG."
        )
    img = Image.open(path)
    img = img.convert("RGB")
    return img


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
        except TypeError:
            # API may require intrinsics+extrinsics as a pair.
            warnings.warn(
                "DA3 API rejected intrinsics-only call; retrying without camera args."
            )
        except Exception as e:
            warnings.warn(
                f"DA3 inference with intrinsics failed ({type(e).__name__}: {e}); "
                "retrying with pose estimation only."
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
                        help="Do not spawn Rerun viewer (still writes .rrd)")
    parser.add_argument("--ray-pose", action="store_true",
                        help="Use ray-based pose head (slightly slower, often more accurate)")
    parser.add_argument("--lens", choices=("auto", "main", "ultrawide"), default="auto",
                        help="iPhone 16 Pro lens profile: 'main' (1x / 24mm-equiv), "
                             "'ultrawide' (0.5x / 13mm-equiv), or 'auto' (use EXIF, "
                             "default to main if EXIF is missing). Default: auto.")
    parser.add_argument("--sample-ratio", type=float, default=0.15,
                        help="Fraction of points to log to Rerun cloud (default: 0.15)")
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

    print(f"[room_capture] Lens profile: {args.lens}")
    for path in paths:
        img = load_image(path)
        pil_images.append(img)
        W, H = img.size
        orig_sizes.append((W, H))
        arr = np.asarray(img, dtype=np.uint8)
        rgb_arrays.append(arr)
        K_info = build_K(img, lens=args.lens)
        Ks_orig.append(K_info.K)
        sources.append(K_info.source)
        hfovs.append(K_info.hfov_deg)
        lenses.append(K_info.lens)
        print(
            f"  [K] {path.name}: {W}x{H}, lens={K_info.lens}, "
            f"source={K_info.source}, HFoV={K_info.hfov_deg:.2f} deg, "
            f"fx={K_info.K[0,0]:.1f} fy={K_info.K[1,1]:.1f}"
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
    infer_gs = not args.no_gs
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
    rrd_path = args.output / "room.rrd"
    if args.no_rerun:
        rr.init("RoomCapture", spawn=False)
        rr.save(str(rrd_path))
    else:
        init_world("RoomCapture", save_path=None)  # spawn viewer
        rr.save(str(rrd_path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    c2ws: list[np.ndarray] = []
    for i in range(N):
        rr.set_time("view", sequence=i)
        c2w = log_camera(
            idx=i,
            rgb=proc_imgs[i],
            depth=depths[i],
            intrinsic=int_pred[i],
            extrinsic_3x4=ext_pred[i],
        )
        c2ws.append(c2w)

        # Also log our iPhone-EXIF K as a diagnostic sibling pinhole (no image).
        rr.log(
            f"world/diag/camera_{i}_iphoneK",
            rr.Transform3D(
                translation=c2w[:3, 3],
                mat3x3=c2w[:3, :3],
            ),
        )
        rr.log(
            f"world/diag/camera_{i}_iphoneK",
            rr.Pinhole(
                image_from_camera=Ks_iphone_out[i].tolist(),
                resolution=[out_w, out_h],
                camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
                image_plane_distance=0.2,
            ),
        )

    n_pts = log_merged_point_cloud(
        rgbs=list(proc_imgs),
        depths=list(depths),
        intrinsics=list(int_pred),
        c2ws=c2ws,
        confs=list(conf) if conf is not None else None,
        conf_percentile=40.0,
        sample_ratio=args.sample_ratio,
        entity_path="world/points",
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
    print(f"  - Rerun recording : {rrd_path}")
    print(f"  - Exported by DA3 : see files matching '{export_format}' in {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[room_capture] Interrupted.")
        sys.exit(130)
