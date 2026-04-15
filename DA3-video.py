"""
DA3-video.py
Runs Depth Anything 3 on a video file and visualises the results in Rerun.
Uses the HuggingFace API (like test.py) — no local da3_streaming dependencies.

Per-frame Rerun output:
  world/camera          — Transform3D + Pinhole
  world/camera/image    — RGB
  world/camera/depth    — DepthImage (metric)
  world/points          — 3-D point cloud
"""

import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import rerun as rr
import torch
from depth_anything_3.api import DepthAnything3


# ── helpers ─────────────────────────────────────────────────────────────────────

def depth_to_world_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Back-project depth into world-space 3-D points.

    Args:
        depth:     [H, W] float32
        intrinsic: [3, 3] float32
        c2w:       [4, 4] float32 camera-to-world transform

    Returns:
        [H*W, 3] float32 world-space positions
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

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)  # [H,W,4]
    pts_world = (c2w @ pts_cam.reshape(-1, 4).T).T[:, :3]                    # [H*W, 3]
    return pts_world


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert a float32 depth map to a BGR uint8 image using the Inferno colormap."""
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)  # BGR


def log_frame(
    frame_idx: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    sample_ratio: float,
    depth_writer: cv2.VideoWriter | None = None,
) -> None:
    """Log one frame to Rerun and optionally write a colourised depth frame.

    Args:
        frame_idx:    global frame index (timeline value)
        rgb:          [H, W, 3] uint8 RGB
        depth:        [H, W] float32
        intrinsic:    [3, 3] float32
        extrinsic:    [3, 4] float32 w2c
        sample_ratio: fraction of valid points to log
        depth_writer: optional VideoWriter to save colourised depth video
    """
    rr.set_time("frame", sequence=frame_idx)

    H, W = rgb.shape[:2]

    # c2w from w2c
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = extrinsic
    c2w = np.linalg.inv(w2c)

    # ── camera pose ──────────────────────────────────────────────────────────
    rr.log(
        "world/camera",
        rr.Transform3D(
            translation=c2w[:3, 3],
            mat3x3=c2w[:3, :3],
        ),
    )

    # ── pinhole ──────────────────────────────────────────────────────────────
    rr.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=intrinsic.tolist(),
            resolution=[W, H],
            camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
            image_plane_distance=0.35,
        ),
    )

    # ── RGB image ─────────────────────────────────────────────────────────────
    rr.log("world/camera/image", rr.Image(rgb))

    # ── depth image ───────────────────────────────────────────────────────────
    rr.log("world/camera/depth", rr.DepthImage(depth, meter=1.0))

    # ── depth video frame ────────────────────────────────────────────────────
    if depth_writer is not None:
        depth_writer.write(colorize_depth(depth))

    # ── 3-D point cloud ───────────────────────────────────────────────────────
    # filter on camera-space depth before transforming to world
    depth_flat = depth.reshape(-1)
    valid      = (depth_flat > 0.05) & (depth_flat < 300.0)

    pts_world = depth_to_world_points(depth, intrinsic, c2w)
    colors    = rgb.reshape(-1, 3)

    pts_world = pts_world[valid]
    colors    = colors[valid]

    if sample_ratio < 1.0 and len(pts_world) > 0:
        n   = max(1, int(len(pts_world) * sample_ratio))
        idx = np.random.choice(len(pts_world), n, replace=False)
        pts_world = pts_world[idx]
        colors    = colors[idx]

    rr.log("world/points", rr.Points3D(positions=pts_world, colors=colors))


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DA3-video: Depth Anything 3 + Rerun")
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/da3-small",
        help="HuggingFace model ID (default: depth-anything/da3-small)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS to sample from the video (default: every frame)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Number of frames per inference batch (default: 16)",
    )
    parser.add_argument(
        "--rrd",
        type=str,
        default=None,
        metavar="PATH",
        help="Save Rerun recording to .rrd file instead of spawning a viewer",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.05,
        help="Fraction of points to log per frame (default: 0.05)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    # ── load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model} on {device}…")
    model = DepthAnything3.from_pretrained(args.model).to(device)
    model.eval()

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, round(native_fps / args.fps)) if args.fps else 1
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {native_fps:.1f} fps, ~{total} frames — "
          f"sampling every {sample_every} frame(s), chunk size {args.chunk_size}")

    # ── init rerun ────────────────────────────────────────────────────────────
    rr.init("DA3_Video", spawn=args.rrd is None)
    if args.rrd:
        rr.save(args.rrd)
        print(f"Saving Rerun recording → {args.rrd}")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # ── process video in chunks ───────────────────────────────────────────────
    chunk_frames: list[np.ndarray] = []   # BGR frames
    chunk_global_indices: list[int] = []  # corresponding global frame indices
    frame_idx    = 0
    global_idx   = 0

    depth_writer: cv2.VideoWriter | None = None
    depth_video_path = os.path.splitext(args.video)[0] + "_depth.mp4"

    def process_chunk(frames: list[np.ndarray], indices: list[int]) -> None:
        nonlocal depth_writer

        rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

        with torch.no_grad():
            prediction = model.inference(rgb_list)

        depths     = np.squeeze(prediction.depth, axis=1) if prediction.depth.ndim == 4 else prediction.depth
        extrinsics = prediction.extrinsics   # [N, 3, 4]
        intrinsics = prediction.intrinsics   # [N, 3, 3]

        out_h, out_w = depths.shape[1], depths.shape[2]
        images = np.stack([
            cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            for rgb in rgb_list
        ])  # [N, out_H, out_W, 3]

        # Create depth VideoWriter on first chunk once we know the output resolution
        if depth_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = (native_fps / sample_every) if args.fps is None else args.fps
            depth_writer = cv2.VideoWriter(depth_video_path, fourcc, out_fps, (out_w, out_h))
            print(f"Depth video → {depth_video_path}")

        for local_i, global_i in enumerate(indices):
            log_frame(
                frame_idx    = global_i,
                rgb          = images[local_i],
                depth        = depths[local_i],
                intrinsic    = intrinsics[local_i],
                extrinsic    = extrinsics[local_i],
                sample_ratio = args.sample_ratio,
                depth_writer = depth_writer,
            )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            chunk_frames.append(frame)
            chunk_global_indices.append(global_idx)
            global_idx += 1

            if len(chunk_frames) == args.chunk_size:
                print(f"Processing frames {chunk_global_indices[0]}–{chunk_global_indices[-1]}…")
                process_chunk(chunk_frames, chunk_global_indices)
                chunk_frames          = []
                chunk_global_indices  = []

        frame_idx += 1

    # flush remaining frames
    if chunk_frames:
        print(f"Processing final {len(chunk_frames)} frame(s)…")
        process_chunk(chunk_frames, chunk_global_indices)

    cap.release()
    if depth_writer is not None:
        depth_writer.release()
        print(f"Depth video saved → {depth_video_path}")
    print("Done.")


if __name__ == "__main__":
    main()
