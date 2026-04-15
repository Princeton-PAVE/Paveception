import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import rerun as rr
from PIL import Image


def deproject_pixel_to_point(u: float, v: float, depth: float, fx: float, fy: float, ppx: float, ppy: float) -> Tuple[float, float, float]:
    """Convert a pixel coordinate + depth to camera-space 3D point."""
    if depth <= 0:
        return 0.0, 0.0, 0.0

    x = (u - ppx) * depth / fx
    y = (v - ppy) * depth / fy
    z = depth
    return x, y, z


def build_bev_map(
    points: np.ndarray,
    object_centers: List[Tuple[float, float]],
    object_labels: List[str],
    map_size: Tuple[int, int] = (700, 900),
    pixels_per_meter: float = 60.0,
    max_forward_m: float = 12.0,
    max_side_m: float = 6.0,
) -> np.ndarray:
    """Create a top-down bird's-eye-view map image."""
    width, height = map_size
    bev = np.zeros((height, width, 3), dtype=np.uint8)
    center_x = width // 2
    base_y = height - 20

    # Draw camera origin
    cv2.drawMarker(bev, (center_x, base_y), (255, 255, 255), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=20, thickness=2)
    cv2.putText(bev, "Camera", (center_x - 40, base_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if points.size:
        # Only keep points inside our intended BEV range
        xs = points[:, 0]
        zs = points[:, 2]
        valid = (zs > 0) & (zs < max_forward_m) & (np.abs(xs) < max_side_m)
        xs = xs[valid]
        zs = zs[valid]

        map_x = np.clip((center_x + xs * pixels_per_meter).astype(int), 0, width - 1)
        map_y = np.clip((base_y - zs * pixels_per_meter).astype(int), 0, height - 1)

        for x, y in zip(map_x, map_y):
            bev[y, x] = (64, 190, 230)

    for i, (cx, cz) in enumerate(object_centers):
        map_x = int(np.clip(center_x + cx * pixels_per_meter, 0, width - 1))
        map_y = int(np.clip(base_y - cz * pixels_per_meter, 0, height - 1))
        color = (0, 255, 0) if i % 2 == 0 else (0, 180, 255)
        cv2.circle(bev, (map_x, map_y), 10, color, -1)

        label = object_labels[i]
        cv2.putText(bev, label, (map_x + 8, map_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.rectangle(bev, (0, 0), (width - 1, height - 1), (150, 150, 150), 1)
    return bev


def compute_object_depth_and_center(
    mask: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute object center in image pixels and median depth in meters."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    depths = depth_m[ys, xs]
    valid = depths > 0
    if not np.any(valid):
        return None

    depths = depths[valid]
    xs = xs[valid]
    ys = ys[valid]
    median_depth = float(np.median(depths))
    center_u = float(np.mean(xs))
    center_v = float(np.mean(ys))

    x, y, z = deproject_pixel_to_point(center_u, center_v, median_depth, fx, fy, ppx, ppy)
    return x, y, z, median_depth


def normalize_depth_to_image(depth_m: np.ndarray, max_distance: float = 10.0) -> np.ndarray:
    depth_norm = np.clip(depth_m, 0.0, max_distance)
    depth_norm = (depth_norm / max_distance * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def run_mapping(args: argparse.Namespace) -> None:
    # Initialize Rerun visualization
    rr.init("Mapping_Visualization")
    rr.spawn()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    model = YOLO(args.model)

    # Load depth frames if provided
    depth_frames = None
    if args.depth_frames:
        try:
            depth_frames = np.load(args.depth_frames)
            print(f"Loaded depth frames from {args.depth_frames}, shape: {depth_frames.shape}")
        except Exception as e:
            print(f"Warning: Could not load depth frames: {e}")

    # Open video file or use camera
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Opened video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Using camera feed")

    if not cap.isOpened():
        print("Error: Could not open video/camera")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Camera intrinsics (adjust based on your camera)
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    ppx = frame_width / 2.0  # principal point x
    ppy = frame_height / 2.0  # principal point y

    frame_idx = 0
    try:
        while True:
            ret, color_image = cap.read()
            if not ret:
                break

            # Resize if downsampling is requested
            if args.downsample != 1:
                color_image = cv2.resize(color_image, (frame_width // args.downsample, frame_height // args.downsample))

            # Get depth image
            if depth_frames is not None and frame_idx < len(depth_frames):
                depth_image = depth_frames[frame_idx].astype(np.float32)
                # Resize depth to match color image if needed
                if depth_image.shape != color_image.shape[:2]:
                    depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))
            else:
                # Create dummy depth if not available
                depth_image = np.ones((color_image.shape[0], color_image.shape[1]), dtype=np.float32) * 5.0

            results = model(color_image, conf=args.confidence, device=args.device)
            r = results[0]

            object_centers: List[Tuple[float, float]] = []
            object_labels: List[str] = []

            if hasattr(r, "masks") and r.masks is not None:
                masks = r.masks.data.cpu().numpy()
            else:
                masks = None

            if hasattr(r, "boxes") and r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy()
            else:
                boxes = np.empty((0, 4), dtype=float)
                classes = np.array([], dtype=int)
                confidences = np.array([], dtype=float)

            bev_points = []
            for y in range(0, depth_image.shape[0], args.point_step):
                for x in range(0, depth_image.shape[1], args.point_step):
                    depth_m = depth_image[y, x]
                    if depth_m <= 0 or depth_m > args.max_distance:
                        continue
                    px, py, pz = deproject_pixel_to_point(float(x), float(y), float(depth_m), fx, fy, ppx, ppy)
                    bev_points.append((px, py, pz))
            bev_points = np.array(bev_points, dtype=np.float32)

            annotated = color_image.copy()
            object_meta: List[Dict[str, object]] = []
            if masks is not None and masks.shape[0] > 0:
                for idx, mask in enumerate(masks):
                    if idx >= len(classes):
                        break

                    obj_result = compute_object_depth_and_center(mask.astype(bool), depth_image, fx, fy, ppx, ppy)
                    if obj_result is None:
                        continue

                    x, y, z, dist = obj_result
                    class_id = int(classes[idx])
                    label = model.model.names[class_id] if hasattr(model, "model") and hasattr(model.model, "names") else str(class_id)
                    object_centers.append((x, z))
                    object_labels.append(f"{label} {dist:.2f}m")
                    object_meta.append({"label": label, "distance": dist, "center": (x, y, z)})

                    # Draw detection on camera image
                    x1, y1, x2, y2 = boxes[idx].astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label} {dist:.2f}m", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    center_u = int((x1 + x2) / 2)
                    center_v = int((y1 + y2) / 2)
                    cv2.circle(annotated, (center_u, center_v), 4, (0, 255, 255), -1)
            else:
                # If masks are absent, use bounding boxes only
                for idx, bbox in enumerate(boxes):
                    x1, y1, x2, y2 = bbox.astype(int)
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)
                    depth_patch = depth_image[y1:y2, x1:x2].flatten()
                    depth_patch = depth_patch[depth_patch > 0]
                    if depth_patch.size == 0:
                        continue
                    dist = float(np.median(depth_patch))
                    px, py, pz = deproject_pixel_to_point(cx, cy, dist, fx, fy, ppx, ppy)
                    label = model.model.names[int(classes[idx])] if hasattr(model, "model") and hasattr(model.model, "names") else str(int(classes[idx]))
                    object_centers.append((px, pz))
                    object_labels.append(f"{label} {dist:.2f}m")
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(annotated, f"{label} {dist:.2f}m", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            bev = build_bev_map(
                bev_points,
                object_centers,
                object_labels,
                map_size=(args.map_width, args.map_height),
                pixels_per_meter=args.pixels_per_meter,
                max_forward_m=args.max_distance,
                max_side_m=args.side_range,
            )

            depth_vis = normalize_depth_to_image(depth_image, max_distance=args.max_distance)
            
            # Resize for consistent display
            camera_resized = cv2.resize(annotated, (640, 480))
            depth_resized = cv2.resize(depth_vis, (640, 480))
            bev_resized = cv2.resize(bev, (640, 480))
            
            # Create layout: Camera (left) | Depth Map (center) | BEV Map (right)
            top_row = np.hstack((camera_resized, depth_resized))
            full_display = np.hstack((top_row, bev_resized))
            
            # Display main visualization
            cv2.imshow("Mapping Visualization: [Camera] [Depth] [BEV Map]", full_display)
            
            # Also show BEV map larger in separate window
            cv2.imshow("BEV Occupancy Map (Full Size)", bev)
            
            # Log to Rerun (lighter logging, just key data)
            img_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            rr.log("world/camera/image", rr.Image(img_rgb))
            
            if bev_points.size > 0:
                rr.log("world/points", rr.Points3D(positions=bev_points, colors=[100, 150, 200]))
            
            if object_centers:
                # Log object detections as text summary
                obj_summary = "\n".join(object_labels)
                rr.log("world/objects", rr.TextDocument(obj_summary))

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-based object mapping to bird's-eye-view with Rerun visualization")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (if None, uses camera)")
    parser.add_argument("--depth-frames", type=str, default="depth_frames.npy", help="Path to depth frames .npy file")
    parser.add_argument("--model", type=str, default="yoloe-11l-seg", help="YOLO model for object detection")
    parser.add_argument("--device", type=str, default="cpu", help="Device for YOLO inference")
    parser.add_argument("--width", type=int, default=640, help="Color frame width")
    parser.add_argument("--height", type=int, default=480, help="Color frame height")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--max-distance", type=float, default=10.0, help="Max distance in meters for BEV projection")
    parser.add_argument("--map-width", type=int, default=700, help="Bird's-eye-view map width")
    parser.add_argument("--map-height", type=int, default=900, help="Bird's-eye-view map height")
    parser.add_argument("--pixels-per-meter", type=float, default=60.0, help="BEV scale in pixels per meter")
    parser.add_argument("--side-range", type=float, default=6.0, help="Max half-width in meters for BEV map")
    parser.add_argument("--point-step", type=int, default=6, help="Step between depth sample points for BEV cloud")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample color frames for inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mapping(args)
