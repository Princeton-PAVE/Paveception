import numpy as np
from ultralytics import YOLO, YOLOE
import cv2
import torch

import rerun as rr
from depth_anything_3.api import DepthAnything3
from sklearn.cluster import DBSCAN

# Import orientation fitting utilities
from orientationfit import process_chair_points

# Match DA3-video: fraction of points to log (full cloud is heavy)
POINT_SAMPLE_RATIO = 0.05

# Mirror display left–right (reflection about the vertical y-axis in the image plane). Set False to disable.
DISPLAY_MIRROR_Y_AXIS = True


def depth_to_world_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """Back-project depth into world-space 3-D points (same as DA3-video.py)."""
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


rr.init("SLAM_Visualization")
rr.spawn()

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

model = YOLOE("yoloe-26s-seg.pt")
names = ["chair"]
model.set_classes(names, model.get_text_pe(names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
da3 = DepthAnything3.from_pretrained("depth-anything/da3-small").to(device)
da3.eval()

def homogenize(points):
    addition = np.expand_dims(np.ones(points.shape[0]), axis=-1)
    return np.concat([points_3d, addition], axis=-1)

prev_points = None
shrink = 0.5

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Float depth → BGR uint8 colormap (same as DA3-video.py)."""
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)



# --- Parameters ---
top_n_rows = 20  # Number of top rows to use
f = 10           # Number of frames to skip before using reference
frame_count = 0
reference_top_rows = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=shrink, fy=shrink)

    if not ret:
        break

    img = frame[:, :, ::-1]  # BGR → RGB
    with torch.no_grad():
        prediction = da3.inference([img])

    d = np.asarray(prediction.depth)
    if d.ndim == 4:
        d = np.squeeze(d, axis=1)
    depth = np.asarray(d[0], dtype=np.float32)

    intrinsic = np.asarray(prediction.intrinsics[0], dtype=np.float32)
    extrinsic = np.asarray(prediction.extrinsics[0], dtype=np.float32)

    out_h, out_w = depth.shape[0], depth.shape[1]
    rgb = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    if DISPLAY_MIRROR_Y_AXIS:
        rgb_display = cv2.flip(rgb, 1)
        depth_display = np.ascontiguousarray(np.fliplr(depth))
        intrinsic_display = intrinsic.copy()
        intrinsic_display[0, 2] = float(out_w - 1) - intrinsic_display[0, 2]
    else:
        rgb_display = rgb
        depth_display = depth
        intrinsic_display = intrinsic

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = extrinsic
    c2w = np.linalg.inv(w2c)


    # # --- Isolate top n rows of disparity and rescale to match reference ---
    # # For first f frames, just collect reference
    # if frame_count < f:
    #     if frame_count == f-1:
    #         # Use the top n rows of the fth frame as reference
    #         reference_top_rows = np.copy(disparity[:top_n_rows, :])
    #     frame_count += 1
    #     # Show normal disparity and continue
    #     cv2.imshow('Depth Map', rescale(disparity).astype('uint8'))
    #     key = cv2.waitKey(1)
    #     if key & 0xFF == ord('q'):
    #         break
    #     continue

    # # After f frames, rescale so at least 50% of top n rows match reference
    # if reference_top_rows is not None:
    #     # Try different scaling factors to maximize overlap
    #     best_scale = 1.0
    #     best_match = 0
    #     for scale in np.linspace(0.8, 1.2, 21):
    #         scaled_disp = disparity * scale
    #         # Compare top n rows with reference
    #         curr_top_rows = scaled_disp[:top_n_rows, :]
    #         # Compute pixel-wise closeness (within a threshold)
    #         match = np.sum(np.abs(curr_top_rows - reference_top_rows) < 1e-2)
    #         if match > best_match:
    #             best_match = match
    #             best_scale = scale
    #     # If less than 50% match, use best scale
    #     if best_match < 0.5 * reference_top_rows.size:
    #         disparity = disparity * best_scale

    # YOLO (no OpenCV windows — view in Rerun only)
    model.track(frame, persist=True, conf=0.1)

    depth_flat = depth.reshape(-1)
    valid = (depth_flat > 0.05) & (depth_flat < 300.0)

    pts_world = depth_to_world_points(depth, intrinsic, c2w)
    colors_flat = rgb.reshape(-1, 3)

    pts_world = pts_world[valid]
    colors_flat = colors_flat[valid]

    if POINT_SAMPLE_RATIO < 1.0 and len(pts_world) > 0:
        n = max(1, int(len(pts_world) * POINT_SAMPLE_RATIO))
        idx = np.random.choice(len(pts_world), n, replace=False)
        pts_world = pts_world[idx]
        colors_flat = colors_flat[idx]

    points_flat = pts_world


    # chair_points_3d = None
    # if results[0].masks is not None:
    #     masks = results[0].masks.data.cpu().numpy()
    #     class_indices = results[0].boxes.cls.cpu().numpy()

    #     combined_mask = masks[0]
    #     for mask in masks[1:]:
    #         combined_mask = np.logical_or(combined_mask, mask)

    #     combined_mask = cv2.resize(combined_mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
    #     masks_flat = combined_mask.reshape(-1).astype(bool)
    #     points_flat = points_flat[masks_flat]
    #     colors_flat = colors_flat[masks_flat]

        # # Save 3D points for orientation fitting
        # chair_points_3d = points_flat.copy()


    # # --- DBSCAN Filtering ---
    # if len(points_flat) > 10 and results[0].masks is not None:
    #     clustering = DBSCAN(eps=0.5, min_samples=10).fit(points_flat)
    #     labels = clustering.labels_
    #     mask_noise = labels != -1
    #     points_flat = points_flat[mask_noise]
    #     colors_flat = colors_flat[mask_noise]
    # # -------------------------

    # # --- 3D Oriented Bounding Boxes for rerun ---
    # if chair_points_3d is not None and len(chair_points_3d) > 10:
    #     orientation_results = process_chair_points(chair_points_3d, dbscan_eps=0.05, dbscan_min_samples=10, plane="xz")
    #     for cluster_id, info in orientation_results.items():
    #         centroid = info["centroid_3d"]
    #         theta = info["theta"]
    #         # Box size (heuristic, e.g., 0.3m x 0.3m x 0.5m)
    #         box_w, box_h, box_y = 0.3, 0.3, 0.5
    #         c, s = np.cos(theta), np.sin(theta)
    #         dx = box_w / 2
    #         dz = box_h / 2
    #         dy = box_y / 2
    #         # 8 corners of the box in local frame (XZ rectangle extruded in Y)
    #         local_corners = np.array([
    #             [dx,  dy, dz],
    #             [dx,  dy, -dz],
    #             [-dx, dy, -dz],
    #             [-dx, dy, dz],
    #             [dx, -dy, dz],
    #             [dx, -dy, -dz],
    #             [-dx, -dy, -dz],
    #             [-dx, -dy, dz],
    #         ])
    #         # Rotation matrix for XZ
    #         rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    #         rotated = local_corners @ rot.T
    #         corners_3d = rotated + centroid

    #         # Log 3D box to rerun
    #         rr.log(
    #             f"world/object/chair_box_{cluster_id}",
    #             rr.Boxes3D(
    #                 centers=[centroid],
    #                 half_sizes=[[dx, dy, dz]],
    #                 rotations=[
    #                     [
    #                         [c, 0, -s],
    #                         [0, 1, 0],
    #                         [s, 0, c]
    #                     ]
    #                 ],
    #                 colors=[[0, 255, 0]]
    #             )
    #         )

    #         # Optionally, also draw 2D projection on annotated_frame for visual feedback
    #         rvec = np.zeros((3, 1))
    #         tvec = np.zeros((3, 1))
    #         imgpts, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, np.zeros((1, 5)))
    #         imgpts = imgpts.squeeze().astype(int)
    #         # Draw box edges (12 lines)
    #         edges = [
    #             (0,1),(1,2),(2,3),(3,0), # top
    #             (4,5),(5,6),(6,7),(7,4), # bottom
    #             (0,4),(1,5),(2,6),(3,7)  # sides
    #         ]
    #         for i,j in edges:
    #             pt1 = tuple(imgpts[i])
    #             pt2 = tuple(imgpts[j])
    #             cv2.line(annotated_frame, pt1, pt2, (0,255,0), 2)

    H, W = rgb.shape[:2]
    rr.set_time("frame", sequence=frame_count)

    rr.log(
        "world/camera",
        rr.Transform3D(
            translation=c2w[:3, 3],
            mat3x3=c2w[:3, :3],
        ),
    )
    rr.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=intrinsic_display.tolist(),
            resolution=[W, H],
            camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
            image_plane_distance=0.35,
        ),
    )
    depth_color_bgr = colorize_depth(depth_display)
    depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
    rr.log("world/camera/image", rr.Image(depth_color_rgb))

    rr.log(
        "world/points",
        rr.Points3D(positions=points_flat, colors=colors_flat),
    )

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    frame_count += 1
