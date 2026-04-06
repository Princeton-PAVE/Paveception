import os
import numpy as np
from ultralytics import YOLO, YOLOE
from transformers import pipeline
from PIL import Image
import cv2
from utils import K

import rerun as rr
from sklearn.cluster import DBSCAN

# Import orientation fitting utilities
from orientationfit import process_chair_points

rr.init("SLAM_Visualization")
rr.spawn()

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

model = YOLOE("yoloe-26s-seg.pt")
names = ["chair"]
model.set_classes(names, model.get_text_pe(names))
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def homogenize(points):
    addition = np.expand_dims(np.ones(points.shape[0]), axis=-1)
    return np.concat([points_3d, addition], axis=-1)

prev_points = None
shrink = 0.5
img_plane_dist = 0.25 * shrink
K = (shrink * K).astype('int64')

def rescale(input):
    result = input.copy().astype('float64')
    result -= np.amin(result)
    result /= np.amax(result)
    result *= 255
    return result



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

    img = frame[:,:,::-1]
    disparity = np.array(pipe(Image.fromarray(img))["depth"])


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

    # YOLO
    results = model.track(frame, persist=True, conf=0.1)
    annotated_frame = results[0].plot()

    cv2.imshow("Detections", annotated_frame)
    cv2.waitKey(1)

    cv2.imshow('Depth Map', rescale(disparity).astype('uint8'))
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


    Q = np.zeros((4,4))
    cv2.stereoRectify(K, np.zeros((1,5)), K, np.zeros((1,5)), (disparity.shape[1], disparity.shape[0]), R=np.eye(3), T=np.array([1, 0, 0]), alpha=-1, Q=Q)
    # Ensure disparity is float32 for OpenCV
    disparity_32f = disparity.astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity_32f, Q)

    points_3d[:,:,2] *= -1
    points_3d *= 0.1

    points_flat = points_3d.reshape(-1, 3)
    colors_flat = img.reshape(-1, 3)


    chair_points_3d = None
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        class_indices = results[0].boxes.cls.cpu().numpy()

        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = np.logical_or(combined_mask, mask)

        combined_mask = cv2.resize(combined_mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
        masks_flat = combined_mask.reshape(-1).astype(bool)
        points_flat = points_flat[masks_flat]
        colors_flat = colors_flat[masks_flat]

        # Save 3D points for orientation fitting
        chair_points_3d = points_flat.copy()


    # --- DBSCAN Filtering ---
    if len(points_flat) > 10 and results[0].masks is not None:
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(points_flat)
        labels = clustering.labels_
        mask_noise = labels != -1
        points_flat = points_flat[mask_noise]
        colors_flat = colors_flat[mask_noise]
    # -------------------------

    # --- 3D Oriented Bounding Boxes for rerun ---
    if chair_points_3d is not None and len(chair_points_3d) > 10:
        orientation_results = process_chair_points(chair_points_3d, dbscan_eps=0.05, dbscan_min_samples=10, plane="xz")
        for cluster_id, info in orientation_results.items():
            centroid = info["centroid_3d"]
            theta = info["theta"]
            # Box size (heuristic, e.g., 0.3m x 0.3m x 0.5m)
            box_w, box_h, box_y = 0.3, 0.3, 0.5
            c, s = np.cos(theta), np.sin(theta)
            dx = box_w / 2
            dz = box_h / 2
            dy = box_y / 2
            # 8 corners of the box in local frame (XZ rectangle extruded in Y)
            local_corners = np.array([
                [dx,  dy, dz],
                [dx,  dy, -dz],
                [-dx, dy, -dz],
                [-dx, dy, dz],
                [dx, -dy, dz],
                [dx, -dy, -dz],
                [-dx, -dy, -dz],
                [-dx, -dy, dz],
            ])
            # Rotation matrix for XZ
            rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
            rotated = local_corners @ rot.T
            corners_3d = rotated + centroid

            # Log 3D box to rerun
            rr.log(
                f"world/object/chair_box_{cluster_id}",
                rr.Boxes3D(
                    centers=[centroid],
                    half_sizes=[[dx, dy, dz]],
                    rotations=[
                        [
                            [c, 0, -s],
                            [0, 1, 0],
                            [s, 0, c]
                        ]
                    ],
                    colors=[[0, 255, 0]]
                )
            )

            # Optionally, also draw 2D projection on annotated_frame for visual feedback
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            imgpts, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, np.zeros((1, 5)))
            imgpts = imgpts.squeeze().astype(int)
            # Draw box edges (12 lines)
            edges = [
                (0,1),(1,2),(2,3),(3,0), # top
                (4,5),(5,6),(6,7),(7,4), # bottom
                (0,4),(1,5),(2,6),(3,7)  # sides
            ]
            for i,j in edges:
                pt1 = tuple(imgpts[i])
                pt2 = tuple(imgpts[j])
                cv2.line(annotated_frame, pt1, pt2, (0,255,0), 2)

    rr.log(
        "world/camera",
        rr.Pinhole(
            image_from_camera=K.tolist(),  # 3x3 matrix
            resolution=[frame.shape[1], frame.shape[0]],  # [width, height]
            camera_xyz=rr.ViewCoordinates.LEFT_HAND_Y_UP,
            image_plane_distance=img_plane_dist + 0.1,
        )
    )

    rr.log(
        "world/camera/image",
        rr.Image(cv2.flip(img, 1))
    )

    rr.log(
        "world/object/points1",
        rr.Points3D(
            positions=points_flat,
            colors=colors_flat
        )
    )
    frame_count += 1
