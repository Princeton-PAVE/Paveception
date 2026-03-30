import os
import numpy as np
from ultralytics import YOLO, YOLOE
from transformers import pipeline
from PIL import Image
import cv2
from utils import K
import rerun as rr
from sklearn.cluster import DBSCAN

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

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=shrink, fy=shrink)

    if not ret:
        break

    img = frame[:,:,::-1]
    disparity = np.array(pipe(Image.fromarray(img))["depth"])
    
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
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    points_3d[:,:,2] *= -1

    points_flat = points_3d.reshape(-1, 3)
    colors_flat = img.reshape(-1, 3)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        class_indices = results[0].boxes.cls.cpu().numpy()

        combined_mask = masks[0]
        for mask in masks[1:]:
            print(mask.shape)
            combined_mask = np.logical_or(combined_mask, mask)

        combined_mask = cv2.resize(combined_mask.astype('uint8'), (frame.shape[1], frame.shape[0]))
        masks_flat = combined_mask.reshape(-1).astype(bool)
        points_flat = points_flat[masks_flat]
        colors_flat = colors_flat[masks_flat]

    # --- DBSCAN Filtering ---
    if len(points_flat) > 10 and results[0].masks is not None:

        # 2. Run DBSCAN
        # eps: Max distance between two samples for them to be in the same cluster
        # min_samples: Min points to form a dense region
        clustering = DBSCAN(eps=0.05, min_samples=10).fit(points_flat)
        labels = clustering.labels_

        # 3. Filter out noise (label -1 is noise in DBSCAN)
        mask_noise = labels != -1
        points_flat = points_flat[mask_noise]
        colors_flat = colors_flat[mask_noise]
    # -------------------------

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
