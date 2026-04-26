import numpy as np
from transformers import pipeline
from PIL import Image
import cv2
from utils import K
import rerun as rr

rr.init("SLAM_Visualization")
rr.spawn()

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

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


cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        continue
    
    frame = cv2.resize(frame, None, fx=shrink, fy=shrink)

    if not ret:
        break

    img = frame[:,:,::-1]
    disparity = np.array(pipe(Image.fromarray(img))["depth"], dtype='float32')
    print(disparity.dtype)


    # --- Isolate top n rows of disparity and rescale to match reference ---
    # For first f frames, just collect reference
    cv2.imshow('Depth Map', rescale(disparity).astype('uint8'))
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

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
