import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch import device, cuda
from depth_anything_3.api import DepthAnything3
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load model from Hugging Face Hub
d = device("cuda" if cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-small")
model = model.to(device=d)

# Run inference on images
images = ["semidetailed_room.jpg"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(images)

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
    
    # Ensure disparity is float32 for OpenCV
    # disparity_32f = disparity.astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    points_3d[:,:,2] *= -1
    # points_3d *= 0.1

    points_flat = points_3d.reshape(-1, 3)
    colors_flat = img.reshape(-1, 3)

    # rr.log(
    #     "world/camera",
    #     rr.Pinhole(
    #         image_from_camera=K.tolist(),  # 3x3 matrix
    #         resolution=[frame.shape[1], frame.shape[0]],  # [width, height]
    #         camera_xyz=rr.ViewCoordinates.LEFT_HAND_Y_UP,
    #         image_plane_distance=img_plane_dist + 0.1,
    #     )
    # )

    # rr.log(
    #     "world/camera/image",
    #     rr.Image(cv2.flip(img, 1))
    # )

    # rr.log(
    #     "world/object/points1",
    #     rr.Points3D(
    #         positions=points_flat,
    #         colors=colors_flat
    #     )
    # )
    frame_count += 1


# Access results
# print(type(prediction.depth))
# print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
# print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
# print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
# print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32

# result = np.squeeze(prediction.depth, axis=0)
# result = np.expand_dims(result, axis=-1)
# plt.imshow(result)
# plt.show()
