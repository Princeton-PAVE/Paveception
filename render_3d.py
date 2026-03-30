import os
import numpy as np
import cv2
from utils import K, P1
import rerun as rr

rr.init("SLAM_Visualization")
rr.spawn()

rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

def homogenize(points):
    addition = np.expand_dims(np.ones(points.shape[0]), axis=-1)
    return np.concat([points_3d, addition], axis=-1)

prev_points = None
img_plane_dist = 0.25

def rescale(input):
    result = input.copy().astype('float64')
    result -= np.amin(result)
    result /= np.amax(result)
    result *= 255
    return result


# for fname in nums:
disparity = cv2.imread(f"room4.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread(f"pic.jpg", cv2.IMREAD_COLOR_RGB)

# stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15) # Adjust parameters as needed
# disparity = stereo.compute(frame1, frame2)

cv2.imshow('Depth Map', rescale(disparity).astype('uint8'))
cv2.waitKey(1)
Q = np.zeros((4,4))
cv2.stereoRectify(K, np.zeros((1,5)), K, np.zeros((1,5)), (disparity.shape[1], disparity.shape[0]), R=np.eye(3), T=np.array([1, 0, 0]), alpha=-1, Q=Q)
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# condition = np.logical_and(points_3d[:,:,2] > -15, points_3d[:,:,2] < -img_plane_dist)
# condition = np.logical_and(condition, points_3d[:,:,1] > -0.5)
points_3d[:,:,2] *= -1
# points_3d = points_3d[condition]

# masking z values 
z = points_3d[:,:,2]
valid_mask = (z >= 0.3) & (z <= 3.0)


# points_2d1 = 0.53*frame1.shape[1]*points_3d[:, 0] / points_3d[:, 2]
# points_2d1 += 385

# points_2d2 = 1.3*frame1.shape[0]*points_3d[:, 1] / points_3d[:, 2]
# points_2d2 *= -1
# points_2d2 += 165

# points_2d1 = points_2d1.astype(int)
# points_2d2 = points_2d2.astype(int)

# frame2_flipped = cv2.flip(frame2, 1)
# colors = [frame2_flipped[points_2d2[i]][points_2d1[i]] for i in range(len(points_3d))]
# colors = np.stack([colors, colors, colors], axis=-1)

colors = np.stack([img, img, img], axis=-1)

points_flat = points_3d.reshape(-1, 3)
colors_flat = img.reshape(-1, 3)

print(points_flat)
print(colors_flat)

valid_mask_flat = valid_mask.reshape(-1)
points_flat = points_flat[valid_mask_flat]
colors_flat = colors_flat[valid_mask_flat]
# points_4d = homogenize(points_3d)
# if prev_points is not None:
#     A, _, _, _ = np.linalg.lstsq(prev_points, points_4d, rcond=None)
#     print(A)
# prev_points = points_4d

rr.log(
    "world/camera",
    rr.Pinhole(
        image_from_camera=K.tolist(),  # 3x3 matrix
        resolution=[img.shape[1], img.shape[0]],  # [width, height]
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
