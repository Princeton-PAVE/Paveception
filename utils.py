import numpy as np
import cv2
from typing import List

# Intrinsics Matrix

f = 600
px = 1920//2
py = 1080//2

K = np.array([
    [f, 0, px],
    [0, f, py],
    [0, 0, 1]
])


# Perspective Matrix
P1 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
])

P1 = K @ P1

SHRINK = 2



class Mapper():
    x_over_f = None
    y_over_f = None
    x_offset = None
    y_offset = None

    # SIFT Keypoints
    detector = cv2.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    # Camera Center in Real-World Coordinates
    camera_center = np.array([0, 0, 0])

    def __init__(self, frame1, frame2):
        self.frame1 = frame1
        self.frame2 = frame2

        frame1_shrunk = cv2.resize(frame1, (frame1.shape[1]//SHRINK, frame1.shape[0]//SHRINK))
        frame2_shrunk = cv2.resize(frame2, (frame2.shape[1]//SHRINK, frame2.shape[0]//SHRINK))

        if frame1.shape[-1] == 1 or len(frame1.shape) == 2:
            self.input_frame1 = frame1_shrunk
        else:
            self.input_frame1 = cv2.cvtColor(frame1_shrunk, cv2.COLOR_BGR2GRAY)

        if frame1.shape[-1] == 1 or len(frame1.shape) == 2:
            self.input_frame2 = frame2_shrunk
        else:
            self.input_frame2 = cv2.cvtColor(frame2_shrunk, cv2.COLOR_BGR2GRAY)

        if Mapper.x_over_f is None:
            Mapper.x_over_f = frame1.shape[0] / K[0][0]
            Mapper.y_over_f = frame1.shape[1] / K[1][1]
            Mapper.x_offset = (K[0][2] - frame1.shape[1]//2)/(frame1.shape[1])
            Mapper.y_offset = (K[1][2] - frame1.shape[0]//2)/(frame1.shape[0]//2)

    def get_matched_homogenous_coordinates(self):
        self.kp1, self.des1 = Mapper.detector.detectAndCompute(self.input_frame1, None)
        self.kp2, self.des2 = Mapper.detector.detectAndCompute(self.input_frame2, None)

        if self.des1 is None or self.des2 is None:
            print("Couldn't match any keypoints")
            return None, None
        
        self.matches = Mapper.flann.knnMatch(self.des1, self.des2,k=2)
        self.matchesMask = [[0,0] for i in range(len(self.matches))]
        
        pts1 = []
        pts2 = []
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(self.matches):
            if m.distance < 0.5*n.distance:
                self.matchesMask[i]=[1,0]
                pts2.append(self.kp2[m.trainIdx].pt)
                pts1.append(self.kp1[m.queryIdx].pt)

        self.pts1 = np.array(pts1, dtype=np.float32)
        self.pts2 = np.array(pts2, dtype=np.float32)

        return self.pts1, self.pts2
    
    # Draw SIFT matches
    def drawMatches(self) -> np.ndarray:
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = self.matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        
        for kp in self.kp1:
            kp.pt = (kp.pt[0]*SHRINK, kp.pt[1]*SHRINK)

        for kp in self.kp2:
            kp.pt = (kp.pt[0]*SHRINK, kp.pt[1]*SHRINK)

        shown_frame = cv2.drawMatchesKnn(self.frame2, self.kp1, self.frame1, self.kp2, self.matches, None, **draw_params)

        return shown_frame


    # Matched 2d points to 3d Real-World Coordinates
    def get_points_3d(self, R: np.ndarray, t: np.ndarray, return_colors: bool = False) -> np.ndarray:
        pts1 = self.pts1.T
        pts2 = self.pts2.T
        P2 = K @ np.hstack((R, t))

        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)  # shape (4, N)
        points_3d_flat = points_4d[:3]

        points_4d[3] /= np.median(points_4d[3])
        points_3d = points_4d[:3] / points_4d[3] 
        points_3d = points_3d.T

        points_3d = np.abs(points_3d)
        points_3d[:, 0] = 2*points_3d[:, 0] - Mapper.x_over_f*points_3d[:, 2] + Mapper.x_offset*points_3d[:, 2]
        # points_3d[:, 0] *= -1
        points_3d[:, 1] = 2*points_3d[:, 1] - 0.275*Mapper.y_over_f*points_3d[:, 2] - Mapper.y_offset*points_3d[:, 2]

        self.points_3d = points_3d

        points_3d_flat = points_3d_flat.T
        points_3d_flat = np.abs(points_3d_flat)
        points_3d_flat[:, 0] = 2*points_3d_flat[:, 0] - Mapper.x_over_f*points_3d_flat[:, 2] + Mapper.x_offset*points_3d_flat[:, 2]
        points_3d_flat[:, 1] = 2*points_3d_flat[:, 1] - 0.275*Mapper.y_over_f*points_3d_flat[:, 2] - Mapper.y_offset*points_3d_flat[:, 2]

        if return_colors:
            colors = [self.frame1[pt.astype(int)] for pt in self.pts1]
            colors = np.stack(colors)
            return points_3d, colors
        else:
            return points_3d, points_3d_flat
        
    # def get_points_3d(self, R1: np.ndarray, R2: np.ndarray, t: np.ndarray, return_colors: bool = False) -> np.ndarray:
    #     pts1 = self.pts1.T
    #     pts2 = self.pts2.T
    #     P2_1 = K @ np.hstack((R1, t))
    #     P2_2 = K @ np.hstack((R2, t))

    #     points_4d1 = cv2.triangulatePoints(P1, P2_1, pts1, pts2)  # shape (4, N)
    #     points_3d1_flat = points_4d1[:3]

    #     points_4d1[3] /= np.median(points_4d1[3])
    #     points_3d1 = points_4d1[:3] / points_4d1[3] 
    #     points_3d1 = points_3d1.T

    #     points_3d = np.abs(points_3d)
    #     points_3d1[:, 0] = 2*points_3d1[:, 0] - Mapper.x_over_f*points_3d1[:, 2] + Mapper.x_offset*points_3d1_flat[:, 2]
    #     points_3d1[:, 1] = 2*points_3d1[:, 1] - 0.275*Mapper.y_over_f*points_3d1[:, 2] - Mapper.y_offset*points_3d1_flat[:, 2]

    #     self.points_3d1 = points_3d1

    #     points_3d1_flat = points_3d1_flat.T
    #     points_3d1_flat = np.abs(points_3d1_flat)
    #     points_3d1_flat[:, 0] = 2*points_3d1_flat[:, 0] - Mapper.x_over_f*points_3d1_flat[:, 2] + Mapper.x_offset*points_3d1_flat[:, 2]
    #     points_3d1_flat[:, 1] = 2*points_3d1_flat[:, 1] - 0.275*Mapper.y_over_f*points_3d1_flat[:, 2] - Mapper.y_offset*points_3d1_flat[:, 2]

    #     #---------

    #     points_4d_2 = cv2.triangulatePoints(P1, P2_2, pts1, pts2)  # shape (4, N)
    #     points_3d2_flat = points_4d_2[:3]

    #     points_4d_2[3] /= np.median(points_4d_2[3])
    #     points_3d2 = points_4d_2[:3] / points_4d_2[3] 
    #     points_3d2 = points_3d2.T

    #     points_3d2 = np.abs(points_3d2)
    #     points_3d2[:, 0] = 2*points_3d2[:, 0] - Mapper.x_over_f*points_3d2[:, 2] + Mapper.x_offset*points_3d2[:, 2]
    #     points_3d2[:, 1] = 2*points_3d2[:, 1] - 0.275*Mapper.y_over_f*points_3d2[:, 2] - Mapper.y_offset*points_3d2[:, 2]

    #     self.points_3d2 = points_3d2

    #     points_3d2_flat = points_3d2_flat.T
    #     points_3d2_flat = np.abs(points_3d2_flat)
    #     points_3d2_flat[:, 0] = 2*points_3d2_flat[:, 0] - Mapper.x_over_f*points_3d2_flat[:, 2] + Mapper.x_offset*points_3d2_flat[:, 2]
    #     points_3d2_flat[:, 1] = 2*points_3d2_flat[:, 1] - 0.275*Mapper.y_over_f*points_3d2_flat[:, 2] - Mapper.y_offset*points_3d2_flat[:, 2]

    #     #----
    #     return points_3d1, points_3d1_flat, points_3d2, points_3d2_flat
    
    # Get lines for Rerun Visualization
    def get_lines(self):
        lines = [[Mapper.camera_center, pt] for pt in self.points_3d]
        return lines