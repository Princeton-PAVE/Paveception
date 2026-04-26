import rerun as rr
import trimesh
from depth_anything_3.api import DepthAnything3
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


rr.init("SLAM_Visualization")
rr.spawn()
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

d = "cpu"
model = DepthAnything3.from_pretrained("depth-anything/da3-base").to(d).eval()

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

VCLIP, HCLIP, ZCLIP = 0.2, 0.3, 0.5

try:
    while cap0.isOpened() and cap1.isOpened():
        ret, frame0 = cap0.read()
        ret, frame1 = cap1.read()
        if not ret: continue

        img_rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        img_rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        r = model.inference(image=[img_rgb0, img_rgb1], export_dir='./', export_format="glb", process_res=504, conf_thresh_percentile=0, num_max_points=1000000000)

        scene = trimesh.load("scene.glb")

        points = scene.geometry['geometry_0'].vertices
        colors = scene.geometry['geometry_0'].visual.vertex_colors

        clip1 = np.logical_and(points[:,0] > -HCLIP, points[:,0] < HCLIP)
        clip2 = np.logical_and(points[:,1] > -VCLIP, points[:,1] < VCLIP)
        clip3 = np.logical_and(points[:,2] > -ZCLIP, points[:,2] < ZCLIP)
        clip = np.logical_and(clip1, clip2)
        clip = np.logical_and(clip, clip3)
        points_to_cluster = points[clip]

        # --- Clustering and Bounding ---
        if len(points_to_cluster) > 10:
            clustering = DBSCAN(eps=0.025, min_samples=20).fit(points_to_cluster)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            if -1 in unique_labels: unique_labels.remove(-1)

            box_centers = []
            box_sizes = []
            box_colors = []

            for label in unique_labels:
                cluster_mask = (labels == label)
                cluster_points = points_to_cluster[cluster_mask]

                min_bound = np.min(cluster_points, axis=0)
                max_bound = np.max(cluster_points, axis=0)
                
                center = (min_bound + max_bound) / 2
                size = max_bound - min_bound
                
                box_centers.append(center)
                box_sizes.append(size)
                box_colors.append([(label * 50) % 255, (label * 80) % 255, (label * 120) % 255])
            if box_centers:
                rr.log(
                    "world/boxes",
                    rr.Boxes3D(
                        centers=box_centers,
                        sizes=box_sizes,
                        colors=box_colors
                    )
                )

        rr.log("world/point_cloud", rr.Points3D(points, colors=colors))

except KeyboardInterrupt as k:
    print("Stopping.")
finally:
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()