import cv2
from ultralytics import YOLO, YOLOE
import numpy as np
import pyrealsense2 as rs

try:
    import rerun as rr
except ImportError:
    rr = None


def depth_to_xyz(depth_image, intrinsics):
    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_image
    x = (u - intrinsics.ppx) * z / intrinsics.fx
    y = (v - intrinsics.ppy) * z / intrinsics.fy
    return x, y, z


def fit_plane_from_points(points):
    p1, p2, p3 = points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    d = -np.dot(normal, p1)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)


def refine_plane_svd(inlier_points):
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)


def fit_ground_plane_ransac(points, iterations=150, dist_thresh=0.05):
    if points.shape[0] < 3:
        return None

    best_plane = None
    best_inlier_mask = None
    best_inlier_count = 0
    rng = np.random.default_rng()

    for _ in range(iterations):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        plane = fit_plane_from_points(points[idx])
        if plane is None:
            continue

        a, b, c, d = plane
        distances = np.abs(points @ np.array([a, b, c]) + d)
        inlier_mask = distances < dist_thresh
        inlier_count = int(np.count_nonzero(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_plane = plane

    if best_plane is None or best_inlier_mask is None or best_inlier_count < 3:
        return None

    refined_plane = refine_plane_svd(points[best_inlier_mask])
    return refined_plane


def voxel_downsample(points, voxel_size):
    if points.shape[0] == 0:
        return points
    vox = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def dbscan_3d(points, eps=0.25, min_samples=12):
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    eps2 = eps * eps

    def region_query(idx):
        delta = points - points[idx]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        return np.where(dist2 <= eps2)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)

        if neighbors.size < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed = list(neighbors.tolist())
        seed_set = set(seed)
        k = 0
        while k < len(seed):
            j = seed[k]
            if not visited[j]:
                visited[j] = True
                nbr_j = region_query(j)
                if nbr_j.size >= min_samples:
                    for q in nbr_j.tolist():
                        if q not in seed_set:
                            seed.append(q)
                            seed_set.add(q)
            if labels[j] == -1:
                labels[j] = cluster_id
            k += 1

        cluster_id += 1

    return labels


def extract_obstacle_clusters(collision_points, eps_m, min_samples, voxel_size_m, min_size_m):
    points_ds = voxel_downsample(collision_points, voxel_size_m)
    labels = dbscan_3d(points_ds, eps=eps_m, min_samples=min_samples)
    obstacles = []
    for label in np.unique(labels):
        if label < 0:
            continue
        cluster = points_ds[labels == label]
        if cluster.shape[0] == 0:
            continue

        centroid = np.mean(cluster, axis=0)
        extents = np.ptp(cluster, axis=0)
        size_m = float(np.max(extents))
        if size_m < min_size_m:
            continue

        obstacles.append({
            "centroid": centroid,
            "size_m": size_m,
            "num_points": int(cluster.shape[0]),
        })
    return obstacles

# Load the YOLO11 model
# model = YOLO("/Users/mayanksengupta/Desktop/CV_Training/runs/segment/train3/weights/best.pt")

model = YOLOE("yoloe-11l-seg.pt")
model = model.to("cuda")
names = ["person"]
model.set_classes(names, model.get_text_pe(names))

# Loop through the video frames
moveYOLOWindow = True
moveMaskWindow = True
moveDepthWindow = True
moveGroundMaskWindow = True
moveCollisionMaskWindow = True
i = 0
interval = 1

pipeline = rs.pipeline()
ctx = rs.context()

serials = [ d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
print(f"Serials: {serials}")
config = rs.config()
config.enable_device(serials[0])
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
# depth_sensor.set_option(rs.option.exposure, 1000)  # microseconds, lower = faster shutter
# depth_sensor.set_option(rs.option.enable_auto_exposure, 0)  # disable auto exposure
depth_sensor.set_option(rs.option.laser_power, 360)
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
 
spatial_filter  = rs.spatial_filter()   # smooths edges, fills small holes
temporal_filter = rs.temporal_filter()  # reduces flicker/noise across frames

GROUND_MIN_DEPTH_M = 0.5
GROUND_RANSAC_THRESH_M = 0.05
GROUND_RANSAC_ITERS = 150
GROUND_MAX_DEPTH_M = 8.0
GROUND_MASK_RATIO = 0.50
ABOVE_GROUND_MAX_HEIGHT_M = 0.6

CLUSTER_EPS_M = 0.28
CLUSTER_MIN_SAMPLES = 14
CLUSTER_VOXEL_SIZE_M = 0.06
OBSTACLE_MIN_SIZE_M = 0.15
OBSTACLE_DRAW_SIZE_SCALE_M = 0.6
CLUSTER_MAX_POINTS_FOR_RUNTIME = 6000

USE_RERUN = False
RERUN_SPAWN_VIEWER = True
RERUN_MAX_POINTS = 20000


def maybe_downsample_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx]

all_depth_frames = []
frame_idx = 0

if USE_RERUN and rr is not None:
    rr.init("paveception", spawn=RERUN_SPAWN_VIEWER)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)

while True:
    # Read a frame from the video
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial_filter.process(depth_frame)
    depth_frame = temporal_filter.process(depth_frame)
    depth_frame = disparity_to_depth.process(depth_frame)

    if not color_frame or not depth_frame: continue
    
    frame_bgr = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

    x3d, y3d, z3d = depth_to_xyz(depth_image, depth_intrinsics)
    h, w = depth_image.shape
    bottom_mask = np.zeros((h, w), dtype=bool)
    bottom_mask[int(h * GROUND_MASK_RATIO):, :] = True

    candidate_mask = (
        bottom_mask
        & (z3d > GROUND_MIN_DEPTH_M)
        & (z3d < GROUND_MAX_DEPTH_M)
    )
    candidate_points = np.column_stack((x3d[candidate_mask], y3d[candidate_mask], z3d[candidate_mask]))

    plane = fit_ground_plane_ransac(
        candidate_points,
        iterations=GROUND_RANSAC_ITERS,
        dist_thresh=GROUND_RANSAC_THRESH_M,
    )

    ground_mask = np.zeros_like(depth_image, dtype=bool)
    collision_mask = np.zeros_like(depth_image, dtype=bool)
    if plane is not None:
        a, b, c, d = plane
        distances = np.abs(a * x3d + b * y3d + c * z3d + d)
        valid_for_ground = (z3d > GROUND_MIN_DEPTH_M) & (z3d < GROUND_MAX_DEPTH_M)
        ground_mask = valid_for_ground & (distances < GROUND_RANSAC_THRESH_M)
        collision_mask = valid_for_ground & (~ground_mask) & (distances < ABOVE_GROUND_MAX_HEIGHT_M)

    collision_points = np.column_stack((x3d[collision_mask], y3d[collision_mask], z3d[collision_mask]))
    if collision_points.shape[0] > CLUSTER_MAX_POINTS_FOR_RUNTIME:
        sample_idx = np.random.choice(collision_points.shape[0], CLUSTER_MAX_POINTS_FOR_RUNTIME, replace=False)
        collision_points = collision_points[sample_idx]

    obstacle_clusters = extract_obstacle_clusters(
        collision_points,
        eps_m=CLUSTER_EPS_M,
        min_samples=CLUSTER_MIN_SAMPLES,
        voxel_size_m=CLUSTER_VOXEL_SIZE_M,
        min_size_m=OBSTACLE_MIN_SIZE_M,
    )

    if USE_RERUN and rr is not None:
        rr.set_time_sequence("frame", frame_idx)
        rr.log("camera/color", rr.Image(frame_bgr))
        rr.log("camera/depth", rr.DepthImage(depth_image, meter=1.0))
        rr.log("camera/mask/ground", rr.SegmentationImage(ground_mask.astype(np.uint8)))
        rr.log("camera/mask/collision", rr.SegmentationImage(collision_mask.astype(np.uint8)))

        ground_points = np.column_stack((x3d[ground_mask], y3d[ground_mask], z3d[ground_mask]))
        ground_points = maybe_downsample_points(ground_points, RERUN_MAX_POINTS)
        if ground_points.shape[0] > 0:
            rr.log("world/points/ground", rr.Points3D(ground_points, colors=[80, 220, 80], radii=0.01))

        collision_points_log = np.column_stack((x3d[collision_mask], y3d[collision_mask], z3d[collision_mask]))
        collision_points_log = maybe_downsample_points(collision_points_log, RERUN_MAX_POINTS)
        if collision_points_log.shape[0] > 0:
            rr.log("world/points/collision", rr.Points3D(collision_points_log, colors=[255, 120, 120], radii=0.012))

        if obstacle_clusters:
            centers = np.array([o["centroid"] for o in obstacle_clusters], dtype=np.float32)
            sizes = np.array([o["size_m"] for o in obstacle_clusters], dtype=np.float32)
            rr.log(
                "world/obstacles/centroids",
                rr.Points3D(
                    centers,
                    colors=np.tile(np.array([[80, 120, 255]], dtype=np.uint8), (len(centers), 1)),
                    radii=np.clip(0.5 * sizes, 0.05, 0.6),
                ),
            )
        rr.log("metrics/obstacle_count", rr.Scalars([len(obstacle_clusters)]))

    all_depth_frames.append(depth_image)

    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if frame is not None:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.5)
        result = results[0]

        # Visualize the results on the frame
        annotated_frame = result.plot()

        points = []
    
        if result.boxes is not None:
            number_of_boxes = len(result.boxes)

            for i, box in enumerate(result.boxes):
                track_id = int(box.id) if box.id is not None else None
                conf     = float(box.conf)
                cls      = int(box.cls)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_mid_x, box_mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cv2.circle(annotated_frame, (box_mid_x, box_mid_y), 5, (0, 255, 0), -1)

                if result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()  # shape (H, W)
                    print(mask.shape, depth_image.shape)
                    # reshape the depth image to match the mask if needed
                    depth_image_resized = cv2.resize(depth_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    ground_mask_resized = cv2.resize(
                        ground_mask.astype(np.uint8),
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    collision_mask_resized = cv2.resize(
                        collision_mask.astype(np.uint8),
                        (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                    mask_bool = mask > 0.5
                    valid_obj_pixels = (
                        mask_bool
                        & collision_mask_resized
                        & (depth_image_resized > GROUND_MIN_DEPTH_M)
                    )
                    valid_depth_values = depth_image_resized[valid_obj_pixels]

                    if valid_depth_values.size > 0:
                        object_depth = float(np.median(valid_depth_values))
                        points.append((box_mid_x, object_depth))
                        print(f"Track ID: {track_id}, Class: {cls}, Confidence: {conf:.2f}, Depth: {object_depth:.2f}m")
                    else:
                        print(f"Track ID: {track_id}, Class: {cls}, Confidence: {conf:.2f}, Depth: NaN (no non-ground depth)")
                    
        # Azimuth estimation

        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

        fov = 2 * np.arctan2(color_intrinsics.width / 2, color_intrinsics.fx)  # horizontal FOV in radians

        angles = []

        for (box_mid_x, object_depth) in points:
            if object_depth > 0:
                # Calculate the angle from the center of the image
                angle = (box_mid_x - color_intrinsics.width / 2) / (color_intrinsics.width / 2) * (fov / 2)
                angle_degrees = np.degrees(angle)
                print(f"Object at depth {object_depth:.2f}m has an azimuth of {angle_degrees:.2f} degrees")
                angles.append((angle_degrees, object_depth))

        # --- Radar (half-circle) view (scalable to 720p) ---
        # Configuration: change RADAR_RES or RADAR_MAX_RANGE_M to resize/scale
        RADAR_RES = (1280, 720)               # target radar resolution (width, height)
        RADAR_BG = (20, 20, 20)
        RADAR_MAX_RANGE_M = 6.0               # meters shown on radar
        RADAR_RING_FRACTIONS = np.linspace(0.25, 1.0, 4)

        radar_w, radar_h = RADAR_RES
        radar_img = np.full((radar_h, radar_w, 3), RADAR_BG, dtype=np.uint8)

        # center and sizing for semicircle
        cx, cy = radar_w // 2, radar_h - int(0.02 * radar_h)   # small bottom margin
        max_radius = int(min(cx * 0.95, cy * 0.9))
        ring_color = (80, 80, 80)

        # draw semicircle arcs / range rings (line thickness & font scale adapt to size)
        thickness = max(1, radar_w // 1000)
        font_scale = max(0.45, radar_w / 2500)
        for r_frac in RADAR_RING_FRACTIONS:
            r_pix = int(max_radius * r_frac)
            cv2.ellipse(radar_img, (cx, cy), (r_pix, r_pix), 0, 180, 360, ring_color, thickness)
            dist_label = f"{r_frac * RADAR_MAX_RANGE_M:.1f}m"
            txt_x = cx + r_pix - int(0.04 * radar_w)
            txt_y = cy - int(0.02 * radar_h)
            cv2.putText(radar_img, dist_label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ring_color, thickness)

        # draw center base line
        cv2.line(radar_img, (int(0.01 * radar_w), cy), (radar_w - int(0.01 * radar_w), cy), (40, 40, 40), thickness)

        # Plot detections onto radar
        point_radius = max(4, radar_w // 320)
        txt_offset_x = int(0.01 * radar_w)
        txt_offset_y = int(0.01 * radar_h)
        for (angle_deg, depth_m) in angles:
            # clamp distance and convert to pixel radius
            d = min(max(depth_m, 0.0), RADAR_MAX_RANGE_M)
            r = int((d / RADAR_MAX_RANGE_M) * max_radius)
            theta = np.radians(angle_deg)  # right-positive from center
            px = int(cx + r * np.sin(theta))
            py = int(cy - r * np.cos(theta))
            color = (0, 200, 0)
            cv2.circle(radar_img, (px, py), point_radius, color, -1)
            txt = f"{d:.1f}m"
            cv2.putText(radar_img, txt, (px + txt_offset_x, py - txt_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Plot clustered obstacle candidates as blue circles (size reflects 3D obstacle extent).
        for obs in obstacle_clusters:
            cx3, _, cz3 = obs["centroid"]
            if cz3 <= 0:
                continue
            angle_deg = float(np.degrees(np.arctan2(cx3, cz3)))
            depth_m = float(cz3)
            d = min(max(depth_m, 0.0), RADAR_MAX_RANGE_M)
            r = int((d / RADAR_MAX_RANGE_M) * max_radius)
            theta = np.radians(angle_deg)
            px = int(cx + r * np.sin(theta))
            py = int(cy - r * np.cos(theta))

            size_m = obs["size_m"]
            cluster_radius = int(np.clip((size_m / OBSTACLE_DRAW_SIZE_SCALE_M) * point_radius * 1.8, point_radius, point_radius * 4))
            cv2.circle(radar_img, (px, py), cluster_radius, (255, 0, 0), 2)
            cv2.putText(
                radar_img,
                f"{size_m:.2f}m",
                (px + txt_offset_x, py + int(1.5 * txt_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 120, 120),
                thickness,
            )

        cv2.putText(
            radar_img,
            f"Clusters: {len(obstacle_clusters)}",
            (int(0.01 * radar_w), int(0.09 * radar_h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (220, 180, 180),
            thickness,
        )

        cv2.imshow("Radar", radar_img)
        if moveMaskWindow:
            cv2.moveWindow("Radar", 0, 500)
            moveMaskWindow = False

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame[:,:,::-1])
        if moveYOLOWindow:
            cv2.moveWindow("YOLO11 Tracking", 700, 0)
            moveYOLOWindow=False
        
        cv2.imshow("Depth", cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255.0 / RADAR_MAX_RANGE_M), cv2.COLORMAP_JET))
        if moveDepthWindow:
            cv2.moveWindow("Depth", 700, 500)
            moveDepthWindow = False

        ground_vis = np.zeros_like(depth_image, dtype=np.uint8)
        ground_vis[ground_mask] = 255
        cv2.imshow("Ground Mask", ground_vis)
        if moveGroundMaskWindow:
            cv2.moveWindow("Ground Mask", 0, 0)
            moveGroundMaskWindow = False

        collision_vis = np.zeros_like(depth_image, dtype=np.uint8)
        collision_vis[collision_mask] = 255
        cv2.imshow("Collision Mask", collision_vis)
        if moveCollisionMaskWindow:
            cv2.moveWindow("Collision Mask", 350, 0)
            moveCollisionMaskWindow = False

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop video feed is cut
        break

    frame_idx += 1

all_depth_frames = np.stack(all_depth_frames, axis=0)  # shape (num_frames, H, W)
print("Saving depth frames to depth_frames.npy...")
print(f"Depth frames shape: {all_depth_frames.shape}, dtype: {all_depth_frames.dtype}")
np.save("depth_frames.npy", all_depth_frames)

# Close the display window
cv2.destroyAllWindows()