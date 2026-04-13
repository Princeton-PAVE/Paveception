import cv2
import numpy as np
import pyrealsense2 as rs

# ---------- RealSense setup ----------
pipeline = rs.pipeline()
ctx = rs.context()

serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
print(f"Serials: {serials}")

config = rs.config()
config.enable_device(serials[0])
config.enable_stream(rs.stream.depth)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.laser_power, 360)
depth_scale = depth_sensor.get_depth_scale()

# Depth post-processing filters
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()

# Grab intrinsics from the depth stream
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
depth_intrinsics = depth_stream.get_intrinsics()
W, H = depth_intrinsics.width, depth_intrinsics.height
fx = depth_intrinsics.fx
hfov = 2 * np.arctan2(W / 2, fx)  # horizontal FOV in radians
print(f"Depth stream: {W}x{H}, hFOV={np.degrees(hfov):.1f} deg")

# ---------- Sampling grid (100 points) ----------
# A 10x10 grid across the depth image; we'll take the median depth in each cell
# and use the cell's center x for azimuth. This gives 100 (angle, distance) samples.
GRID_COLS = 10
GRID_ROWS = 10
NUM_SAMPLES = GRID_COLS * GRID_ROWS

# Precompute cell boundaries
col_edges = np.linspace(0, W, GRID_COLS + 1, dtype=int)
row_edges = np.linspace(0, H, GRID_ROWS + 1, dtype=int)
col_centers = ((col_edges[:-1] + col_edges[1:]) / 2).astype(int)

# ---------- Radar config ----------
RADAR_RES = (1280, 720)
RADAR_BG = (20, 20, 20)
RADAR_MAX_RANGE_M = 6.0
RADAR_RING_FRACTIONS = np.linspace(0.25, 1.0, 4)

moveRadarWindow = True
moveDepthWindow = True

all_depth_frames = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Apply filters (disparity domain is recommended)
        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        all_depth_frames.append(depth_image)

        # ---------- Sample 100 points from depth map ----------
        samples = []  # list of (angle_deg, depth_m, px, py)
        for r in range(GRID_ROWS):
            y0, y1 = row_edges[r], row_edges[r + 1]
            for c in range(GRID_COLS):
                x0, x1 = col_edges[c], col_edges[c + 1]
                cell = depth_image[y0:y1, x0:x1]
                valid = cell[(cell > 0) & (cell < RADAR_MAX_RANGE_M * 2)]
                if valid.size == 0:
                    continue
                d = float(np.median(valid))
                cx_pix = col_centers[c]
                # Azimuth: map pixel x to angle using horizontal FOV
                angle = (cx_pix - W / 2) / (W / 2) * (hfov / 2)
                angle_deg = float(np.degrees(angle))
                cy_pix = int((y0 + y1) / 2)
                samples.append((angle_deg, d, cx_pix, cy_pix))

        # ---------- Depth visualization ----------
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=255.0 / RADAR_MAX_RANGE_M),
            cv2.COLORMAP_JET,
        )
        # Overlay sample points
        for (_, _, px, py) in samples:
            cv2.circle(depth_vis, (px, py), 3, (255, 255, 255), -1)

        cv2.imshow("Depth", depth_vis)
        if moveDepthWindow:
            cv2.moveWindow("Depth", 700, 0)
            moveDepthWindow = False

        # ---------- Radar rendering ----------
        radar_w, radar_h = RADAR_RES
        radar_img = np.full((radar_h, radar_w, 3), RADAR_BG, dtype=np.uint8)

        cx, cy = radar_w // 2, radar_h - int(0.02 * radar_h)
        max_radius = int(min(cx * 0.95, cy * 0.9))
        ring_color = (80, 80, 80)
        thickness = max(1, radar_w // 1000)
        font_scale = max(0.45, radar_w / 2500)

        for r_frac in RADAR_RING_FRACTIONS:
            r_pix = int(max_radius * r_frac)
            cv2.ellipse(radar_img, (cx, cy), (r_pix, r_pix), 0, 180, 360, ring_color, thickness)
            dist_label = f"{r_frac * RADAR_MAX_RANGE_M:.1f}m"
            txt_x = cx + r_pix - int(0.04 * radar_w)
            txt_y = cy - int(0.02 * radar_h)
            cv2.putText(radar_img, dist_label, (txt_x, txt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, ring_color, thickness)

        cv2.line(radar_img,
                 (int(0.01 * radar_w), cy),
                 (radar_w - int(0.01 * radar_w), cy),
                 (40, 40, 40), thickness)

        # Plot each sample on the radar. Color by distance (closer = hotter).
        point_radius = max(3, radar_w // 400)
        for (angle_deg, depth_m, _, _) in samples:
            d = min(max(depth_m, 0.0), RADAR_MAX_RANGE_M)
            r = int((d / RADAR_MAX_RANGE_M) * max_radius)
            theta = np.radians(angle_deg)
            px = int(cx + r * np.sin(theta))
            py = int(cy - r * np.cos(theta))
            # Hotter (red) when close, cooler (green) when far
            t = d / RADAR_MAX_RANGE_M
            color = (0, int(200 * t), int(200 * (1 - t)))  # BGR
            cv2.circle(radar_img, (px, py), point_radius, color, -1)

        # Info overlay
        cv2.putText(radar_img, f"Samples: {len(samples)}/{NUM_SAMPLES}",
                    (int(0.01 * radar_w), int(0.05 * radar_h)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)

        cv2.imshow("Radar", radar_img)
        if moveRadarWindow:
            cv2.moveWindow("Radar", 0, 500)
            moveRadarWindow = False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if all_depth_frames:
        all_depth_frames = np.stack(all_depth_frames, axis=0)
        print(f"Saving depth_frames.npy, shape={all_depth_frames.shape}, dtype={all_depth_frames.dtype}")
        np.save("depth_frames.npy", all_depth_frames)