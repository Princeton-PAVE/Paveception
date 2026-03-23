import cv2
from ultralytics import YOLO, YOLOE
import numpy as np
import pyrealsense2 as rs

# Load the YOLO11 model
# model = YOLO("/Users/mayanksengupta/Desktop/CV_Training/runs/segment/train3/weights/best.pt")

model = YOLOE("yoloe-11l-seg.pt")
model = model.to("cuda")
names = ["chair"]
model.set_classes(names, model.get_text_pe(names))

# Loop through the video frames
moveYOLOWindow = True
moveMaskWindow = True
i = 0
interval = 1

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
# depth_sensor.set_option(rs.option.exposure, 1000)  # microseconds, lower = faster shutter
# depth_sensor.set_option(rs.option.enable_auto_exposure, 0)  # disable auto exposure
depth_sensor.set_option(rs.option.laser_power, 360)
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
 
spatial_filter  = rs.spatial_filter()   # smooths edges, fills small holes
temporal_filter = rs.temporal_filter()  # reduces flicker/noise across frames

all_depth_frames = []

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

    # all_depth_frames.append(depth_frame)
    
    frame_bgr = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

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

                    masked_depth = depth_image_resized * mask
                    object_depth = np.median(masked_depth[masked_depth > 0])  # Average

                    points.append((box_mid_x, object_depth))

                    print(f"Track ID: {track_id}, Class: {cls}, Confidence: {conf:.2f}, Depth: {object_depth:.2f}m")
                    
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

        cv2.imshow("Radar", radar_img)
        if moveMaskWindow:
            cv2.moveWindow("Radar", 0, 500)
            moveMaskWindow = False

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame[:,:,::-1])
        if moveYOLOWindow:
            cv2.moveWindow("YOLO11 Tracking", 700, 0)
            moveYOLOWindow=False

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop video feed is cut
        break

all_depth_frames = np.stack(all_depth_frames, axis=0)  # shape (num_frames, H, W)
np.save("depth_frames.npy", all_depth_frames)

# Close the display window
cv2.destroyAllWindows()