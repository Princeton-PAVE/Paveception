import cv2
from ultralytics import YOLO, YOLOE
import numpy as np
import pyrealsense2 as rs

# Load the YOLO11 model
# model = YOLO("/Users/mayanksengupta/Desktop/CV_Training/runs/segment/train3/weights/best.pt")

model = YOLOE("yoloe-11l-seg.pt")
names = ["person"]
model.set_classes(names, model.get_text_pe(names))

# Loop through the video frames
moveYOLOWindow = True
moveMaskWindow = True
i = 0
interval = 1

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

while True:
    # Read a frame from the video
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame: continue
    
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
                    object_depth = np.mean(masked_depth[masked_depth > 0])  # Average

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
        

        # --- Radar (half-circle) view ---
        # Configure radar appearance
        radar_w, radar_h = 600, 220
        radar_bg = (20, 20, 20)
        radar_img = np.full((radar_h, radar_w, 3), radar_bg, dtype=np.uint8)
        cx, cy = radar_w // 2, radar_h - 10
        max_radius = min(cx - 10, cy - 10)
        max_range_m = 6.0  # meters shown on radar (adjust)

        # draw semicircle arcs / range rings
        ring_color = (80, 80, 80)
        for r_frac in np.linspace(0.25, 1.0, 4):
            r_pix = int(max_radius * r_frac)
            cv2.ellipse(radar_img, (cx, cy), (r_pix, r_pix), 0, 180, 360, ring_color, 1)
            # label ring with distance
            dist_label = f"{r_frac * max_range_m:.1f}m"
            cv2.putText(radar_img, dist_label, (cx + r_pix, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ring_color, 1)

        # draw center base line
        cv2.line(radar_img, (10, cy), (radar_w - 10, cy), (40,40,40), 1)

        # Plot detections onto radar
        for (angle_deg, depth_m) in angles:
            # clamp distance
            d = min(max(depth_m, 0.0), max_range_m)
            r = int((d / max_range_m) * max_radius)
            theta = np.radians(angle_deg)  # right-positive from center
            # convert polar to image coords (0deg -> up)
            px = int(cx + r * np.sin(theta))
            py = int(cy - r * np.cos(theta))
            color = (0, 200, 0)
            cv2.circle(radar_img, (px, py), 6, color, -1)
            txt = f"{d:.1f}m"
            cv2.putText(radar_img, txt, (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # show combined windows
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

# Close the display window
cv2.destroyAllWindows()