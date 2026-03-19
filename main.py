import cv2
from ultralytics import YOLO, YOLOE
from ultralytics.utils.plotting import Annotator
import pyrealsense2 as rs
import numpy as np
import time

model = YOLOE("yoloe-11l-seg.pt")
# We'll target the "person" class (COCO class name)
target_class_name = "person"

# Configure Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())  # BGR image
        depth_image = np.asanyarray(depth_frame.get_data()).astype(float) * depth_scale
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

        # Run the model (use a small resize if needed)
        results = model(frame, conf=0.25, device='cpu')
        print()
        r = results[0]
        
        

        # Prepare arrays (handles both torch Tensor or numpy)
        boxes = []
        classes = []
        confidences = []
        if hasattr(r, "boxes") and r.boxes is not None:
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy()
            except Exception:
                boxes = np.array(r.boxes.xyxy)
                classes = np.array(r.boxes.cls).astype(int)
                confidences = np.array(r.boxes.conf)

        masks = None
        if hasattr(r, "masks") and r.masks is not None:
            try:
                masks = r.masks.data.cpu().numpy()  # shape: (N, H, W)
            except Exception:
                masks = np.array(r.masks.data)

        annotated = r.plot()

        for i, cls in enumerate(classes):
            print(f"i={i}, cls={cls}")
            name = model.model.names[cls] if hasattr(model, "model") and hasattr(model.model, "names") else model.names.get(cls, str(cls))
            if name != target_class_name:
                continue

            # Draw mask if available
            if masks is not None and i < masks.shape[0]:
                mask = masks[i].astype(bool)
                color = (0, 255, 0)  # green for person
                colored_mask = np.zeros_like(annotated, dtype=np.uint8)
                colored_mask[mask] = color
                annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.5, 0)

            # Draw bounding box and label
            x1, y1, x2, y2 = boxes[i].astype(int)
            conf = confidences[i] if len(confidences) > i else 0.0
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

            # Get median distance
            if masks is not None and i < masks.shape[0]:
                mask = masks[i].astype(bool)
                # Resize mask to match depth/color frame if necessary
                # Lowkey shouldnt need
                h, w = depth_image.shape[:2]
                if mask.shape[:2] != (h, w):
                    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask_resized = mask

                # Extract valid depth values under the mask
                # mask_resized = mask

                masked_depth = depth_image[mask_resized]
                masked_depth = masked_depth[masked_depth > 0]
                median_distance = float('nan') if masked_depth.size == 0 else float(np.median(masked_depth))

                # Compute center of mass of the mask (in pixel coordinates)
                ys, xs = np.where(mask_resized)
                if xs.size:
                    center_x = xs.mean()
                    center_y = ys.mean()
                    cv2.circle(annotated, (int(center_x), int(center_y)), 4, (0,0,255), -1)

                    # Horizontal angle in degrees relative to camera principal point using intrinsics
                    angle_rad = np.arctan2((center_x - color_intrinsics.ppx), color_intrinsics.fx)
                    angle_deg = np.degrees(angle_rad)
                else:
                    center_x = center_y = np.nan
                    angle_deg = np.nan

                # Annotate median distance and angle
                info = f"{median_distance:.2f}m {angle_deg:+.1f}deg"
                cv2.putText(annotated, info, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

        cv2.imshow("Paveception - RealSense (press q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()