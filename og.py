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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

while True:
    # Read a frame from the video
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame is not None:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

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