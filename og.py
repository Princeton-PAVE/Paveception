import cv2
from ultralytics import YOLO, YOLOE
import numpy as np
import pyrealsense2 as rs
import socket
import threading
import time

from rrt_planning.plan import fill_bev_matrix, get_controls_curvy
from rrt_planning.visualizer import BevEnv
from rrt_planning.planner import rrt_star

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

controls_lock = threading.Lock()
controls = []
current_control = 0

PLAN_WIDTH = 100
PLAN_HEIGHT = 100

GRID_SIZE = 1 # TODO units

def control_loop(angle_degrees=None, object_depth=None):
    while(True):
        with controls_lock:
            if(len(controls) <= current_control):
                # set steering angle and power to 0
                continue
            (power, steering_angle, time) = controls[current_control]

            # set steering angle and power

            current_control += 1
            time.sleep(time) # sleep takes seconds

def planning_loop():
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

        all_depth_frames.append(depth_image)

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if frame is not None:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=0.5)
            person_id = [k for k, v in model.names.items() if v == 'person'][0]
            result = results[0]

            # Visualize the results on the frame
            annotated_frame = result.plot()

            points = []
            people = []
        
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
                        if cls == person_id:
                            people.append((track_id, conf, object_depth))

                        print(f"Track ID: {track_id}, Class: {cls}, Confidence: {conf:.2f}, Depth: {object_depth:.2f}m")
                                    
            # Azimuth estimation

            color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

            fov = 2 * np.arctan2(color_intrinsics.width / 2, color_intrinsics.fx)  # horizontal FOV in radians

            angles = []

            for (box_mid_x, object_depth) in points:
                angle_degrees = find_azimuth(box_mid_x, object_depth, color_intrinsics)
                angles.append((angle_degrees, object_depth))
            # ADD CONTROL LOGIC HERE USING ANGLES
                # 1. verify that we have a human through class
                # 2. get the angles[] of human
                # 3. access angle_degrees & object_depth
                # control_loop(angles[ind])
            min_person = min(people, key=lambda x: x[2]) if people else None
            start = (PLAN_HEIGHT / 2, 0) # y, x

            # Find the goal coordinates from the closest person
            goal = None
            if min_person is not None:
                min_person_depth = min_person[2]
                # Find the point that matches this person's depth
                for (box_mid_x, object_depth) in points:
                    if abs(object_depth - min_person_depth) < 0.01:  # small tolerance for float comparison
                        # Convert pixel position to BEV coordinates (y, x in grid)
                        goal = (PLAN_HEIGHT / 2 + min_person_depth / GRID_SIZE, box_mid_x / GRID_SIZE)
                        break

            bev = fill_bev_matrix(None)  # uint8, 0/255

            env = BevEnv(bev)
            #path comes in as List of (y,x)'s
            if goal is not None:
                path, nodes = rrt_star(
                    env,
                    start=start,
                    goal=goal,
                    step_size=10.0,
                    radius=30.0,
                    max_iter=1250,
                    goal_thresh=15.0,
                    rebuild_every=10,
                    coord_order="rc",
                )
            else:
                path = None

            controls, init_state = None, None
            if path is not None: #sometimes there is no viable path so check this
                init_state = (start[0], start[1]) # starting position + velocity is zero, heading angle is zero (right)
                curr_control = get_controls_curvy(path) #not full path just these paths
                if curr_control:
                    print("curvy path!: " + str(controls))
                    # Use MPC - apply the first control
                    with controls_lock:
                        controls = curr_control
                        current_control = 0
                        # First control will be executed by control_loop
        

def find_azimuth(box_mid_x, object_depth, color_intrinsics):
    if object_depth > 0:
        # Calculate the angle from the center of the image
        angle = (box_mid_x - color_intrinsics.width / 2) / (color_intrinsics.width / 2) * (fov / 2)
        angle_degrees = np.degrees(angle)
        print(f"Object at depth {object_depth:.2f}m has an azimuth of {angle_degrees:.2f} degrees")
        return angle_degrees
    return None


threading.Thread(target=planning_loop, args=(), kwargs={}).start()
threading.Thread(target=control_loop, args=(), kwargs={}).start()

all_depth_frames = np.stack(all_depth_frames, axis=0)  # shape (num_frames, H, W)
print("Saving depth frames to depth_frames.npy...")
print(f"Depth frames shape: {all_depth_frames.shape}, dtype: {all_depth_frames.dtype}")
np.save("depth_frames.npy", all_depth_frames)

# Close the display window
cv2.destroyAllWindows()