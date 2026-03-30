import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
model = model.to(device=device)

# Run inference on images
images = ["download.jpeg"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
post_proccsesed_images = prediction.processed_images 
prediction_depth = prediction.depth
print(prediction_depth)        # Depth maps: [N, H, W] float32
# conversion metric from gemini: f * (net_output/C) 

# grabbing Ns 
fx = prediction.intrinsics[:, 0, 0]  
fy = prediction.intrinsics[:, 1, 1]  
focal = (fx + fy) / 2 
focal = focal[:, None, None]

C = 300 # idk scaling factor from gemini
metric_depth = focal * (prediction_depth / C) 

print(metric_depth)          # Metric depth maps: [N, H, W] float32 


print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32 (open cv2 format)
print(prediction.intrinsics.shape)  # Camera intrinsics: [N, 3, 3] float32 


