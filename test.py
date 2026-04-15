import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch import device, cuda
from depth_anything_3.api import DepthAnything3
import numpy as np
import matplotlib.pyplot as plt

# Load model from Hugging Face Hub
d = device("cuda" if cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-small")
model = model.to(device=d)

# Run inference on images
images = ["semidetailed_room.jpg"]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(images)

# Access results
print(type(prediction.depth))
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32

result = np.squeeze(prediction.depth, axis=0)
result = np.expand_dims(result, axis=-1)
plt.imshow(result)
plt.show()
