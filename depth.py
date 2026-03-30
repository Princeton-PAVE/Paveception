from transformers import pipeline
import cv2
import numpy as np
from ultralytics import YOLO, YOLOE
from PIL import Image

def rescale(input):
    result = input.copy().astype('float64')
    result -= np.amin(result)
    result /= np.amax(result)
    result *= 255
    return result

model = YOLOE("yoloe-26s-seg.pt")
names = ["person"]
emb = model.get_text_pe(names)
model.set_classes(names, emb)
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=shrink, fy=shrink)

    if not ret:
        break

    img = frame[:,:,::-1]
    disparity = np.array(pipe(Image.fromarray(img))["depth"])

    results = model.track(img, persist=True, conf=0.5)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    cv2.imshow('Depth Map', rescale(disparity).astype('uint8'))
    cv2.imshow('Annotated', annotated_frame[:,:,::-1])
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
