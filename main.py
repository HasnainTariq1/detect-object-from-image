import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load YOLO model (pre-trained on COCO dataset)
@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')  

model = load_yolo_model()

# Title
st.title("Object Detection App")
st.write("Upload an image to detect objects in it.")


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")

    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # Grayscale image check
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    # Perform object detection
    results = model(image_np)

    # Render the results
    detected_image = np.squeeze(results.render())  # Get the image with detections
    st.image(detected_image, caption="Detected Objects", use_column_width=True)

    # Print object details
    st.write("Detection Details:")
    detections = results.pandas().xyxy[0]  # Bounding boxes and details as a pandas DataFrame
    st.write(detections)
