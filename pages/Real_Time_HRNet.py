import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
from hrnet import HRNetModel  # Assume you have an HRNet model implementation

# Define the model path
HRNET_MODEL_PATH = "hrnet_facial_landmarks.pth"

# Load HRNet model
def load_model():
    model = HRNetModel()
    model.load_state_dict(torch.load(HRNET_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def process_frame(frame, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(Image.fromarray(frame)).unsqueeze(0)

    with torch.no_grad():
        landmarks = model(image_tensor)
    landmarks = landmarks.squeeze().numpy()

    h, w, _ = frame.shape
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h

    for (x, y) in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

    return frame

st.title("Real-Time Facial Landmark Detection using HRNet")

model = load_model()

# Streamlit live webcam feed
run = st.checkbox("Enable Camera")
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture image")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = process_frame(frame, model)
    FRAME_WINDOW.image(processed_frame, channels="RGB")
else:
    cap.release()
