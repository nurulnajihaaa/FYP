import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

st.title("Grayscale and Facial Landmark Detection with MediaPipe")
st.write("Upload an image to process:")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if img is not None:
        # Step 1: Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Process Facial Landmark with MediaPipe on the original (color) image
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
            results = face_mesh.process(image_rgb)

            # Annotate facial landmarks if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                # Display the original image with facial landmarks and the grayscale image
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image with Facial Landmarks', use_column_width=True)
                st.image(gray_img, caption='Grayscale Image', use_column_width=True)
            else:
                st.write("No faces detected in the image.")
    else:
        st.write("Error: Could not read the image. Please ensure it's a valid image format.")
