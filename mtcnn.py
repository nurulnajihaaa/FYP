import streamlit as st
from mtcnn_cv2 import MTCNN
from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image

# Streamlit App Title
st.title("Face Detection with MTCNN")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to NumPy array
    image_np = np.array(image)

    try:
        # Process the image with MTCNN
        detector = MTCNN()
        bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        faces = detector.detect_faces(bgr_image)

        # Draw detections
        for face in faces:
            x, y, width, height = face['box']
            keypoints = face['keypoints']

            # Draw bounding box
            cv2.rectangle(bgr_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Annotate facial landmarks
            for keypoint_name, (x_coord, y_coord) in keypoints.items():
                cv2.circle(bgr_image, (x_coord, y_coord), 2, (0, 0, 255), 2)
                cv2.putText(bgr_image, keypoint_name, (x_coord, y_coord),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Convert the processed BGR image back to RGB for display
        annotated_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Display the annotated image
        st.subheader("Processed Image")
        st.image(annotated_image, caption="Detected Faces and Landmarks", use_column_width=True)

    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an image to start detection.")
