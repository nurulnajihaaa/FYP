import streamlit as st
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
    st.write("Image successfully uploaded and processed as a NumPy array.")
    # Convert the image to a NumPy array
    image_np = np.array(image)
else:
    st.info("Please upload an image to display.")

#else:
   # st.info("Please upload an image to display and convert.")

from mtcnn import MTCNN
import cv2
from google.colab.patches import cv2_imshow

try:
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    for face in faces:
        x, y, width, height = face['box']
        keypoints = face['keypoints']

        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Annotate facial landmarks
        for keypoint_name, (x_coord, y_coord) in keypoints.items():
            cv2.circle(image, (x_coord, y_coord), 2, (0, 0, 255), 2)
            cv2.putText(image, keypoint_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2_imshow(image)

except Exception as e:
    print(f"Error processing the image with MTCNN: {e}")
