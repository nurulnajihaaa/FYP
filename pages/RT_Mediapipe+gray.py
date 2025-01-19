import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

#enable = st.checkbox("Enable camera")
#picture = st.camera_input("Take a picture", disabled=not enable)

#if picture:
#    st.image(picture)

# Step 1: Camera Input or File Uploader
img_file_buffer = st.camera_input("Take a picture")  # Capture from camera
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    #st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    #st.write(cv2_img.shape)

    # Convert the image to a NumPy array
    image_np = np.array(cv2_img)
    st.write("Image successfully uploaded and processed as a NumPy array.")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Streamlit App Title
st.title("Face Mesh Detection with MediaPipe")

# Load image (from camera or file uploader)
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
elif uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
else:
    st.info("Please upload an image or capture one using the camera.")
    st.stop()

# Step 2: Process the image with MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        annotated_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the face
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )

        # Convert BGR to RGB for Streamlit display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Display the annotated image
        st.subheader("Annotated Image with Facial Mesh")
        st.image(annotated_image_rgb, caption="Detected Face Mesh", use_column_width=True)

    else:
        st.warning("No face detected. Please upload a clearer image.")

# Success Message
st.success("Face Mesh detection completed successfully!")

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title of the Streamlit app
st.title("Image Grayscale Conversion")

# Upload the image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file as a byte stream
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if img is not None:
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display the original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

        # Display the grayscale image
        st.subheader("Grayscale Image")
        st.image(gray_img, channels="GRAY")
    else:
        st.error("Error: Could not read the image. Please ensure it's a valid image format.")
