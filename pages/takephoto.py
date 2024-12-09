import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

#enable = st.checkbox("Enable camera")
#picture = st.camera_input("Take a picture", disabled=not enable)

#if picture:
#    st.image(picture)


img_file_buffer = st.camera_input("Take a picture")

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

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

