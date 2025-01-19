import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Streamlit App Title
st.title("Webcam Image Capture and Processing")

# Step 1: Capture image from webcam
img_file_buffer = st.camera_input("Take a picture using your webcam")

if img_file_buffer is not None:
    # Step 2: Read the image as a NumPy array
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if image is not None:
        # Step 3: Process the image with MediaPipe Face Mesh
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

                # Step 4: Convert the image to grayscale
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Display the annotated image with facial landmarks
                st.subheader("Annotated Image with Facial Mesh")
                st.image(annotated_image_rgb, caption="Detected Face Mesh", use_column_width=True)

                # Display the grayscale image
                st.subheader("Grayscale Image")
                st.image(gray_img, caption="Grayscale Image", channels="GRAY", use_column_width=True)

            else:
                st.warning("No face detected. Please try again with a clearer image.")
    else:
        st.error("Error: Could not process the image. Please try again.")
