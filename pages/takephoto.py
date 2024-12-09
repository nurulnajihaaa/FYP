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

uploaded = files.upload()

for fn in uploaded.keys():
  image_path = fn
  image = cv2.imread(image_path)

  with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
      print("No face detected in the image.")
      continue

    for face_landmarks in results.multi_face_landmarks:
        # Landmark indices based on Mediapipe's FaceMesh model (may require adjustments based on actual landmark positions).
        landmarks = {
            "Glabella": 10,
            "Pupil_left": 473,  # Approximation - may need refinement based on iris detection
            "Pupil_right": 468, # Approximation
            "Medial_Canthus_left": 33,
            "Medial_Canthus_right": 263,
            "Nasion": 168,
            "Otobasion_Inferius_left": 79,
            "Otobasion_Inferius_right": 317,
            "Subnasale": 2,
            "Pronasale": 4,
            "Cheilion_left": 61,
            "Cheilion_right": 291,
            "Menton": 152,
            "Gonion_left": 32,
            "Gonion_right": 321,
            "Preaureculare_left": 128,
            "Preaureculare_right": 356,
        }

        for label, idx in landmarks.items():
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(image, label, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Display or save the image
    cv2_imshow(image) #Requires a suitable environment for displaying images
    cv2.imwrite("output_image.jpg", image)
