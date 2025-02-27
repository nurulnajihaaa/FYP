import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import sqlite3
from PIL import Image

# Initialize database
conn = sqlite3.connect("face_mesh.db")
c = conn.cursor()

# Create table for storing face mesh landmarks
c.execute('''
    CREATE TABLE IF NOT EXISTS face_mesh (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        landmark_index INTEGER,
        x REAL,
        y REAL,
        z REAL
    )
''')
conn.commit()

# Streamlit App Title
st.title("Face Mesh Detection with MediaPipe and Database Storage")

# Camera Input or File Upload
img_file_buffer = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
elif uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
else:
    st.info("Please upload an image or capture one using the camera.")
    st.stop()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        annotated_image = image.copy()
        
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                c.execute("INSERT INTO face_mesh (landmark_index, x, y, z) VALUES (?, ?, ?, ?)", (idx, x, y, z))
                conn.commit()
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
            )
        
        # Convert BGR to RGB for Streamlit display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.subheader("Annotated Image with Facial Mesh")
        st.image(annotated_image_rgb, caption="Detected Face Mesh", use_column_width=True)
        st.success("Face Mesh detection completed and data saved to database!")
    else:
        st.warning("No face detected. Please upload a clearer image.")

# Display stored data
st.subheader("Stored Face Mesh Data")
if st.button("Show Database Records"):
    c.execute("SELECT * FROM face_mesh")
    records = c.fetchall()
    for record in records:
        st.write(record)

# Close database connection
conn.close()
