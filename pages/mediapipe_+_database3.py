import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import datetime
from PIL import Image

# Initialize database
def init_db():
    conn = sqlite3.connect("face_landmarks.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS face_landmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            landmark_index INTEGER,
            x REAL,
            y REAL,
            z REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()  # Run once at the start

# Streamlit App
st.title("Real-Time Facial Landmark Detection")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Capture Image
img_file_buffer = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer:
    bytes_data = img_file_buffer.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
elif uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
else:
    st.warning("Please provide an image.")
    st.stop()

# Process image with MediaPipe
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        annotated_image = image.copy()
        user_id = f"user_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"  # Unique user ID

        # Save landmarks to database
        conn = sqlite3.connect("face_landmarks.db")
        c = conn.cursor()

        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                c.execute("INSERT INTO face_landmarks (user_id, timestamp, landmark_index, x, y, z) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?)",
                          (user_id, idx, x, y, z))
        
        conn.commit()
        conn.close()

        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
        )

        # Convert BGR to RGB for Streamlit display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, caption="Detected Face Mesh", use_column_width=True)

        st.success("Facial landmark data saved successfully!")
    else:
        st.warning("No face detected, please try again.")

# Display stored data
def fetch_data():
    conn = sqlite3.connect("face_landmarks.db")
    c = conn.cursor()
    c.execute("SELECT * FROM face_landmarks ORDER BY timestamp DESC LIMIT 10")  # Show latest 10 records
    records = c.fetchall()
    conn.close()
    return records

if st.button("Show Saved Landmark Data"):
    records = fetch_data()
    if records:
        import pandas as pd
        df = pd.DataFrame(records, columns=["ID", "User ID", "Timestamp", "Landmark Index", "X", "Y", "Z"])
        st.dataframe(df)  # Display as a table
    else:
        st.warning("No data found.")
