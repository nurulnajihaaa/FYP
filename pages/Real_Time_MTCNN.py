import streamlit as st
import cv2
import numpy as np

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

from mtcnn import MTCNN
import cv2
#from google.colab.patches import cv2_imshow

try:
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    detector = MTCNN()
    st.write("Detected succesfully.")
    faces = detector.detect_faces(image)
    st.write("Detected face succesfully.")

    for face in faces:
        x, y, width, height = face['box']
        keypoints = face['keypoints']

    # Draw bounding box around the face
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Annotate facial landmarks
    for keypoint_name, (x_coord, y_coord) in keypoints.items():
        cv2.circle(image, (x_coord, y_coord), 2, (0, 0, 255), 2)
        cv2.putText(image, keypoint_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert BGR back to RGB for Streamlit display
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the annotated image in Streamlit
    st.subheader("Annotated Image")
    st.image(annotated_image, caption="Detected Faces and Landmarks", use_column_width=True)
    st.success("Image successfully annotated.")
    #st.image(image)
    #st.write("Image successfully annotated.")

except Exception as e:
    print(f"Error processing the image with MTCNN: {e}")
