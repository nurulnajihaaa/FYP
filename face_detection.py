import streamlit as st
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
else:
    st.info("Please upload an image to display.")


import numpy as np

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a NumPy array
    image_np = np.array(image)
    st.subheader("Image as NumPy Array")
    st.text(f"Image Shape: {image_np.shape}")
    st.write(image_np)  # Display NumPy array values
else:
    st.info("Please upload an image to display and convert.")
