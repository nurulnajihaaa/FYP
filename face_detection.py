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
        # Convert the image to a NumPy array
    image_np = np.array(image)
else:
    st.info("Please upload an image to display.")
    st.write("Image successfully uploaded and processed as a NumPy array.")

#else:
   # st.info("Please upload an image to display and convert.")
