import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title
st.title("Display and Save Image with Streamlit")

# Step 1: File Uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Step 2: Perform any OpenCV processing (example: add a rectangle)
    height, width, _ = image_bgr.shape
    cv2.rectangle(image_bgr, (50, 50), (width - 50, height - 50), (0, 255, 0), 3)

    # Step 3: Display the processed image using st.image
    st.subheader("Processed Image")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Convert back to RGB for Streamlit
    st.image(image_rgb, caption="Processed Image", use_column_width=True)

    # Step 4: Save the processed image locally
    save_path = "output_image.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # Save as BGR format

    # Step 5: Provide a download link for the user
    with open(save_path, "rb") as file:
        st.download_button(
            label="Download Processed Image",
            data=file,
            file_name="output_image.jpg",
            mime="image/jpeg",
        )
        st.success("Image saved and ready for download!")
