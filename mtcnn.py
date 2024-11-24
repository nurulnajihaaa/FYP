!pip install mtcnn

from google.colab import files
import io
from PIL import Image
import numpy as np

def upload_and_process_image():
  """
  Allows user to upload an image from their local device,
  opens it using PIL, and returns the image as a NumPy array.
  """

  uploaded = files.upload()
  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    image_data = uploaded[fn]

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Convert the image to a NumPy array
    image_np = np.array(image)
    return image_np


# Example usage:
try:
  image_array = upload_and_process_image()
  print("Image successfully uploaded and processed as a NumPy array.")
  # Further processing with image_array can be done here
except Exception as e:
  print(f"Error during image upload and processing: {e}")

from mtcnn import MTCNN
import cv2
from google.colab.patches import cv2_imshow

try:
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    for face in faces:
        x, y, width, height = face['box']
        keypoints = face['keypoints']

        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Annotate facial landmarks
        for keypoint_name, (x_coord, y_coord) in keypoints.items():
            cv2.circle(image, (x_coord, y_coord), 2, (0, 0, 255), 2)
            cv2.putText(image, keypoint_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2_imshow(image)

except Exception as e:
    print(f"Error processing the image with MTCNN: {e}")
