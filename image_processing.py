```python
# image_processing.py

import cv2
import numpy as np
from config import IMAGE_SIZE

def process_image(image_path):
    """
    Function to process a single image
    :param image_path: str, path to the image
    :return: np.array, processed image
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Resize the image
    img = cv2.resize(img, IMAGE_SIZE)

    # Normalize the image
    img = img / 255.0

    # Return the processed image
    return img

def process_images(image_paths):
    """
    Function to process a list of images
    :param image_paths: list of str, paths to the images
    :return: np.array, processed images
    """
    # Initialize an empty list to store the processed images
    processed_images = []

    # Process each image
    for image_path in image_paths:
        processed_image = process_image(image_path)
        processed_images.append(processed_image)

    # Convert the list of processed images to a numpy array
    processed_images = np.array(processed_images)

    # Return the processed images
    return processed_images
```
