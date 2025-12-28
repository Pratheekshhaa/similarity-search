"""
ingestion.webcam_preprocess
--------------------------
Utilities for light-weight face cropping of webcam/uploaded images.
This module locates the largest face in an image using an OpenCV Haar
cascade, expands the bounding box slightly to include glasses, and
overwrites the input image with the cropped face region.

Behavior notes:
- The function is intentionally simple and conservative — if no face is
  detected or the image cannot be read, the original `image_path` is
  returned unchanged as a fallback.
"""

import cv2
import numpy as np

# Preload Haar cascade for frontal face detection. Using OpenCV's
# packaged cascade ensures the runtime does not require external model
# files to be shipped separately.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face_region(image_path: str) -> str:
    """Crop the largest detected face region and overwrite the image.

    Args:
        image_path: Path to an image file on disk.

    Returns:
        The input `image_path`. If processing failed or no face was
        detected the file is left unchanged and the same path is returned.
    """

    # Read image from disk (OpenCV returns None on failure)
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    # Convert to grayscale for cascade detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detectMultiScale returns a list of (x, y, w, h) boxes
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5
    )

    # If no faces found, return original path as a safe fallback
    if len(faces) == 0:
        return image_path

    # Choose the largest detected face (area heuristic) — this helps when
    # multiple faces or background detections are present; largest usually
    # corresponds to the primary subject in webcam captures.
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # Add a small padding to ensure glasses are included in the crop.
    pad_x = int(0.2 * w)
    pad_y = int(0.2 * h)

    # Clamp coordinates to image bounds
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    # Slice and overwrite the original image file with the cropped region
    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(image_path, cropped)
    return image_path
