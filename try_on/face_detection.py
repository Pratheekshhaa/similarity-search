"""
face_detection.py
-----------------
Detects face region in an image using OpenCV Haar Cascades.
Used for optional virtual try-on.
"""

import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(image_path):
    """
    Detect face bounding box.
    Returns (x, y, w, h) or None.
    """
    # Read image from disk. `cv2.imread` returns None on failure which will
    # raise in the following `cvtColor` call; callers should ensure the
    # path points to a valid image file.
    img = cv2.imread(image_path)

    # Convert to grayscale for Haar cascade detection — cascades operate
    # on single-channel images.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detectMultiScale returns a list of (x, y, w, h) bounding boxes.
    # Tuning parameters: `scaleFactor` controls image pyramid scaling and
    # `minNeighbors` adjusts detection strictness. These values are
    # conservative defaults for frontal faces in webcam-style images.
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    # If no faces found, return None as a clear signal to the caller.
    if len(faces) == 0:
        return None

    # Return the first detected face. For more robustness, callers can
    # choose the largest face by area or apply additional heuristics.
    return faces[0]
