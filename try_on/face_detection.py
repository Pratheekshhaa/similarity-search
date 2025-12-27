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
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        return None

    return faces[0]  # first detected face
