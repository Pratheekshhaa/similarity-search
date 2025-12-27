"""
suitability.py
--------------
Phase 2 (Reworked): Face shape detection using OpenCV geometry
(NO MediaPipe, NO ML dependency)
"""

import cv2
import numpy as np

# Load Haar Cascade (comes with OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------
# Face shape classifier
# ---------------------------------------------------------

def classify_face_shape(face_w, face_h):
    """
    Classify face shape using bounding-box geometry
    """

    ratio = face_h / face_w

    if ratio < 1.1:
        return "Round"
    elif 1.1 <= ratio <= 1.25:
        return "Square"
    elif 1.25 < ratio <= 1.4:
        return "Oval"
    else:
        return "Heart"


# ---------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------

def analyze_face(image_path):
    """
    Analyze face and return face shape + recommendations
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return {
            "face_shape": "Unknown",
            "recommended_frames": [],
            "explanation": ["No face detected"]
        }

    # Take largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face_shape = classify_face_shape(w, h)

    recommendations = {
        "Round": ["Square", "Rectangular"],
        "Square": ["Round", "Oval"],
        "Oval": ["Aviator", "Rectangle"],
        "Heart": ["Thin", "Round"]
    }

    explanations = {
        "Round": [
            "Angular frames balance soft facial curves",
            "Adds definition to round features"
        ],
        "Square": [
            "Rounded frames soften sharp jawlines",
            "Improves facial symmetry"
        ],
        "Oval": [
            "Most frame styles suit oval faces",
            "Maintains natural facial balance"
        ],
        "Heart": [
            "Thin frames reduce forehead emphasis",
            "Round frames balance a narrow chin"
        ]
    }

    return {
        "face_shape": face_shape,
        "recommended_frames": recommendations.get(face_shape, []),
        "explanation": explanations.get(face_shape, [])
    }
