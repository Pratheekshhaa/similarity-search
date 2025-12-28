"""
suitability.py
--------------
Reliable face-shape detection using:
- Haar Face
- Haar Eyes
- Geometry ratios (NO MediaPipe)
"""

import cv2
import numpy as np

# ---------------------------------------------------------
# Load cascades
# ---------------------------------------------------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ---------------------------------------------------------
# Face shape classifier (REAL)
# ---------------------------------------------------------
def classify_face_shape(face_h, face_w, eye_dist):
    ratio_hw = face_h / face_w
    ratio_eye = eye_dist / face_w

    print("DEBUG ratios → H/W:", round(ratio_hw, 2),
          "Eye/W:", round(ratio_eye, 2))

    # Round: wide face, close eyes
    if ratio_hw < 1.05 and ratio_eye < 0.45:
        return "Round"

    # Square: similar H/W, wider eyes
    if 1.0 <= ratio_hw <= 1.15 and ratio_eye >= 0.45:
        return "Square"

    # Oval: longer face
    if 1.15 < ratio_hw <= 1.35:
        return "Oval"

    # Heart: long face, narrow jaw impression
    return "Heart"


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------
def analyze_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image path")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120)
    )

    if len(faces) == 0:
        return {
            "face_shape": "Unknown",
            "recommended_frames": [],
            "explanation": ["No face detected"]
        }

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_gray = gray[y:y+h, x:x+w]

    # -----------------------------------------------------
    # Detect eyes
    # -----------------------------------------------------
    eyes = EYE_CASCADE.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(eyes) < 2:
        # fallback (still works)
        eye_dist = w * 0.4
    else:
        # Take two largest eyes
        eyes = sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
        centers = [(ex + ew//2, ey + eh//2) for ex, ey, ew, eh in eyes]
        eye_dist = abs(centers[0][0] - centers[1][0])

    face_shape = classify_face_shape(h, w, eye_dist)

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
        "recommended_frames": recommendations[face_shape],
        "explanation": explanations[face_shape]
    }
