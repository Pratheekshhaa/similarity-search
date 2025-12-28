"""
suitability.py
--------------
Lightweight face-shape analysis and simple frame recommendations.

This module performs a conservative, explainable face-shape analysis using
OpenCV Haar cascades (face and eye detectors) and simple geometric ratios.
It intentionally avoids heavyweight dependencies (e.g., MediaPipe) so the
component remains lightweight and easy to reproduce.

Primary function:
- `analyze_face(image_path)` — returns a dict with keys:
        - `face_shape`: detected shape label (Round, Square, Oval, Heart, Unknown)
        - `recommended_frames`: list of suggested frame categories
        - `explanation`: human-readable list explaining the recommendation

Notes and design choices:
- Uses the largest detected face when multiple faces are present.
- Uses eye-center distance as a secondary geometric cue; falls back to an
    approximate default when eyes are not reliably detected.
- Recommendations are rule-based and intentionally conservative — they are
    guidance, not precise fit or comfort estimates.
"""

import cv2
import numpy as np

# ---------------------------------------------------------
# Load cascades
# ---------------------------------------------------------
# Load OpenCV Haar cascades from the installed OpenCV data files. These are
# classical detectors that work well for clear, frontal faces in many images.
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
    """Classify face shape using simple geometric ratios.

    Args:
        face_h: face bounding box height in pixels
        face_w: face bounding box width in pixels
        eye_dist: pixel distance between detected eye centers (approx.)

    Returns:
        One of the shape labels: 'Round', 'Square', 'Oval', 'Heart'.

    The classifier uses two ratios:
    - H/W (height divided by width) to measure elongation
    - Eye/W (eye distance divided by width) to measure eye spacing relative to face
    These thresholds are simple heuristics chosen for clarity and explainability.
    """

    ratio_hw = face_h / face_w
    ratio_eye = eye_dist / face_w

    # Debugging print (can be removed or guarded behind a verbosity flag)
    print("DEBUG ratios → H/W:", round(ratio_hw, 2), "Eye/W:", round(ratio_eye, 2))

    # Heuristics (ordered from most specific to fallback):
    # - Round faces: relatively wide (H/W small) and eyes closer together
    if ratio_hw < 1.05 and ratio_eye < 0.45:
        return "Round"

    # - Square faces: roughly equal H/W but eyes are relatively wide
    if 1.0 <= ratio_hw <= 1.15 and ratio_eye >= 0.45:
        return "Square"

    # - Oval faces: noticeably taller than wide but not extremely so
    if 1.15 < ratio_hw <= 1.35:
        return "Oval"

    # - Heart (fallback): longer faces with narrower jaw impression
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

    # Choose the largest detected face as the primary subject (handles group
    # photos where multiple faces are present). The detector returns (x,y,w,h).
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_gray = gray[y : y + h, x : x + w]

    # -----------------------------------------------------
    # Detect eyes
    # -----------------------------------------------------
    # Detect eyes within the face region. Eye detection can fail for many
    # reasons (glasses, occlusion, pose), so we provide a sensible fallback
    # distance when fewer than two eyes are found.
    eyes = EYE_CASCADE.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(eyes) < 2:
        # Fallback: approximate inter-eye distance as a fraction of face width
        eye_dist = w * 0.4
    else:
        # Use the two largest detections (likely true eyes) and compute center x coords
        eyes = sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
        centers = [(ex + ew // 2, ey + eh // 2) for ex, ey, ew, eh in eyes]
        eye_dist = abs(centers[0][0] - centers[1][0])

    face_shape = classify_face_shape(h, w, eye_dist)

    # Simple, human-readable recommendations and explanations for each shape.
    # These are intentionally high-level — they map face shapes to frame
    # categories rather than brand-level product picks.
    recommendations = {
        "Round": ["Square", "Rectangular"],
        "Square": ["Round", "Oval"],
        "Oval": ["Aviator", "Rectangle"],
        "Heart": ["Thin", "Round"],
    }

    explanations = {
        "Round": [
            "Angular frames balance soft facial curves",
            "Adds definition to round features",
        ],
        "Square": [
            "Rounded frames soften sharp jawlines",
            "Improves facial symmetry",
        ],
        "Oval": [
            "Most frame styles suit oval faces",
            "Maintains natural facial balance",
        ],
        "Heart": [
            "Thin frames reduce forehead emphasis",
            "Round frames balance a narrow chin",
        ],
    }

    # Return a structured analysis consumed by the Flask template
    return {
        "face_shape": face_shape,
        "recommended_frames": recommendations[face_shape],
        "explanation": explanations[face_shape],
    }
