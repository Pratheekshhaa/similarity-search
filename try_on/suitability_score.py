"""
suitability_score.py
--------------------
Computes a simple suitability score between face and eyewear.
"""

def compute_suitability(face_shape, frame_shape):
    """
    Simple rule-based suitability scoring.
    """

    rules = {
        "round": ["rectangle", "square"],
        "square": ["round", "oval"],
        "oval": ["aviator", "wayfarer"],
    }

    if face_shape in rules and frame_shape in rules[face_shape]:
        return 0.9

    return 0.6
