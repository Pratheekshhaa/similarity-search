"""
suitability_score.py
--------------------
Computes a simple suitability score between face and eyewear.
"""

def compute_suitability(face_shape, frame_shape):
    """
    Simple rule-based suitability scoring.
    """
    # Lightweight, explainable rules mapping face shapes -> preferred frame shapes
    rules = {
        "round": ["rectangle", "square"],
        "square": ["round", "oval"],
        "oval": ["aviator", "wayfarer"],
    }

    # If the pair matches a preferred mapping, return a high suitability
    # score (0.9). Otherwise, return a neutral score (0.6). These values
    # are intentionally coarse — they are intended for demonstration and
    # can be replaced by a learned model or more granular heuristics later.
    if face_shape in rules and frame_shape in rules[face_shape]:
        return 0.9

    return 0.6
