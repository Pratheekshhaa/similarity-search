"""
feedback_boost.py
-----------------
Implements a lightweight feedback loop for visual search.

Idea:
- Track how often a product is marked as "relevant"
- Penalize when marked as "not relevant"
- Use a bounded boost during future searches
- No model retraining required

This simulates online learning from user interaction.
"""

import os
import json
from collections import defaultdict

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

FEEDBACK_FILE = "feedback/feedback_store.json"

# Max absolute boost added to final score
BOOST_WEIGHT = 0.05   # safe, bounded influence

# ---------------------------------------------------------
# Load / Save Feedback Store
# ---------------------------------------------------------

def _load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return defaultdict(int)

    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)

    return defaultdict(int, data)


def _save_feedback(feedback_data):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=2)


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def record_feedback(image_name: str, relevant: bool = True):
    """
    Record user feedback for a product image.

    👍 relevant   -> +1
    👎 irrelevant -> -1
    """
    feedback = _load_feedback()

    if relevant:
        feedback[image_name] += 1
    else:
        feedback[image_name] -= 1

    _save_feedback(feedback)


def get_feedback_score(image_name: str) -> float:
    """
    Return a bounded feedback boost in range [-0.05, +0.05].

    Positive score  -> frequently liked
    Negative score  -> frequently disliked
    """
    feedback = _load_feedback()

    if image_name not in feedback:
        return 0.0

    raw_score = feedback[image_name]

    # Clamp feedback influence
    boost = raw_score * BOOST_WEIGHT
    return max(-BOOST_WEIGHT, min(BOOST_WEIGHT, boost))
