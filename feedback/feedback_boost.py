"""
feedback_boost.py
-----------------
Lightweight feedback store for the visual search demo.

This module records simple per-image relevance signals coming from the UI
(users marking results as relevant or not relevant). The stored integer counts
are converted to a small bounded boost that can be applied to ranking scores
so the system can respond to user preferences without retraining the model.

Design choices:
- Keep feedback influence small and bounded (`BOOST_WEIGHT`) to avoid
    destabilizing ranking.
- Store feedback as a simple JSON map so signals survive restarts.
"""

import os
import json
from collections import defaultdict

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

# Path to persistent feedback store. JSON maps image_name -> integer score.
FEEDBACK_FILE = "feedback/feedback_store.json"

# Max absolute boost added to an item's final score. The stored integer
# feedback count is multiplied by this weight and clamped to [-BOOST_WEIGHT, BOOST_WEIGHT].
# This ensures feedback can nudge results but not dominate the model's visual score.
BOOST_WEIGHT = 0.05   # safe, bounded influence

# ---------------------------------------------------------
# Load / Save Feedback Store
# ---------------------------------------------------------

def _load_feedback():
    """Load the feedback store from disk.

    Returns a defaultdict(int) mapping image_name -> integer score. If the
    file does not exist an empty defaultdict is returned.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return defaultdict(int)

    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)

    # Wrap the plain dict in a defaultdict(int) for safe incrementing
    return defaultdict(int, data)


def _save_feedback(feedback_data):
    """Persist feedback mapping to disk as pretty JSON.

    Note: feedback_data should be a mapping (e.g., defaultdict or dict).
    """
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
    # Load current feedback counts, update the counter, and persist.
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
    # Read current feedback counts and convert to a small bounded boost.
    feedback = _load_feedback()

    if image_name not in feedback:
        return 0.0

    raw_score = feedback[image_name]

    # Convert raw integer to a bounded float boost
    boost = raw_score * BOOST_WEIGHT
    return max(-BOOST_WEIGHT, min(BOOST_WEIGHT, boost))
