"""
ranking.py
----------
Combines multiple scoring signals into a final ranking score.

Signals used:
- similarity score (FAISS)
- feedback boost
- attribute match (optional, query-level)
- filter match (optional)

Ranking is transparent and explainable.
"""

from feedback.feedback_boost import apply_feedback_boost

# ---------------------------------------------------------
# WEIGHTS (tuneable)
# ---------------------------------------------------------

W_SIMILARITY = 0.85     # dominant signal
W_ATTRIBUTE  = 0.10     # small bonus
W_FILTER     = 0.05     # smallest influence


# ---------------------------------------------------------
# Attribute Match Scoring (Query-level)
# ---------------------------------------------------------

def compute_attribute_score(result, query_shape=None):
    """
    Gives a bonus if product name heuristically matches query shape.
    (Optional, safe fallback)

    Args:
        result: {"image": ..., "score": ...}
        query_shape: string (e.g., "aviator")

    Returns:
        float
    """

    if not query_shape:
        return 0.0

    image_name = result.get("image", "").lower()

    # very lightweight heuristic (safe)
    if query_shape.lower() in image_name:
        return 1.0

    return 0.0


# ---------------------------------------------------------
# Filter Match Scoring (Optional)
# ---------------------------------------------------------

def compute_filter_score(result, applied_filters=None):
    """
    Small bonus if filters were applied.
    (Does not assume metadata attachment)
    """

    if not applied_filters:
        return 0.0

    # If item survived filtering, reward slightly
    return 1.0


# ---------------------------------------------------------
# Final Re-ranking Function
# ---------------------------------------------------------

def rerank_results(results, query_shape=None, applied_filters=None):
    """
    Combine similarity + feedback + optional bonuses.

    Args:
        results: list from visual_search.py
        query_shape: output from attribute_classifier
        applied_filters: dict of filters applied

    Returns:
        Re-ranked list of results
    """

    # Step 1: feedback learning
    boosted = apply_feedback_boost(results)

    final_ranked = []

    for r in boosted:
        sim_score = r["score"]

        attr_score = compute_attribute_score(r, query_shape)
        filter_score = compute_filter_score(r, applied_filters)

        final_score = (
            W_SIMILARITY * sim_score +
            W_ATTRIBUTE  * attr_score +
            W_FILTER     * filter_score
        )

        final_ranked.append({
            **r,
            "final_score": round(final_score, 4)
        })

    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return final_ranked
