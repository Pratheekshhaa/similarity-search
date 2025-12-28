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

    # If no attribute provided, no bonus
    if not query_shape:
        return 0.0

    image_name = result.get("image", "").lower()

    # Lightweight heuristic: if query shape appears in the filename,
    # give a small deterministic bonus. This is intentionally simple
    # and conservative to avoid introducing noisy scoring signals.
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

    # If any filters were applied and the item survived them, provide a
    # small deterministic bonus to favor results that match user intent.
    return 1.0


# ---------------------------------------------------------
# Final Re-ranking Function
# ---------------------------------------------------------

def rerank_results(results, query_shape=None, applied_filters=None):
    """
    Combine similarity + optional bonuses.
    Feedback is handled at app-level.
    """

    final_ranked = []

    for r in results:
        # Primary similarity score from FAISS (expected to be in [0,1])
        sim_score = r.get("score", 0)

        # Small, explainable bonuses computed independently of the ANN
        # similarity and applied in a weighted sum to produce `final_score`.
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

    # Sort descending by the computed final score
    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return final_ranked
