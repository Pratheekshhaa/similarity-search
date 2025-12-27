"""
filters.py
----------
Applies metadata-based filters to visual search results.

Filtering is performed AFTER FAISS similarity search,
using structured product metadata.

Supported filters:
- brand
- color
- material
- price range (min_price, max_price)
"""

import os
import csv

# ---------------------------------------------------------
# CONFIG (MATCHES YOUR STRUCTURE)
# ---------------------------------------------------------

METADATA_PATH = "data/product_eyewear/metadata.csv"

# ---------------------------------------------------------
# Load metadata once (cached)
# ---------------------------------------------------------

_metadata = None

def _load_metadata():
    """
    Loads metadata.csv into a dictionary:

    {
        "prod_00001.jpg": {
            "brand": "rayban",
            "color": "black",
            "material": "metal",
            "price": 1999.0
        },
        ...
    }
    """
    global _metadata

    if _metadata is not None:
        return _metadata

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}"
        )

    metadata = {}

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image = row.get("image")
            if not image:
                continue

            metadata[image] = {
                "brand": row.get("brand", "").strip().lower(),
                "color": row.get("color", "").strip().lower(),
                "material": row.get("material", "").strip().lower(),
                "price": float(row.get("price", 0))
            }

    _metadata = metadata
    print(f"[INFO] Loaded metadata for {len(metadata)} products")

    return _metadata


# ---------------------------------------------------------
# Core Filter Function
# ---------------------------------------------------------

def apply_filters(
    results,
    brand=None,
    color=None,
    material=None,
    min_price=None,
    max_price=None
):
    """
    Apply structured filters to FAISS search results.

    Args:
        results: list of dicts from visual_search.py
                 [{"image": "prod_00001.jpg", "score": 0.92}, ...]

        brand, color, material: optional strings (case-insensitive)
        min_price, max_price: optional numeric values

    Returns:
        Filtered list of results
    """

    metadata = _load_metadata()
    filtered_results = []

    for r in results:
        image = r.get("image")

        if image not in metadata:
            # Skip results without metadata
            continue

        meta = metadata[image]

        # Brand filter (partial match allowed)
        if brand and brand.lower() not in meta["brand"]:
            continue

        # Color filter
        if color and color.lower() not in meta["color"]:
            continue

        # Material filter
        if material and material.lower() not in meta["material"]:
            continue

        # Price range filter
        price = meta["price"]
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue

        filtered_results.append(r)

    return filtered_results
