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

    # Fail fast if metadata file isn't present. The calling code expects
    # structured metadata to be available for filtering and display.
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}"
        )

    metadata = {}

    # Read CSV with a header that includes at minimum: image, brand,
    # color, material, price. Missing or malformed rows are skipped.
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image = row.get("image")
            if not image:
                # Skip empty or header-only rows
                continue

            # Normalize textual fields to lower-case for case-insensitive
            # comparisons in the filtering stage.
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

    # Load metadata once (cached) and apply filters in a single pass.
    metadata = _load_metadata()
    filtered_results = []

    for r in results:
        image = r.get("image")

        # Skip items with no metadata (can't be filtered reliably)
        if image not in metadata:
            continue

        meta = metadata[image]

        # Brand filter (supports partial, case-insensitive matching)
        if brand and brand.lower() not in meta["brand"]:
            continue

        # Color filter (case-insensitive substring match)
        if color and color.lower() not in meta["color"]:
            continue

        # Material filter (case-insensitive substring match)
        if material and material.lower() not in meta["material"]:
            continue

        # Price range filter (numeric comparators)
        price = meta["price"]
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue

        # Item passed all filters — keep it
        filtered_results.append(r)

    return filtered_results
