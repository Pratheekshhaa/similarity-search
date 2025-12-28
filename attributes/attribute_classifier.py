"""
attribute_classifier.py
-----------------------
Lightweight centroid-based classifier for eyewear attributes.

Why centroid classifier?
- No training required
- Uses the same 2048-D embeddings from ResNet50
- Manually labeled examples per class
- Computes class centroids
- Classifies new images by nearest centroid (cosine similarity)

Attributes supported:
- frame_shape (aviator, round, square, wayfarer, cat_eye, rimless)
"""

import os
import json
import numpy as np
from numpy.linalg import norm

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

CENTROID_FILE = os.path.join(
    os.path.dirname(__file__),
    "shape_centroids.json"
)

# ---------------------------------------------------------
# Load Centroids (cached)
# ---------------------------------------------------------

_centroids = None

def _load_centroids():
    global _centroids

    if _centroids is None:
        if not os.path.exists(CENTROID_FILE):
            raise FileNotFoundError(
                f"Centroid file not found at {CENTROID_FILE}. "
                "Run build_shape_centroids() first."
            )

        with open(CENTROID_FILE, "r") as f:
            data = json.load(f)

        # Convert lists → normalized numpy arrays
        _centroids = {
            k: np.array(v, dtype="float32") / norm(v)
            for k, v in data.items()
        }

        print(f"[INFO] Loaded {len(_centroids)} shape centroids")

    return _centroids


# ---------------------------------------------------------
# Build Centroids (run once manually)
# ---------------------------------------------------------

def build_shape_centroids(labeled_examples: dict):
    """
    Build centroids from manually labeled examples.

    Args:
        labeled_examples: dict
            {
                "aviator": ["prod_0001.jpg", "prod_0014.jpg"],
                "round":   ["prod_0020.jpg", "prod_0033.jpg"]
            }

    Requires:
        - vector_store/embeddings.npy
        - vector_store/image_names.json

    Saves:
        attributes/shape_centroids.json
    """

    EMB_PATH = "vector_store/embeddings.npy"
    NAME_PATH = "vector_store/image_names.json"

    if not os.path.exists(EMB_PATH) or not os.path.exists(NAME_PATH):
        raise FileNotFoundError("Run embed_products.py first.")

    embeddings = np.load(EMB_PATH).astype("float32")
    with open(NAME_PATH, "r") as f:
        image_names = json.load(f)

    name_to_vec = {name: embeddings[i] for i, name in enumerate(image_names)}

    centroids = {}

    for shape, img_list in labeled_examples.items():
        vecs = []

        for img in img_list:
            if img not in name_to_vec:
                print(f"[WARN] {img} not found in embeddings, skipping")
                continue
            vecs.append(name_to_vec[img])

        if not vecs:
            print(f"[WARN] No valid examples for {shape}, skipping")
            continue

        centroid = np.mean(np.vstack(vecs), axis=0)
        centroid = centroid / norm(centroid)   # normalize
        centroids[shape] = centroid.tolist()

    os.makedirs(os.path.dirname(CENTROID_FILE), exist_ok=True)
    with open(CENTROID_FILE, "w") as f:
        json.dump(centroids, f, indent=2)

    print(f"[DONE] Saved centroids → {CENTROID_FILE}")


# ---------------------------------------------------------
# Classify New Image
# ---------------------------------------------------------

def classify_shape(query_embedding: np.ndarray):
    centroids = _load_centroids()

    if not centroids:
        return {"shape": "unknown", "distance": None}

    best_shape = None
    best_dist = float("inf")

    q = query_embedding.astype("float32")
    q = q / (np.linalg.norm(q) + 1e-8)   # 🔑 normalize query


    for shape, centroid in centroids.items():
        c = centroid.astype("float32")

        # cosine distance
        dist = 1 - np.dot(q, c)
        if dist < best_dist:
            best_dist = dist
            best_shape = shape

    return {
        "shape": best_shape if best_shape else "unknown",
        "distance": round(float(best_dist), 4) if best_shape else None
    }


# ---------------------------------------------------------
# Script Test
# ---------------------------------------------------------

if __name__ == "__main__":
    dummy = np.random.rand(2048).astype("float32")
    print(classify_shape(dummy))
