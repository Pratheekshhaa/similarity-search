"""
auto_shape_cluster.py
---------------------
Automatic discovery of dominant frame shapes from product embeddings.

This utility performs a lightweight pipeline to:
- load precomputed image embeddings (from `vector_store/embeddings.npy`)
- reduce dimensionality via PCA
- cluster embeddings with KMeans
- inspect a few images per cluster using simple geometry heuristics
- emit a mapping of shape -> centroid vector to `attributes/shape_centroids.json`

The output can be used as an initial set of centroids for the centroid-based
attribute classifier (`attributes/attribute_classifier.py`). This script is a
convenience tool to bootstrap class centroids when labeled examples are scarce.
"""

import os
import json
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

# Paths to required artifacts. Ensure `ingestion.embed_products` and
# `vector_store.build_index` have been run so embeddings and names exist.
EMB_PATH = "vector_store/embeddings.npy"
NAME_PATH = "vector_store/image_names.json"

# Where product images are stored (used for simple geometric inference)
IMAGE_DIR = "api/static/products"

# Output: inferred shape centroids written here
OUT_PATH = "attributes/shape_centroids.json"

# ---------------------------------------------------------
# Shape heuristics (automatic)
# ---------------------------------------------------------

def infer_shape_from_geometry(img_path):
    """Infer a coarse frame shape from a product image using simple geometry.

    This function is intentionally heuristic and noisy — it is used only to
    vote within clusters when building centroids. It returns one of the shape
    labels ('round', 'aviator', 'cat_eye', 'square') or None when inference
    fails.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Edge-based contour extraction and heuristics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Use the largest external contour as a proxy for the frame silhouette
    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None

    # Circularity: closer to 1 means more circular
    circularity = 4 * np.pi * area / (peri * peri)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h > 0 else 0

    # Heuristic thresholds (tunable)
    if circularity > 0.65:
        return "round"
    if aspect > 1.35:
        return "aviator"
    if aspect < 0.75:
        return "cat_eye"
    return "square"

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def auto_discover_shapes(n_clusters=4):
    # Load embeddings and corresponding image names
    embeddings = np.load(EMB_PATH).astype("float32")
    names = json.load(open(NAME_PATH))

    # Dimensionality reduction to remove noise and speed up clustering
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(embeddings)

    # KMeans clustering in the reduced space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)

    # Group indices by cluster id
    clusters = {}
    for idx, cid in enumerate(labels):
        clusters.setdefault(cid, []).append(idx)

    shape_centroids = {}

    # For each cluster, sample some images and vote on inferred shapes
    for cid, indices in clusters.items():
        votes = {}

        # Inspect up to 30 images per cluster to build a voting distribution
        for i in indices[:30]:
            img_name = names[i]
            img_path = os.path.join(IMAGE_DIR, img_name)

            shape = infer_shape_from_geometry(img_path)
            if shape:
                votes[shape] = votes.get(shape, 0) + 1

        if not votes:
            print(f"[WARN] Cluster {cid}: no shape inferred")
            continue

        # Choose the most-voted shape for this cluster
        final_shape = max(votes, key=votes.get)

        # Compute centroid in original embedding space for the cluster indices
        centroid = np.mean(embeddings[indices], axis=0)

        shape_centroids[final_shape] = centroid.tolist()
        print(f"[INFO] Cluster {cid} → {final_shape}")

    # Persist discovered centroids
    os.makedirs("attributes", exist_ok=True)
    json.dump(shape_centroids, open(OUT_PATH, "w"), indent=2)
    print(f"[DONE] Saved → {OUT_PATH}")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    auto_discover_shapes()
