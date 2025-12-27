import os
import json
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

EMB_PATH = "vector_store/embeddings.npy"
NAME_PATH = "vector_store/image_names.json"
IMAGE_DIR = "api/static/products"
OUT_PATH = "attributes/shape_centroids.json"

# ---------------------------------------------------------
# Shape heuristics (automatic)
# ---------------------------------------------------------

def infer_shape_from_geometry(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return None

    circularity = 4 * np.pi * area / (peri * peri)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h > 0 else 0

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
    embeddings = np.load(EMB_PATH).astype("float32")
    names = json.load(open(NAME_PATH))

    # Reduce noise
    pca = PCA(n_components=50)
    reduced = pca.fit_transform(embeddings)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)

    # Collect indices per cluster
    clusters = {}
    for idx, cid in enumerate(labels):
        clusters.setdefault(cid, []).append(idx)

    shape_centroids = {}

    for cid, indices in clusters.items():
        votes = {}

        # sample a few images per cluster
        for i in indices[:30]:
            img_name = names[i]
            img_path = os.path.join(IMAGE_DIR, img_name)

            shape = infer_shape_from_geometry(img_path)
            if shape:
                votes[shape] = votes.get(shape, 0) + 1

        if not votes:
            print(f"[WARN] Cluster {cid}: no shape inferred")
            continue

        final_shape = max(votes, key=votes.get)
        centroid = np.mean(embeddings[indices], axis=0)

        shape_centroids[final_shape] = centroid.tolist()
        print(f"[INFO] Cluster {cid} → {final_shape}")

    os.makedirs("attributes", exist_ok=True)
    json.dump(shape_centroids, open(OUT_PATH, "w"), indent=2)
    print(f"[DONE] Saved → {OUT_PATH}")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    auto_discover_shapes()
