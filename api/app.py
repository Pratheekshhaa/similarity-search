import os
from flask import Flask, request, render_template, redirect, url_for
import torch

from ingestion.preprocess import preprocess_from_bytes
from ingestion.embed_products import load_embedding_model
from search.visual_search import search_similar
from feedback.feedback_boost import record_feedback
from attributes.attribute_classifier import classify_shape
from search.ranking import rerank_results
from attributes.color_detector import detect_frame_color

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

UPLOAD_FOLDER = "api/static/uploads"
PRODUCT_FOLDER = "api/static/products"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# APP INIT
# ---------------------------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------------------------

print("[INFO] Loading embedding model...")
embedding_model = load_embedding_model(device=DEVICE)
print("[INFO] Model loaded")

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "":
        return redirect(url_for("home"))

    # -----------------------------------------------------
    # Read & save uploaded image
    # -----------------------------------------------------

    image_bytes = file.read()
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    with open(upload_path, "wb") as f:
        f.write(image_bytes)

    # -----------------------------------------------------
    # Query image processing
    # -----------------------------------------------------

    tensor = preprocess_from_bytes(image_bytes, device=DEVICE)

    with torch.no_grad():
        query_embedding = embedding_model(tensor)
        query_embedding = query_embedding.squeeze(0).cpu().numpy()

    # -----------------------------------------------------
    # Attribute detection (QUERY IMAGE)
    # -----------------------------------------------------

    shape_info = classify_shape(query_embedding)
    query_shape = shape_info["shape"]

    # ✅ FIX: detect QUERY IMAGE COLOR
    query_color = detect_frame_color(upload_path)

    # -----------------------------------------------------
    # Visual similarity search
    # -----------------------------------------------------

    results = search_similar(query_embedding, top_k=10)

    # Remove self-match
    uploaded_name = file.filename.lower()
    results = [
        r for r in results
        if r["image"].lower() != uploaded_name
    ]

    # Keep top 5 only
    results = results[:5]

    # -----------------------------------------------------
    # Re-ranking
    # -----------------------------------------------------

    results = rerank_results(
        results,
        query_shape=query_shape,
        applied_filters=None
    )

    # -----------------------------------------------------
    # Post-processing results
    # -----------------------------------------------------

    for r in results:
        # Clean filename
        r["image"] = os.path.basename(r["image"]).strip().lower()
        img_path = os.path.join(PRODUCT_FOLDER, r["image"])

        # -------------------------------
        # Product attributes
        # -------------------------------
        r["color"] = detect_frame_color(img_path)

        # Color match
        r["color_match"] = (r["color"] == query_color)

        # Shape match
        if query_shape != "unknown":
            r["shape_match"] = True
        else:
            r["shape_match"] = None

        # -------------------------------
        # Scoring
        # -------------------------------
        visual_score = r.get("final_score", r.get("score", 0))

        color_bonus = 0.15 if r["color_match"] else 0.0
        shape_bonus = 0.10 if r["shape_match"] else 0.0

        r["final_score"] = (
            0.75 * visual_score +
            color_bonus +
            shape_bonus
        )

        r["match_percent"] = round(r["final_score"] * 100, 2)

    # ✅ MUST be outside
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    
        
    # -----------------------------------------------------
    # Render results
    # -----------------------------------------------------

    return render_template(
        "results.html",
        query_image=file.filename,
        query_shape=query_shape,
        query_color=query_color,
        results=results
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    image_name = request.form.get("image")
    action = request.form.get("action")

    if image_name and action:
        record_feedback(
            image_name=image_name,
            relevant=(action == "relevant")
        )

    return redirect(request.referrer)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
