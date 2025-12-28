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
from api.suitability import analyze_face
from flask import session
from ingestion.smart_crop import smart_crop_face

import json

# Path to the curated metadata file for product images.
# This should map filename -> metadata dict (brand, frame_shape, frame_color, etc.)
METADATA_PATH = "api\products_metadata.json.json"

# Load product metadata at startup so routes can attach attributes quickly.
with open(METADATA_PATH, "r") as f:
    PRODUCT_METADATA = json.load(f)

def normalize(s):
    if not s:
        return ""
    return (
        str(s)
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )
# Helper mapping for coarse color groups used by the UI filters.
# Keys are filter names; values are lists of color strings to match against product metadata.
COLOR_BUCKETS = {
    "black": ["black", "matte black", "polished black"],
    "blue": ["blue", "navy", "royal blue"],
    "clear": ["clear", "translucent"],
    "gold": ["gold", "rose gold", "bronze", "champagne"],
    "grey": ["grey", "gunmetal", "silver", "charcoal"],
    "pink": ["pink", "rose"],
    "silver": ["silver"],
    "tortoise": ["tortoise"],
    "unique": ["burgundy", "wood", "floral", "marble"]
}

# Config
UPLOAD_FOLDER = "api/static/uploads"
PRODUCT_FOLDER = "api/static/products"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# App initialization

app = Flask(__name__)
# NOTE: For production, set a secure `SECRET_KEY` and remove debug mode.
app.secret_key = "dev"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the visual embedding model once at startup to avoid reloading for every request.
print("[INFO] Loading embedding model...")
embedding_model = load_embedding_model(device=DEVICE)
print("[INFO] Model loaded")

# HTTP routes

@app.route("/", methods=["GET"])
def home():
    # Homepage: minimal UI that exposes the visual search and suitability tools.
    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():

    # `/search` handles two flows:
    # - GET: apply filters to previously-stored results held in the session
    # - POST: accept an uploaded image (or webcam capture), compute its embedding,
    #         run nearest-neighbor search, then post-process and return results

    # GET: Filter request — apply stored session results and filter by query params
    if request.method == "GET":
        brand = request.args.get("brand", "").strip()
        material = request.args.get("material", "").strip()
        price_range = request.args.get("price-range", "").strip()
        color = request.args.get("color", "").strip()

        # Retrieve last search results from the user's session so filters can be applied
        results = session.get("last_results")
        query_image = session.get("last_query_image")
        query_shape = session.get("last_query_shape")
        query_color = session.get("last_query_color")
        query_brand = session.get("last_query_brand")
        query_material = session.get("last_query_material")
        query_price = session.get("last_query_price")

        if not results:
            return redirect(url_for("home"))

        # Apply simple metadata filters (brand, material, color, price range)
        filtered = []
        for r in results:
            if brand and normalize(brand) not in normalize(r["brand"]):
                continue
            if material and normalize(material) not in normalize(r["material"]):
                continue
            if color:
                allowed_colors = COLOR_BUCKETS.get(color, [])
                if not any(c in normalize(r["frame_color"]) for c in allowed_colors):
                    continue

            if price_range:
                low, high = map(int, price_range.split("-"))
                if not (low <= int(r["price"]) <= high):
                    continue
            filtered.append(r)

        return render_template(
            "results.html",
            query_image=query_image,
            query_shape=query_shape,
            query_color=query_color,
            query_brand=query_brand,
            query_material=query_material,
            query_price=query_price,
            results=filtered
        )


    # POST: Image search — validate upload and compute embedding
    # Validate upload presence
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("home"))

    # Read image bytes and write to the uploads directory so templates can access it
    image_bytes = file.read()
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    with open(upload_path, "wb") as f:
        f.write(image_bytes)

    # Smart crop: adjust crop margins for webcam captures vs product images
    # Determine whether the input came from the webcam (filename prefix) and
    # apply a slightly larger crop for webcam captures to reduce background noise.
    is_webcam = file.filename.startswith("webcam")

    if is_webcam:
        smart_crop_face(upload_path, margin_ratio=0.08)
    else:
        smart_crop_face(upload_path, margin_ratio=0.03)

    text_query = request.form.get("text_query", "").strip().lower()

    # Lookup metadata for the query image if available (useful when searching
    # with a product image from the catalog). Defaults are safe fallbacks.
    query_img_name = file.filename.lower()
    query_meta = PRODUCT_METADATA.get(query_img_name, {})

    query_shape = query_meta.get("frame_shape", "Unknown")
    query_color = query_meta.get("frame_color", "Unknown")
    query_brand = query_meta.get("brand", "Unknown")
    query_material = query_meta.get("material", "Unknown")
    query_price = query_meta.get("price", "N/A")

    # Preprocess the uploaded image into a tensor and run the embedding model.
    # The model returns a 2048-D vector representing visual features.
    tensor = preprocess_from_bytes(image_bytes, device=DEVICE)
    with torch.no_grad():
        query_embedding = embedding_model(tensor)
        query_embedding = query_embedding.squeeze(0).cpu().numpy()

    # Retrieve visually-similar candidates from the FAISS-backed index.
    results = search_similar(query_embedding, top_k=20)
    # Remove the exact same image from results (if the query was a catalog image)
    results = [r for r in results if r["image"].lower() != query_img_name]


    # Simple text-to-metadata matching function used to give a small boost when
    # the user's typed text mentions attributes that match the product metadata.
    def text_match_score(text, product):
        if not text:
            return 0.0

        score = 0.0
        t = normalize(text)

        # Prioritize frame shape mentions, then color, brand, and material.
        if normalize(product.get("frame_shape")) in t or t in normalize(product.get("frame_shape")):
            score += 0.4
        if normalize(product.get("frame_color")) in t:
            score += 0.3
        if normalize(product.get("brand")) in t:
            score += 0.2
        if normalize(product.get("material")) in t:
            score += 0.1

        return score



    # Post-process results:
    # - attach metadata from the product catalog
    # - compute simple boolean flags for color/shape matches
    # - compute a blended final score for ranking and display
    # Post-process each candidate: enrich with metadata, compute match flags,
    # and compute a blended final score used for ranking and display.
    for r in results:
        img = os.path.basename(r["image"]).lower()
        meta = PRODUCT_METADATA.get(img, {})

        # Attach readable metadata fields to each result so templates can render them
        r["brand"] = meta.get("brand", "Unknown")
        r["material"] = meta.get("material", "Unknown")
        r["price"] = meta.get("price", 0)
        r["frame_shape"] = meta.get("frame_shape", "Unknown")
        r["frame_color"] = meta.get("frame_color", "Unknown")

        # Simple boolean flags to indicate exact metadata matches with the query
        r["color_match"] = (r["frame_color"] == query_color)
        r["shape_match"] = (r["frame_shape"] == query_shape)

        visual_score = r.get("score", 0)
        text_score = text_match_score(text_query, r)

        # Weighted combination: visual is primary, text refines, then small
        # deterministic boosts for exact color/shape matches.
        r["final_score"] = (
            0.65 * visual_score +
            0.25 * text_score +
            (0.05 if r["color_match"] else 0) +
            (0.05 if r["shape_match"] else 0)
        )

        # Convert to a human-friendly percentage for display in the UI
        r["match_percent"] = round(r["final_score"] * 100, 2)

    # Sort candidates by final score (highest first)
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Save session for filters — persist last query, metadata and results
    # Persist last query and results to the session so the GET filter flow can
    # reuse them without re-running the embedding/search step.
    session["last_results"] = results
    session["last_query_image"] = file.filename
    session["last_query_shape"] = query_shape
    session["last_query_color"] = query_color
    session["last_query_brand"] = query_brand
    session["last_query_material"] = query_material
    session["last_query_price"] = query_price

    return render_template(
        "results.html",
        query_image=file.filename,
        query_shape=query_shape,
        query_color=query_color,
        query_brand=query_brand,
        query_material=query_material,
        query_price=query_price,
        results=results
    )

@app.route("/suitability", methods=["POST"])
def suitability():
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("home"))

    # Save uploaded image to disk for processing and template access
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(upload_path)

    # Run the suitability analysis (face-shape detection + simple rule-based
    # recommendations). This is intentionally lightweight and intended as
    # guidance rather than a precise fitting tool.
    analysis = analyze_face(upload_path)

    return render_template(
        "suitability.html",
        query_image=file.filename,
        face_shape=analysis["face_shape"],
        recommended_frames=analysis["recommended_frames"],
        explanation=analysis["explanation"]
    )


          
    # Render results (fallback)

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
    next_page = request.form.get("next")

    # Record simple relevance feedback which can be used to boost results
    # or inform future re-ranking. Feedback storage is handled in
    # `feedback/feedback_boost.py`.
    if image_name and action:
        record_feedback(
            image_name=image_name,
            relevant=(action == "relevant")
        )

    # 🔑 Always return to results page
    return redirect(next_page or url_for("home"))

# Application entrypoint

if __name__ == "__main__":
    app.run(debug=True)
