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

METADATA_PATH = "api\products_metadata.json.json"

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
app.secret_key = "dev"   # add once globally
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

@app.route("/search", methods=["GET", "POST"])
def search():

    # =====================================================
    # 1️⃣ FILTER REQUEST (GET)
    # =====================================================
    if request.method == "GET":
        brand = request.args.get("brand", "").strip()
        material = request.args.get("material", "").strip()
        price_range = request.args.get("price-range", "").strip()
        color = request.args.get("color", "").strip()

        results = session.get("last_results")
        query_image = session.get("last_query_image")
        query_shape = session.get("last_query_shape")
        query_color = session.get("last_query_color")
        query_brand = session.get("last_query_brand")
        query_material = session.get("last_query_material")
        query_price = session.get("last_query_price")

        if not results:
            return redirect(url_for("home"))

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


    # =====================================================
    # 2️⃣ IMAGE SEARCH (POST)
    # =====================================================
    # Validate upload
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("home"))

    image_bytes = file.read()
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    with open(upload_path, "wb") as f:
        f.write(image_bytes)

    # ---------------------------------------------
    # SMART CROP (MINIMAL – SAFE FOR PRODUCT IMAGES)
    # ---------------------------------------------
    is_webcam = file.filename.startswith("webcam")

    if is_webcam:
        smart_crop_face(upload_path, margin_ratio=0.08)  # webcam
    else:
        smart_crop_face(upload_path, margin_ratio=0.03)  # product images

    text_query = request.form.get("text_query", "").strip().lower()

    query_img_name = file.filename.lower()
    query_meta = PRODUCT_METADATA.get(query_img_name, {})

    query_shape = query_meta.get("frame_shape", "Unknown")
    query_color = query_meta.get("frame_color", "Unknown")
    query_brand = query_meta.get("brand", "Unknown")
    query_material = query_meta.get("material", "Unknown")
    query_price = query_meta.get("price", "N/A")

    tensor = preprocess_from_bytes(image_bytes, device=DEVICE)
    with torch.no_grad():
        query_embedding = embedding_model(tensor)
        query_embedding = query_embedding.squeeze(0).cpu().numpy()

    results = search_similar(query_embedding, top_k=20)
    results = [r for r in results if r["image"].lower() != query_img_name]
    def text_match_score(text, product):
        if not text:
            return 0.0

        score = 0.0
        t = normalize(text)

        if normalize(product.get("frame_shape")) in t or t in normalize(product.get("frame_shape")):
            score += 0.4
        if normalize(product.get("frame_color")) in t:
            score += 0.3
        if normalize(product.get("brand")) in t:
            score += 0.2
        if normalize(product.get("material")) in t:
            score += 0.1

        return score



    # =====================================================
    # 3️⃣ POST-PROCESS RESULTS
    # =====================================================
    for r in results:
        img = os.path.basename(r["image"]).lower()
        meta = PRODUCT_METADATA.get(img, {})

        r["brand"] = meta.get("brand", "Unknown")
        r["material"] = meta.get("material", "Unknown")
        r["price"] = meta.get("price", 0)
        r["frame_shape"] = meta.get("frame_shape", "Unknown")
        r["frame_color"] = meta.get("frame_color", "Unknown")

        r["color_match"] = (r["frame_color"] == query_color)
        r["shape_match"] = (r["frame_shape"] == query_shape)

        visual_score = r.get("score", 0)
        text_score = text_match_score(text_query, r)

        r["final_score"] = (
            0.65 * visual_score +
            0.25 * text_score +
            (0.05 if r["color_match"] else 0) +
            (0.05 if r["shape_match"] else 0)
        )


        r["match_percent"] = round(r["final_score"] * 100, 2)

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # =====================================================
    # 4️⃣ SAVE SESSION FOR FILTERS
    # =====================================================
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

    # Save image
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(upload_path)

    # Phase 1: stub analysis
    analysis = analyze_face(upload_path)

    return render_template(
        "suitability.html",
        query_image=file.filename,
        face_shape=analysis["face_shape"],
        recommended_frames=analysis["recommended_frames"],
        explanation=analysis["explanation"]
    )


          
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
    next_page = request.form.get("next")

    if image_name and action:
        record_feedback(
            image_name=image_name,
            relevant=(action == "relevant")
        )

    # 🔑 Always return to results page
    return redirect(next_page or url_for("home"))

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
