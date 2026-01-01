# Visual Similarity Search for Eyewear
---------------------------------------
This project is an AI-powered visual similarity search system for eyewear. The goal of the system is to allow users to find visually similar glasses using images instead of text. Traditional text-based search fails for eyewear because users often do not know the exact frame shape names, material types, or how to describe subtle visual styles. This system solves that problem by using deep learning image embeddings and vector similarity search. A user can upload an image of glasses, capture an image using a webcam, or optionally add a short text description to refine results. The system then returns visually similar eyewear along with similarity scores and structured filters.
The core input is always an image. Text is only an optional add-on and never replaces visual search.

## Quick summary
- Core idea: image-first retrieval, i.e the image is the primary signal; text only refines results.
- Main components: offline ingestion (embedding extraction), FAISS index runtime Flask app that preprocesses queries, computes query embeddings retrieves nearest neighbors, and applies metadata-based re-ranking + filters.
- This project demonstrates an end-to-end production-style visual search pipeline, including:
    - Image ingestion and preprocessing
    - Deep learning–based feature extraction
    - Vector similarity search
    - Attribute-aware re-ranking
    - Metadata-based filtering
    - User feedback–based ranking boost
    - Smart cropping and multimodal search

The focus is not on UI polish, but on correct AI logic, system design, and search accuracy, exactly as required in the assignment.

Repository layout (important files)
----------------------------------
- `api/` — Flask app and templates (`api/app.py`, `api/suitability.py`)
- `ingestion/` — preprocessing and embedding extraction (`embed_products.py`, `preprocess.py`, `smart_crop.py`)
- `vector_store/` — embeddings and FAISS index builders (`embeddings.npy`, `faiss_index.bin`, `build_index.py`)
- `search/` — FAISS query wrapper and filters (`visual_search.py`, `filters.py`, `ranking.py`)
- `attributes/` — centroid-based attribute classifier (`attribute_classifier.py`, `shape_centroids.json`)
- `try_on/` — face-related helpers (`suitability_score.py`, `overlay_glasses.py`)
- `data/product_eyewear/images` — where catalog images must live (you will create this)
- `requirements.txt` and `environment.yml` — install dependencies

File Structure 
---------------
```
.
├── api
│   ├── __pycache__/
│   ├── static
│   │   ├── products/              # Product catalog images
│   │   ├── uploads/               # User uploaded / webcam images
│   │   └── style.css              # Global frontend styling
│   ├── templates
│   │   ├── index.html              # Image upload UI
│   │   ├── results.html            # Visual search results page
│   │   └── suitability.html        # Virtual try-on & suitability view
│   ├── app.py                      # Flask backend entry point
│   ├── products_metadata.json      # Product attributes (shape, color, brand, etc.)
│   └── suitability.py              # Suitability routing / logic glue
│
├── attributes
│   ├── __pycache__/
│   ├── __init__.py
│   ├── attribute_classifier.py     # Frame shape classification
│   ├── auto_shape_cluster.py        # Unsupervised shape clustering
│   ├── color_detector.py            # Frame color extraction
│   ├── shape_heuristics.py          # Rule-based shape reasoning
│   └── shape_centroids.json         # Learned shape prototypes
│
├── data/                            # Raw / intermediate datasets
│
├── docs/                            # Documentation & reports
│
├── feedback
│   ├── __pycache__/
│   ├── __init__.py
│   ├── boost_scores.json           
│   ├── feedback_boost.py                
│   ├── feedback_store.json                
│
├── ingestion
│   ├── __pycache__/
│   ├── __init__.py
│   ├── embed_products.py            # Image → embedding generation
│   ├── preprocess.py                # Image normalization pipeline
│   ├── smart_crop.py                # Auto crop for eyewear region
│   └── webcam_preprocess.py         # Live webcam frame preprocessing
│
├── search
│   ├── __pycache__/
│   ├── __init__.py
│   ├── filters.py                   # Metadata-based filtering
│   ├── ranking.py                   # Similarity + attribute re-ranking
│   └── visual_search.py             # FAISS-based visual similarity search
│
├── try_on
│   ├── __init__.py
│   ├── face_detection.py            # Face & landmark detection
│   ├── overlay_glasses.py            # Glasses overlay rendering
│   └── suitability_score.py         # Face–frame compatibility scoring
│
├── vector_store
│   ├── __pycache__/
│   ├── build_index.py               # FAISS index construction
│   ├── embeddings.npy               # Stored image embeddings
│   ├── faiss_index.bin              # FAISS ANN index
│   └── image_names.json             # Index ↔ image mapping
│
├── .gitignore
├── environment.yml                  # Conda environment definition
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```
## Architecture image
<img width="1000" height="800" alt="archi_diagram" src="https://github.com/user-attachments/assets/ae8304f2-1a8c-49fb-ae8a-093f1f3b50b2" />


## Requirements & recommended install 
-----------------------------------------------
Before running anything, install all required Python packages. Make sure you are using Python 3.9 or above.
Run:
```python 
pip install -r requirements.txt
```
## Dataset: where to get it and how to place it
-------------------------------------------
I used an eyewear dataset sourced from Kaggle (https://www.kaggle.com/datasets/egorovlvan/glasses-dataset)  
Steps:

1. Download an eyewear/product images dataset from Kaggle (or use your own).
	 - Unzip the downloaded archive to a local folder (we'll call it `dataset/`).

2. Prepare and standardize images: Use the script `prepare_dataset.py`, run the script to extract the images from various folders and move it to a common `api\static\products` folder

3. Metadata: for better filtering and explainability the app expects curated  metadata for each product filename (brand, frame_shape, frame_color, material, price). I have created a metadata and placed it at  `api\products_metadata.json.json`. The search filters will read that file.

## Build embeddings and FAISS index (required before running)
--------------------------------------------------------
You must create the embeddings and the FAISS index before the Flask app can serve meaningful results.

1. Extract embeddings for all product images (offline):

```
# runs ingestion/embed_products.py which writes:
#  - vector_store/embeddings.npy
#  - vector_store/image_names.json
```
```python 
python -m ingestion.embed_products
```

This loads a pretrained ResNet50 backbone (ImageNet weights), removes the classifier head, and produces a 2048-D embedding per image. If a CUDA-capable GPU is available and PyTorch detects it, embedding extraction will use GPU and be much faster. 

2. Build and save the FAISS index:

```
python -m vector_store.build_index
```
This reads `vector_store/embeddings.npy`, normalizes vectors for cosine similarity, builds a FAISS index, and writes `vector_store/faiss_index.bin`. 

Notes:
- If you re-run `embed_products` and the image order changes, re-run 	`build_index` so the FAISS index matches the embeddings.
- You can inspect `vector_store/image_names.json` to confirm the filename order.

## Run the Flask app
-----------------
After embeddings and the FAISS index exist, start the app:

```
python -m api.app

# The app runs in debug mode by default and serves the homepage at http://127.0.0.1:5000/
```
## How to Run the Project
-----------------------------
1. Clone the Repository
```
  git clone https://github.com/Pratheekshhaa/similarity-search.git
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Prepare Dataset
Run the python script after downloading the dataset
```python
python prepare_dataset
```
4. Ensure filenames match metadata keys - Keep products_metadata.json updated
5. Run the Application
```
python api/app.py
```

6. Open in Browser
```
http://127.0.0.1:5000
```

## Routes of interest:
- `/` — homepage
- `/search` — image/text/webcam search
- `/suitability` — face-shape-based suitability analysis

Uploaded images are saved to `api/static/uploads` and product images live in
`api/static/products` (these are used by templates to render results).

Troubleshooting & tips
----------------------
- FAISS on Windows: pip install of `faiss-cpu` may fail. Use conda:
	`conda install -c conda-forge faiss-cpu`.
- PyTorch: install the CPU or CUDA build that matches your system using the
	official PyTorch selector at https://pytorch.org/get-started/locally/.
- If you see mismatched vectors vs. index errors: re-run `python -m ingestion.embed_products` then `python -m vector_store.build_index`.
- If searches are slow: ensure the FAISS index is loading (see console logs from
	`search/visual_search.py`) and that your embeddings file is the correct shape.
- For production deployment on Windows consider `waitress`; on Linux use
	`gunicorn`.
    
## Future advancements

- **Learned re-ranker:** Train a lightweight re-ranker (MLP or small transformer)
  using collected feedback to improve ordering beyond raw cosine similarity.
- **Multimodal embeddings:** Integrate a joint image+text encoder (for example
  CLIP or a finetuned multimodal model) to better fuse optional text queries
  with image signals.
- **Incremental index updates:** Support incremental add/remove of embeddings
  and background index merges to avoid full index rebuilds when the product
  catalog changes.



