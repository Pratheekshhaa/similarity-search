#the dataset contains images in nested folders like dataset/0/product/img1.jpg, dataset/1/product/img2.jpg etc.
#this script will copy all images from these nested folders into a single folder api/static/products


import os
import shutil
from pathlib import Path

# -------- PATHS--------
SOURCE_ROOT = "dataset"  # folder containing 0/, 1/, 2/ ... 1398/
DEST_DIR = "api/static/products"
# -----------------------------------

os.makedirs(DEST_DIR, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png", ".webp"}
count = 0

for product_id in sorted(os.listdir(SOURCE_ROOT)):
    product_path = os.path.join(SOURCE_ROOT, product_id, "product")

    if not os.path.isdir(product_path):
        continue

    for img in os.listdir(product_path):
        ext = os.path.splitext(img)[1].lower()
        if ext not in img_exts:
            continue

        src_img = os.path.join(product_path, img)
        dst_img = os.path.join(DEST_DIR, f"prod_{count:05d}{ext}")

        try:
            shutil.copy2(src_img, dst_img)
            count += 1
        except Exception as e:
            print(f"Skipped {src_img}: {e}")

print(f"✅ Done. Copied {count} product images.")
