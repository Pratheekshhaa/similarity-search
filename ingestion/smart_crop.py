"""
ingestion.smart_crop
--------------------
Small, conservative image cropping utility used to trim border noise
from product images or light-weight webcam captures. The function in this
module modifies images in-place and deliberately performs only a slight
crop so downstream visual embeddings remain stable.

The implementation is non-destructive in the sense that it returns False
on any IO/read/write failures and clamps aggressive margin values to avoid
producing empty images.
"""

import os
from typing import Union
import cv2


def smart_crop_face(image_path: Union[str, os.PathLike], margin_ratio: float = 0.03) -> bool:

    # Normalize to str path and verify file presence before processing.
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        return False

    # Clamp margin_ratio to a safe range to avoid empty/invalid crops.
    if margin_ratio <= 0:
        margin_ratio = 0.0
    elif margin_ratio >= 0.5:
        margin_ratio = 0.49

    # Read image using OpenCV; returns None on failure.
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # Convert fraction to pixel offsets. Integer cast intentionally
    # floors the value which keeps the crop conservative.
    dx = int(w * margin_ratio)
    dy = int(h * margin_ratio)

    # Guard against degenerate crops (zero or negative dimensions).
    if w - 2 * dx <= 0 or h - 2 * dy <= 0:
        return False

    # Perform the crop and overwrite the file. Use a try/except around the
    # write operation to catch unexpected filesystem errors.
    cropped = img[dy : h - dy, dx : w - dx]
    try:
        ok = cv2.imwrite(image_path, cropped)
        return bool(ok)
    except Exception:
        return False
