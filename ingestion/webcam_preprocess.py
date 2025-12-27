import cv2
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face_region(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5
    )

    if len(faces) == 0:
        return image_path  # fallback

    # Take largest face
    x, y, w, h = sorted(
        faces, key=lambda f: f[2]*f[3], reverse=True
    )[0]

    # Expand crop slightly to include glasses
    pad_x = int(0.2 * w)
    pad_y = int(0.2 * h)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)

    cropped = img[y1:y2, x1:x2]

    cv2.imwrite(image_path, cropped)
    return image_path
