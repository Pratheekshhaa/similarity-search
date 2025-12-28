import cv2

def smart_crop_face(image_path, margin_ratio=0.03):
    """
    Ultra-light crop.
    Removes only outer border noise.
    Safe for already-cropped product images.
    """

    img = cv2.imread(image_path)
    if img is None:
        return

    h, w, _ = img.shape

    # Minimal crop: 3% from each side
    dx = int(w * margin_ratio)
    dy = int(h * margin_ratio)

    cropped = img[
        dy : h - dy,
        dx : w - dx
    ]

    cv2.imwrite(image_path, cropped)
