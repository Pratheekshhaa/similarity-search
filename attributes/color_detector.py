import cv2
import numpy as np

COLOR_NAMES = {
    "black": ([0, 0, 0], [50, 50, 50]),
    "white": ([200, 200, 200], [255, 255, 255]),
    "brown": ([60, 40, 20], [150, 100, 60]),
    "gold": ([150, 120, 60], [255, 220, 150]),
    "silver": ([180, 180, 180], [220, 220, 220]),
    "blue": ([0, 0, 100], [100, 100, 255]),
    "red": ([100, 0, 0], [255, 80, 80]),
}

def detect_frame_color(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "unknown"

    h, w, _ = img.shape
    center = img[h//4:3*h//4, w//4:3*w//4]

    avg_color = np.mean(center.reshape(-1, 3), axis=0)

    for name, (low, high) in COLOR_NAMES.items():
        if np.all(avg_color >= low) and np.all(avg_color <= high):
            return name

    return "other"
