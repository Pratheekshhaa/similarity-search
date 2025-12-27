import cv2
import numpy as np

def infer_shape_from_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    if circularity > 0.75:
        return "round"
    elif aspect > 1.3:
        return "aviator"
    elif aspect < 0.85:
        return "cat_eye"
    else:
        return "square"
