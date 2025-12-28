"""
overlay_glasses.py
------------------
Overlays a glasses image on detected face region.
"""

import cv2

def overlay_glasses(face_img_path, glasses_img_path, face_box):
    """
    Overlays glasses on face image.

    face_box: (x, y, w, h)
    """
    # Read images: face image (BGR) and glasses (with alpha channel)
    face_img = cv2.imread(face_img_path)
    glasses_img = cv2.imread(glasses_img_path, cv2.IMREAD_UNCHANGED)

    x, y, w, h = face_box

    # Resize glasses to approximately the face width. Height is set
    # proportionally (here 40% of face height) to keep aspect ratio.
    glasses_resized = cv2.resize(glasses_img, (w, int(h * 0.4)))

    # Vertical offset places the glasses roughly over the eyes area
    y_offset = y + int(h * 0.35)

    # Iterate pixels and copy where the glasses alpha channel is opaque.
    # Note: this nested loop is simple and clear; for larger images a
    # vectorized approach using masks would be faster but is unnecessary
    # for small overlays and keeps behavior deterministic.
    for i in range(glasses_resized.shape[0]):
        for j in range(glasses_resized.shape[1]):
            # Check alpha channel (index 3) to decide whether to copy
            if glasses_resized[i, j][3] > 0:
                face_img[y_offset + i, x + j] = glasses_resized[i, j][:3]

    # Return the combined image as a BGR ndarray. The caller can write
    # this to disk or further process it for display in the UI.
    return face_img
