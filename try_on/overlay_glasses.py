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
    face_img = cv2.imread(face_img_path)
    glasses_img = cv2.imread(glasses_img_path, cv2.IMREAD_UNCHANGED)

    x, y, w, h = face_box

    # Resize glasses to face width
    glasses_resized = cv2.resize(glasses_img, (w, int(h * 0.4)))

    y_offset = y + int(h * 0.35)

    for i in range(glasses_resized.shape[0]):
        for j in range(glasses_resized.shape[1]):
            if glasses_resized[i, j][3] > 0:  # alpha channel
                face_img[y_offset + i, x + j] = glasses_resized[i, j][:3]

    return face_img
