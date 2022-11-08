"""
Filename: FaceAlignment.py
Author: Jeerabhat Supapinit
"""

import cv2 as cv
import numpy as np
from PIL import Image

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord


def face_alignment(img, face_mesh):
    if img.any():
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(img, results, False)
            right_eye_position = list(map(int, np.array([mesh_coords[p] for p in RIGHT_EYE]).mean(axis=0)))
            left_eye_position = list(map(int, np.array([mesh_coords[p] for p in LEFT_EYE]).mean(axis=0)))
            dY = right_eye_position[1] - left_eye_position[1]
            dX = right_eye_position[0] - left_eye_position[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            img = np.array(Image.fromarray(img).rotate(angle))

    return img
