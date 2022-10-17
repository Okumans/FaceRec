"""
Filename: main.py
Author: Jeerabhat Supapinit
"""

# --------------import---------------------
import Logger
from centroidtracker import CentroidTracker
import numpy as np
import time
import general
from general import log, Color, get_from_percent
from copy import deepcopy
import cv2
import ray
import mediapipe as mp

# -----------------setting-----------------
video_source = 0
min_detection_confidence = .75
min_faceBlur_detection = 24  # low = blur, high = not blur
face_check_amount = 5
face_max_disappeared = 20
sharpness_filter = False
cpu_amount = 8

# -------------global variable--------------
ray.init(num_cpus=cpu_amount, logging_level="ERROR")
mp_face_detection = mp.solutions.face_detection
ct = CentroidTracker(maxDisappeared=face_max_disappeared, minFaceBlur=min_faceBlur_detection,
                     minFaceConfidence=min_detection_confidence, faceCheckAmount=face_check_amount)
logger = Logger.Logger()
(H, W) = (None, None)
text_color = (0, 255, 255)
prev_frame_time = 0  # fps counter
new_frame_time = 0
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# -----------------main program--------------
if __name__ == "__main__":
    log(f"{video_source=}\n{min_detection_confidence=}\n{min_faceBlur_detection=}"
        f"{face_check_amount=}\n{sharpness_filter=}\n{cpu_amount=}", color=Color.Yellow)

    cap = cv2.VideoCapture(video_source)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.55, model_selection=1) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            new_frame_time = time.time()

            if not success:
                logger.log("Ignoring empty camera frame.")
                continue

            if H is None or W is None:
                (H, W) = image.shape[:2]

            image.flags.writeable = False
            image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
            if sharpness_filter: image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            rects = []
            if results.detections:
                for detection in results.detections:
                    x_min = detection.location_data.relative_bounding_box.xmin
                    y_min = detection.location_data.relative_bounding_box.ymin
                    x_max = x_min + detection.location_data.relative_bounding_box.width
                    y_max = y_min + detection.location_data.relative_bounding_box.height
                    face_height = y_max - y_min
                    box = (x_min * W, y_min * H, x_max * W, y_max * H)
                    face_image = deepcopy(image[int(box[1])-get_from_percent(face_height, 20):int(box[3]), int(box[0]):int(box[2])])
                    rects.append({box: (detection.score[0], face_image)})
                    general.putBorderText(image,
                                          f"confident: {str(round(detection.score[0], 2))}% blur {CentroidTracker.is_blur(face_image, 24)}",
                                          (int(box[0]), int(box[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                          (0, 0, 0), 2, 3)
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 5)
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), text_color, 3)

            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():
                text = "ID [{}]".format(objectID)
                # noinspection PyUnresolvedReferences
                general.putBorderText(image, text, (centroid[0] - 10, centroid[1] - 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, (0, 0, 0), 2, 3)
                general.putBorderText(image,
                                      general.Most_Common(ray.get(ct.objects_names.get.remote(objectID))) if type(
                                          ray.get(ct.objects_names.get.remote(objectID))) == list else ray.get(
                                          ct.objects_names.get.remote(objectID)), (centroid[0] - 10, centroid[1] - 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, (0, 0, 0), 2, 3)

                cv2.circle(image, (centroid[0], centroid[1]), 4, text_color, -1)

            # Flip the image horizontally for a selfie-view display.
            fps = int(1 / (new_frame_time - prev_frame_time))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(image, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
            prev_frame_time = new_frame_time

            cv2.imshow('FaceDetection_test', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
