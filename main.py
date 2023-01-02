"""
Filename: main.py
Author: Jeerabhat Supapinit
"""

# --------------import---------------------
import Logger
from ShadowRemoval import remove_shadow_grey
from centroidtracker import CentroidTracker
import numpy as np
import time
import general
from general import log, Color, get_from_percent
from copy import deepcopy
import cv2
from FaceAlignment import face_alignment
import ray
import mediapipe as mp

# -----------------setting-----------------
video_source = 0
min_detection_confidence = 0.75
min_faceBlur_detection = 24  # low = blur, high = not blur
autoBrightnessContrast = False
autoBrightnessValue = 80  # from 0 - 255
autoContrastValue = 30  # from 0 - 255
face_check_amount = 5
face_max_disappeared = 20
night_mode_brightness = 40
sharpness_filter = False
debug = True
cpu_amount = 8
face_reg_path = r"C:\general\Science_project\Science_project_cp39\\resources"

# -------------global variable--------------
ray.init(num_cpus=cpu_amount)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
ct = CentroidTracker(
    face_reg_path,
    maxDisappeared=face_max_disappeared,
    minFaceBlur=min_faceBlur_detection,
    minFaceConfidence=min_detection_confidence,
    faceCheckAmount=face_check_amount,
)
logger = Logger.Logger()
(H, W) = (None, None)
text_color = (0, 255, 255)
prev_frame_time = 0  # fps counter
new_frame_time = 0
last_id = -1
already_check = {}
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# -----------------main program--------------
if __name__ == "__main__":
    log(
        f"{video_source=}\n{min_detection_confidence=}\n{min_faceBlur_detection=}"
        f"{face_check_amount=}\n{sharpness_filter=}\n{cpu_amount=}",
        color=Color.Yellow,
    )

    cap = cv2.VideoCapture(video_source)
    face_mesh = mp_face_mesh.FaceMesh()
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
            if general.brightness(image) < night_mode_brightness:
                image = cv2.cvtColor(remove_shadow_grey(image), cv2.COLOR_GRAY2RGB)  # for night vision
                general.putBorderText(
                    image,
                    "NIGHT MODE",
                    (W - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Color.Violet,
                    Color.Black,
                    2,
                    3,
                )
            if sharpness_filter:
                image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            rects = []
            if results.detections:
                for detection in results.detections:
                    x_min = detection.location_data.relative_bounding_box.xmin * W
                    y_min = detection.location_data.relative_bounding_box.ymin * H
                    x_max = x_min + detection.location_data.relative_bounding_box.width * W
                    y_max = y_min + detection.location_data.relative_bounding_box.height * H
                    face_height = y_max - y_min
                    box = (x_min, y_min, x_max, y_max)
                    face_image = face_alignment(
                        deepcopy(
                            image[
                                int(box[1])
                                - get_from_percent(face_height, 20) : int(box[3])
                                + get_from_percent(face_height, 20),
                                int(box[0])
                                - get_from_percent(face_height, 20) : int(box[2])
                                + get_from_percent(face_height, 20),
                            ]
                        ),
                        face_mesh,
                    )
                    #  face_image = deepcopy(image[int(box[1]) - get_from_percent(face_height, 20):int(box[3]) + get_from_percent(face_height, 20), int(box[0]) - get_from_percent(face_height, 20):int(box[2]) + get_from_percent(face_height, 20)])
                    rects.append({box: (detection.score[0], face_image)})

                    if autoBrightnessContrast:
                        face_image = general.change_brightness_to(face_image, autoBrightnessValue)
                        face_image = general.change_contrast_to(face_image, autoContrastValue)
                    # cv2.imshow("test", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

                    general.putBorderText(
                        image,
                        f"confident: {round(detection.score[0], 2)}% blur {CentroidTracker.is_blur(face_image, min_faceBlur_detection)} ",
                        (int(box[0]), int(box[1]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        (0, 0, 0),
                        2,
                        3,
                    )
                    if debug:
                        general.putBorderText(
                            image,
                            f"brightness: {round(general.brightness(face_image), 2)} contrast: {round(general.contrast(face_image), 2)}",
                            (int(box[0]), int(box[1]) + 38),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            (0, 0, 0),
                            2,
                            3,
                        )
                    # f"brightness: {round(general.brightness(face_image), 2)} contrast: {round(general.contrast(face_image), 2)}"
                    cv2.rectangle(
                        image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 0),
                        5,
                    )
                    cv2.rectangle(
                        image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        text_color,
                        3,
                    )

            objects = ct.update(rects)
            for i in [i[0] for i in objects.items()]:
                if i > last_id:
                    last_id = i
                    already_check[i] = False
                    print(f"update id {i-1} -> {i}")
                else:
                    name = ray.get(ct.objects_names.get.remote(objectID))
                    if name not in ["UNKNOWN", "IN_PROCESS", "CHECKED_UNKNOWN"] and not already_check[i]:
                        already_check[i] = True
                        print(f"update id {i-1} -> {i}: '{name}'")

            for (objectID, centroid) in objects.items():
                text = "ID [{}]".format(objectID)
                # noinspection PyUnresolvedReferences
                general.putBorderText(
                    image,
                    text,
                    (centroid[0] - 10, centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    (0, 0, 0),
                    2,
                    3,
                )
                general.putBorderText(
                    image,
                    general.Most_Common(ray.get(ct.objects_names.get.remote(objectID)))
                    if type(ray.get(ct.objects_names.get.remote(objectID))) == list
                    else ray.get(ct.objects_names.get.remote(objectID)),
                    (centroid[0] - 10, centroid[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    (0, 0, 0),
                    2,
                    3,
                )

                cv2.circle(image, (centroid[0], centroid[1]), 4, text_color, -1)

            # Flip the image horizontally for a selfie-view display.
            total_time = new_frame_time - prev_frame_time
            fps = int(1 / total_time) if total_time != 0 else -1
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image,
                str(fps),
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )
            prev_frame_time = new_frame_time

            cv2.imshow("FaceDetection_test", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
