"""
Filename: FaceTrainer.py
Author: Jeerabhat Supapinit
"""
import os
import threading
from uuid import uuid4
import cv2
import mediapipe as mp
import numpy as np
import time
from trainer import spilt_chunk, resize_by_height, split_chunks_of
from general import Direction, putBorderText, change_brightness, get_from_percent
from topData import topData
from copy import deepcopy
from FaceAlignment import face_alignment
import os.path as path
from topData import topData
import pyttsx3


def grid_images(images: list[np.ndarray], width: int, each_image_size=(100, 100)):
    horizontals = []
    images = [cv2.resize(raw_img, each_image_size) for raw_img in images]
    for img_chunk in split_chunks_of(images, width):
        if img_chunk:
            base = np.zeros((each_image_size[1], each_image_size[0] * width, 3), dtype=np.uint8)
            horizon_img = np.concatenate(img_chunk, axis=1)
            base[0: horizon_img.shape[0], 0: horizon_img.shape[1]] = horizon_img
            horizontals.append(base)
    return np.concatenate(horizontals, axis=0)


class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()

    def __say(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

    def say(self, message):
        threading.Thread(self.__say(message)).start()


class VideoFaceTrainer:
    def __init__(self, ID=None, output_path=None, min_detection_score=None, core=None):
        self.output_path: str = "" if output_path is None else output_path
        self.ID: str = uuid4().hex if ID is None else ID
        self.min_detection_score = 0.85 if min_detection_score is None else min_detection_score
        self.core = 8 if core is None else core
        self.H, self.W = None, None

        mp_face_mesh = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        self.__face_direction: Direction = Direction.Undefined
        self.__cap: cv2.VideoCapture = cv2.VideoCapture(0)
        self.__face_detection: mp_face_detection.FaceDetection \
            = mp_face_detection.FaceDetection(min_detection_confidence=0.55, model_selection=0)
        self.__drawing_spec: mp_drawing.DrawingSpec \
            = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 0))
        self.__face_mesh: mp_face_mesh.FaceMesh = mp_face_mesh.FaceMesh()
        self.__speaker = Speaker()
        self.__queue: list[Direction] = [
            Direction((-30, -10, -10), error_rate=(10, 1000, 1000), name="-30x degree"),
            Direction((-20, -10, -10), error_rate=(10, 1000, 1000), name="-20x degree"),
            Direction((-10, -10, -10), error_rate=(10, 1000, 1000), name="-10x degree"),
            Direction((10, 10,10), error_rate=(10, 1000, 1000), name="10x degree"),
            Direction((20, 10, 10), error_rate=(10, 1000, 1000), name="20x degree"),
            Direction((30, 10, 10), error_rate=(10, 1000, 1000), name="30x degree"),
            Direction((-10, -60, -10), error_rate=(1000, 10, 1000), name="-60y degree"),
            Direction((-10, -50, -10), error_rate=(1000, 10, 1000), name="-50y degree"),
            Direction((-10, -40, -10), error_rate=(1000, 10, 1000), name="-40y degree"),
            Direction((-10, -30, -10), error_rate=(1000, 10, 1000), name="-30y degree"),
            Direction((-10, -20, -10), error_rate=(1000, 10, 1000), name="-20y degree"),
            Direction((-10, -10, -10), error_rate=(1000, 10, 1000), name="-10y degree"),
            Direction((10, 0, 10), error_rate=(1000, 200, 1000), name="0y degree"),
            Direction((10, 10, 10), error_rate=(1000, 10, 1000), name="10y degree"),
            Direction((10, 20, 10), error_rate=(1000, 10, 1000), name="20y degree"),
            Direction((10, 30, 10), error_rate=(1000, 10, 1000), name="30y degree"),
            Direction((10, 40, 10), error_rate=(1000, 10, 1000), name="40y degree"),
            Direction((10, 50, 10), error_rate=(1000, 10, 1000), name="50y degree"),
            Direction((10, 60, 10), error_rate=(1000, 10, 1000), name="60y degree")
        ]
        self.__queue_index = 0
        self.__to_check_direction: Direction = self.__queue[self.__queue_index]
        capacity = 20
        self.__to_be_encode: dict[Direction, topData] = {
            Direction((-30, -10, -10), error_rate=(10, 1000, 1000), name="-30x degree"): topData(max_size=capacity),
            Direction((-20, -10, -10), error_rate=(10, 1000, 1000), name="-20x degree"): topData(max_size=capacity),
            Direction((-10, -10, -10), error_rate=(10, 1000, 1000), name="-10x degree"): topData(max_size=capacity),
            Direction((10, 10, 10), error_rate=(10, 1000, 1000), name="10x degree"): topData(max_size=capacity),
            Direction((20, 10, 10), error_rate=(10, 1000, 1000), name="20x degree"): topData(max_size=capacity),
            Direction((30, 10, 10), error_rate=(10, 1000, 1000), name="30x degree"): topData(max_size=capacity),
            Direction((-10, -60, -10), error_rate=(1000, 10, 1000), name="-60y degree"): topData(max_size=capacity),
            Direction((-10, -50, -10), error_rate=(1000, 10, 1000), name="-50y degree"): topData(max_size=capacity),
            Direction((-10, -40, -10), error_rate=(1000, 10, 1000), name="-40y degree"): topData(max_size=capacity),
            Direction((-10, -30, -10), error_rate=(1000, 10, 1000), name="-30y degree"): topData(max_size=capacity),
            Direction((-10, -20, -10), error_rate=(1000, 10, 1000), name="-20y degree"): topData(max_size=capacity),
            Direction((-10, -10, -10), error_rate=(1000, 10, 1000), name="-10y degree"): topData(max_size=capacity),
            Direction((10, 0, 10), error_rate=(1000, 200, 1000), name="0xy degree"): topData(max_size=capacity),
            Direction((10, 10, 10), error_rate=(1000, 10, 1000), name="10y degree"): topData(max_size=capacity),
            Direction((10, 20, 10), error_rate=(1000, 10, 1000), name="20y degree"): topData(max_size=capacity),
            Direction((10, 30, 10), error_rate=(1000, 10, 1000), name="30y degree"): topData(max_size=capacity),
            Direction((10, 40, 10), error_rate=(1000, 10, 1000), name="40y degree"): topData(max_size=capacity),
            Direction((10, 50, 10), error_rate=(1000, 10, 1000), name="50y degree"): topData(max_size=capacity),
            Direction((10, 60, 10), error_rate=(1000, 10, 1000), name="60y degree"): topData(max_size=capacity)
        }

    def run(self):

        while self.__cap.isOpened():
            success, image = self.__cap.read()
            if not success:
                continue
            if self.H is None or self.W is None:
                (self.H, self.W) = image.shape[:2]

            start = time.time()
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results_mesh = self.__face_mesh.process(image)
            results_detection = self.__face_detection.process(image)
            image.flags.writeable = True

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
            dist_matrix: np.ndarray

            if results_detection.detections is not None:
                if results_detection.detections[0] is None:
                    self.__face_not_found(image)
                    continue
                if len(results_detection.detections) > 1:
                    self.__face_not_found(image)
                    continue
            else:
                self.__face_not_found(image)
                continue

            detection = results_detection.detections[0]
            x_min = detection.location_data.relative_bounding_box.xmin * self.W
            y_min = detection.location_data.relative_bounding_box.ymin * self.H
            x_max = x_min + detection.location_data.relative_bounding_box.width * self.W
            y_max = y_min + detection.location_data.relative_bounding_box.height * self.H
            face_width = x_max - x_min
            face_height = y_max - y_min
            box = (x_min, y_min, x_max, y_max)
            now_frame = face_alignment(
                deepcopy(
                    image[
                    int(box[1]) - get_from_percent(face_height, 20): int(box[3]) + get_from_percent(face_height, 20),
                    int(box[0]) - get_from_percent(face_height, 20): int(box[2]) + get_from_percent(face_height, 20),
                    ]
                ),
                detection,
            )
            # print(detection.score[0])

            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w
                    cam_matrix = np.array(
                        [
                            [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1],
                        ]
                    )

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x = angles[0] * 360 * 3.6
                    y = angles[1] * 360 * 3.6
                    z = angles[2] * 360 * 3.6

                    face_direction = Direction((x, y, z))
                    if self.__to_check_direction.name.split(" ")[0][-1] == "x":
                        text = f"Looking {round(face_direction.degree_x, 2)} x"
                    elif self.__to_check_direction.name.split(" ")[0][-1] == "y":
                        text = f"Looking {round(face_direction.degree_y, 2)} y"
                    elif self.__to_check_direction.name.split(" ")[0][-1] == "z":
                        text = f"Looking {round(face_direction.degree_z, 2)} z"

                    print(self.__to_check_direction.maximum_error())
                    if self.__to_check_direction.is_same(face_direction):
                        text = f"Looking {self.__to_check_direction.name}"
                        if not (self.__to_be_encode[self.__to_check_direction].is_full() and
                                self.__to_be_encode[self.__to_check_direction].lowest() >= self.min_detection_score):
                            self.__to_be_encode[self.__to_check_direction].add_image(detection.score[0], now_frame)
                        else:
                            self.__queue_index += 1
                            try:
                                self.__to_check_direction = self.__queue[self.__queue_index]
                                self.__speaker.say(self.__to_check_direction.name)
                            except IndexError:
                                return

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                    # Add the text on the image
                    putBorderText(
                        image,
                        text,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        (0, 0, 0),
                        2,
                        3,
                    )
                    putBorderText(
                        image,
                        f"please look {self.__to_check_direction.name}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        (0, 0, 0),
                        3,
                        4,
                    )
                    cv2.putText(
                        image,
                        "x: " + str(np.round(x, 2)),
                        (500, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        "y: " + str(np.round(y, 2)),
                        (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        "z: " + str(np.round(z, 2)),
                        (500, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                    mp.solutions.drawing_utils.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        landmark_drawing_spec=self.__drawing_spec,
                        connection_drawing_spec=self.__drawing_spec,
                    )

                end = time.time()
                totalTime = end - start
                if totalTime > 0:
                    fps = 1 / totalTime
                else:
                    fps = -1

                cv2.putText(
                    image,
                    f"FPS: {int(fps)} Confidence: {round(detection.score[0], 2)}",
                    (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("win", image)
            cv2.setWindowProperty("win", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.__cap.release()
        cv2.destroyAllWindows()

    def get_result(self):
        return self.__to_be_encode
    def __face_not_found(self, img: np.ndarray):
        putBorderText(
            img,
            "Face not found (T-T) ",
            (int(self.W / 2) - 250, int(self.H / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 145, 30),
            (0, 0, 0),
            3,
            5,
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = change_brightness(img, -10)
        cv2.imshow("win", img)
        if cv2.waitKey(5) & 0xFF == 27:
            quit()


if __name__ == "__main__":
    vs = VideoFaceTrainer()
    vs.run()
    uid = uuid4().hex[:5]
    if not path.exists("angle_result"):
        os.mkdir("angle_result")

    results = vs.get_result()
    for direction in results:
        images = results[direction].get()

        if not path.exists(f"angle_result/{uid}"):
            os.mkdir(f"angle_result/{uid}")

        if not path.exists(f"angle_result/{uid}/{direction.name}"):
            os.mkdir(f"angle_result/{uid}/{direction.name}")

        for ind, image in enumerate(images):
            cv2.imwrite(f"angle_result/{uid}/{direction.name}/{ind}.png",
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



