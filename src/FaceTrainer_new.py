"""
Filename: FaceTrainer.py
Author: Jeerabhat Supapinit
"""
from uuid import uuid4
from typing import Union, Any
import cv2
import mediapipe as mp
import numpy as np
import time
import ray
from copy import deepcopy
import pickle
from src.FaceAlignment import face_alignment
import face_recognition
from threading import Thread
from src.ShadowRemoval import remove_shadow_grey
from src.trainer import spilt_chunk, resize_by_height, split_chunks_of
from src.general import Direction, putBorderText, change_brightness, get_from_percent
from src.topData import topData


def grid_images(images: list[np.ndarray], width: int, each_image_size=(100, 100)):
    horizontals = []
    images = [cv2.resize(raw_img, each_image_size) for raw_img in images]
    for img_chunk in split_chunks_of(images, width):
        if img_chunk:
            base = np.zeros((each_image_size[1], each_image_size[0] * width, 3), dtype=np.uint8)
            horizon_img = np.concatenate(img_chunk, axis=1)
            base[0 : horizon_img.shape[0], 0 : horizon_img.shape[1]] = horizon_img
            horizontals.append(base)
    return np.concatenate(horizontals, axis=0)


@ray.remote
def process_image(info, num_jitters=20, model="large"):
    print("processing image....")
    face_encodings = []
    for img in info:
        face_location = face_recognition.face_locations(img)
        if face_location:
            face_encoding = face_recognition.face_encodings(img, face_location, model=model, num_jitters=num_jitters)
            if face_encoding:
                # print("succc")
                face_encodings.append(face_encoding[0])
        else:
            pass
            # print("unsuccc")
    return face_encodings


ray.init()


class VideoFaceTrainer:
    def __init__(self, ID=None, output_path=None, min_detection_score=None, core=None):
        self.output_path: str = "" if output_path is None else output_path
        self.ID: str = uuid4().hex if ID is None else ID
        self.min_detection_score = 0.85 if min_detection_score is None else min_detection_score
        self.core = 8 if core is None else core
        self.H, self.W = None, None
        self.num_jitters = 20

        mp_face_mesh = mp.solutions.face_mesh
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        self.__face_direction: Direction = Direction.Undefined
        self.__cap: cv2.VideoCapture = cv2.VideoCapture(0)
        self.__face_detection: mp_face_detection.FaceDetection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.55, model_selection=0
        )
        self.__drawing_spec: mp_drawing.DrawingSpec = mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(255, 255, 0)
        )
        self.__face_mesh: mp_face_mesh.FaceMesh = mp_face_mesh.FaceMesh()
        self.__queue: list[Direction] = [
            Direction((-10, -30, -10), error_rate=(1000, 50, 1000), name="lefty degree"),
            Direction((10, 0, 10), error_rate=(1000, 200, 1000), name="forwardy degree"),
            Direction((10, 30, 10), error_rate=(1000, 50, 1000), name="righty degree")
        ]
        self.__queue_index = 0
        self.__to_check_direction: Direction = self.__queue[self.__queue_index]
        capacity = 20
        self.__to_be_encode: dict[Direction, topData] = {
            Direction((-10, -30, -10), error_rate=(1000, 20, 1000), name="left degree"): topData(max_size=capacity),
            Direction((10, 0, 10), error_rate=(1000, 200, 1000), name="forwardy degree"): topData(max_size=capacity*2),
            Direction((10, 30, 10), error_rate=(1000, 20, 1000), name="righty degree"): topData(max_size=capacity)
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
                        int(box[1])
                        - get_from_percent(face_height, 20) : int(box[3])
                        + get_from_percent(face_height, 20),
                        int(box[0])
                        - get_from_percent(face_height, 20) : int(box[2])
                        + get_from_percent(face_height, 20),
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
                    text = ""
                    if self.__to_check_direction.name.split(" ")[0][-1] == "x":
                        text = f"Looking {round(face_direction.degree_x, 2)} x"
                    elif self.__to_check_direction.name.split(" ")[0][-1] == "y":
                        text = f"Looking {round(face_direction.degree_y, 2)} y"
                    elif self.__to_check_direction.name.split(" ")[0][-1] == "z":
                        text = f"Looking {round(face_direction.degree_z, 2)} z"

                    # print(self.__to_check_direction.maximum_error())
                    if self.__to_check_direction.is_same(face_direction):
                        text = f"Looking {self.__to_check_direction.name}"
                        if not (
                            self.__to_be_encode[self.__to_check_direction].is_full()
                            and self.__to_be_encode[self.__to_check_direction].lowest() >= self.min_detection_score
                        ):
                            self.__to_be_encode[self.__to_check_direction].add_image(detection.score[0], now_frame)
                        else:
                            self.__queue_index += 1
                            try:
                                self.__to_check_direction = self.__queue[self.__queue_index]
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

    def __process_images_and_write_images(self, data, ID):
        max_image_amount = len(data)
        # cv2.imshow("win", cv2.cvtColor(grid_images(data, 12), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        a = [process_image.remote(chunk, self.num_jitters) for chunk in spilt_chunk(data, self.core)]
        result = []
        success_images = 0
        for i in ray.get(a):
            for j in i:
                success_images += 1
                result.append(j)
        print(f"Success {success_images}/{max_image_amount} {round((success_images / max_image_amount) * 100, 2)}%")

        with open(self.output_path + f"/{ID}.pkl", "wb") as f:
            pickle.dump({"id": ID, "data": result}, f)
        print("finished..")

    def write_data_normal_gray(self):
        rgb_data = []
        data_grey = []

        for key in self.__to_be_encode:
            for img in self.__to_be_encode[key].get():
                img = resize_by_height(img, 400)
                rgb_data.append(img)
                data_grey.append(cv2.cvtColor(remove_shadow_grey(img), cv2.COLOR_GRAY2RGB))

        self.__process_images_and_write_images(rgb_data, self.ID)
        self.__process_images_and_write_images(data_grey, self.ID + "_GREY")

    def write_data_normal(self):
        rgb_data = []

        for key in self.__to_be_encode:
            for img in self.__to_be_encode[key].get():
                img = resize_by_height(img, 400)
                rgb_data.append(img)

        self.__process_images_and_write_images(rgb_data, self.ID)


class FileFaceTrainer:
    def __init__(self, ID=None, output_path=None, min_detection_score=None, core=None, num_jitters=20):
        self.output_path: str = "" if output_path is None else output_path
        self.ID: str = uuid4().hex if ID is None else ID
        self.min_detection_score: float = 0.85 if min_detection_score is None else min_detection_score
        self.core: int = 8 if core is None else core
        self.files: list[Union[str, Any]] = []
        self.num_jitters = num_jitters

    def __process_images_and_write_images(self, data, ID):
        max_image_amount = len(data)
        a = [process_image.remote(chunk, self.num_jitters) for chunk in spilt_chunk(data, self.core)]
        result = []
        success_images = 0
        for i in ray.get(a):
            for j in i:
                success_images += 1
                result.append(j)
        print(f"Success {success_images}/{max_image_amount} {round((success_images / max_image_amount) * 100, 2)}%")

        with open(self.output_path + f"/{ID}.pkl", "wb") as f:
            pickle.dump({"id": ID, "data": result}, f)
        print("finished..")

    def train_now_normal(self, files):
        self.__process_images_and_write_images([cv2.imread(file) for file in files], self.ID)

    def train_now_gray(self, files):
        self.__process_images_and_write_images(
            [cv2.cvtColor(remove_shadow_grey(file), cv2.COLOR_GRAY2RGB) for file in files], self.ID
        )

    def add(self, file):
        self.files.append(cv2.imread(file))

    def adds(self, files):
        self.files.extend([cv2.imread(file) for file in files])

    def train_normal(self):
        self.__process_images_and_write_images(self.files, self.ID)

    def train_gray(self):
        self.__process_images_and_write_images(
            [cv2.cvtColor(remove_shadow_grey(file), cv2.COLOR_GRAY2RGB) for file in self.files], self.ID
        )


if __name__ == "__main__":
    vf = VideoFaceTrainer(uuid4().hex, r"C:\general\Science_project\Science_project_cp39\resources_test_2\known")
    Thread(target=lambda: (vf.run(), vf.write_data_normal_gray())).start()

