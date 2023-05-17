import glob
import logging
import os.path as path
import pickle
from typing import *
from tabulate import tabulate
from src.DataBase import DataBase
import cv2
import face_recognition
import imutils
import numpy as np
from src import general
from src.general import log, Color, get_from_percent
from json import loads, decoder
from os.path import exists
import mediapipe as mp
from tabulate import tabulate
from src.ShadowRemoval import remove_shadow_grey
import google


def get_all_filename(data_path) -> list:
    if path.isdir(data_path):
        files: list[str] = []
        for file in glob.glob(data_path + "\\*"):
            files.extend(get_all_filename(file))
        return files
    else:
        return [data_path]


class Recognition:
    def __init__(self, data_path, remember=False, name_map_path=None):
        self.loaded_encodings: list[np.ndarray] = []
        self.name_map: dict[str, str] = {}
        self.name_map_path: str = name_map_path
        self.face_detection_method: str = "hog"
        self.loaded_id: list[str] = []
        self.min_confidence: float = 0.5
        self.num_jitters = 2
        self.remember: bool = remember
        self.unknown: str = "UNKNOWN??"
        self.data_path = data_path

        if self.name_map_path is not None:
            if not exists(self.name_map_path):
                open(self.name_map_path, "w").close()
            with open(self.name_map_path, "r") as file:
                try:
                    exists_name = file.read()
                    if exists_name:
                        self.name_map = loads(exists_name)
                except decoder.JSONDecodeError:
                    print("json error decoding")

        if path.isdir(self.data_path):
            for filename in get_all_filename(self.data_path):
                if path.splitext(filename)[1].lower() in [".pkl", ".pickle"]:
                    with open(filename, "rb") as file:
                        data = pickle.load(file)
                        if data.get("id") is not None and data.get("data") is not None:
                            self.loaded_encodings.extend(data["data"])
                            self.loaded_id.extend([data["id"] for _ in range(len(data["data"]))])
                        else:
                            print(f'cannot read file named "{path.basename(filename)}"')
                            continue
        else:
            print("only accepts directory.")

        log("\n" +
            tabulate(
                list(
                    zip(
                        sorted(list(set(self.loaded_id)), key=len),
                        [self.loaded_id.count(i) for i in sorted(list(set(self.loaded_id)), key=len)],
                    )
                ),
                headers=["ID", "amount"],
                tablefmt="rounded_grid",
            ),
            color=Color.Cyan,
            )

    def update(self, storage: DataBase.Storage):
        encodings: List[google.cloud.storage.blob.Blob] = list(storage.bucket.list_blobs(prefix="encoded/"))
        print(f"found new \"{len(encodings)}\" in firebase storage.")
        print(tabulate([[j + 1, i.name, i.size] for j, i in enumerate(encodings)], headers=("index", "name", "size"),
                       tablefmt="rounded_grid"))
        for encoding in encodings:
            if encoding.name.lstrip('encoded/'):
                encoding.download_to_filename(
                    path.join(self.data_path, "known", encoding.name.lstrip("encoded/")) + ".pkl"
                )
                print(f"downloaded encoding \"{encoding.name.lstrip('encoded/')}\" successful!")
                print(f"trying to delete {encoding.name} from the storage")
                storage.delete_encoding(encoding.name.lstrip('encoded/'))
                print(f"delete encoding \"{encoding.name.lstrip('encoded/')}\" successful!")

        if self.name_map_path is not None:
            if not path.exists(self.name_map_path):
                open(self.name_map_path, "w").close()
            with open(self.name_map_path, "r") as file:
                try:
                    exists_name = file.read()
                    if exists_name:
                        self.name_map = loads(exists_name)
                except decoder.JSONDecodeError:
                    print("json error decoding")

        if path.isdir(self.data_path):
            for filename in general.scan_files(self.data_path):
                if path.splitext(filename)[1].lower() in [".pkl", ".pickle"]:
                    with open(filename, "rb") as file:
                        data = pickle.load(file)
                        if data.get("id") is not None and data.get("data") is not None:
                            self.loaded_encodings.extend(data["data"])
                            self.loaded_id.extend([data["id"] for _ in range(len(data["data"]))])
                        else:
                            print(f'cannot read file named "{path.basename(filename)}"')
                            continue
        else:
            print("only accepts directory.")

        log("\n" +
            tabulate(
                list(
                    zip(
                        sorted(list(set(self.loaded_id)), key=len),
                        [self.loaded_id.count(i) for i in
                         sorted(list(set(self.loaded_id)), key=len)],
                    )
                ),
                headers=["ID", "amount"],
                tablefmt="rounded_grid",
            ),
            color=Color.Green
            )

    def recognition(self, img):
        # return face and confidence of the face
        if not self.loaded_encodings and not self.remember:
            print("false because i dont know")
            return (False, 0), None

        # print(np.shape(self.loaded_encodings), self.loaded_id, len(self.loaded_id))

        mp_face_detection = mp.solutions.face_detection
        W, H, _ = img.shape
        img = cv2.cvtColor(remove_shadow_grey(img), cv2.COLOR_GRAY2RGB)

        if not (W and H):
            print("false because w or h is 0")
            return (False, 0), None

        img = imutils.resize(img, width=230)
        W, H, _ = img.shape

        if self.face_detection_method == "mp":
            face_location = []  # noting
            with mp_face_detection.FaceDetection(model_selection=0) as detection:
                results = detection.process(img)
                if results.detections:
                    detection = results.detections[0]
                    x_min = detection.location_data.relative_bounding_box.xmin * W
                    y_min = detection.location_data.relative_bounding_box.ymin * H
                    x_max = x_min + detection.location_data.relative_bounding_box.width * W
                    y_max = y_min + detection.location_data.relative_bounding_box.height * H
                    face_height = y_max - y_min
                    face_width = x_max - x_min
                    box = (
                        int(x_min) + get_from_percent(face_width, 10),
                        int(y_min) + get_from_percent(face_height, 10),
                        int(x_max) - get_from_percent(face_width, 10),
                        int(y_max) - get_from_percent(face_height, 10),
                    )

                    face_location = [box]

        elif self.face_detection_method == "knn":
            face_location = face_recognition.face_locations(img, model="knn")
        else:
            face_location = face_recognition.face_locations(img)  # base model = hog

        try:
            new_encoding = face_recognition.face_encodings(img, face_location, num_jitters=self.num_jitters,
                                                           model="large")[0]
            # new_encoding = face_recognition.face_encodings(img, num_jitters=self.num_jitters)[0]

        except IndexError:
            print("false because index error")
            return (False, 0), None

        if not self.loaded_encodings:
            return (self.unknown, 0.3), new_encoding
        matches = face_recognition.compare_faces(self.loaded_encodings, new_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(self.loaded_encodings, new_encoding)

        avg: dict[str, general.Average] = {}
        for idx, face in enumerate(self.loaded_id):
            if avg.get(face) is None:
                avg[face] = general.Average()
            try:
                avg[face].add(face_distances[idx])
            except IndexError:
                pass
                # print(len(self.loaded_id), len(self.loaded_encodings))

        match_names = set()
        for j, i in enumerate(matches):
            if i:
                match_names.add(self.loaded_id[j])
        match_names = list(match_names)
        match_names_value = [avg[i].get() for i in match_names]

        name = self.unknown
        best_match_value = 0.7

        if match_names_value:
            best_match_index = np.argmin(match_names_value)
            best_match_value = match_names_value[best_match_index]
            if 1 - best_match_value > self.min_confidence:
                # name = f"{self.loaded_id[best_match_index]} [{round(1-face_distances[best_match_index],2)}]"
                name = f"{match_names[best_match_index]}"

            print(f"recognized: {name}")

            if name == self.unknown and self.remember:
                return (name, 1 - best_match_value), new_encoding

        return (name, 1 - best_match_value), new_encoding
