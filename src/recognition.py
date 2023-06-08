from __future__ import annotations

import collections
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
from uuid import uuid4


def get_all_filename(data_path) -> list:
    if path.isdir(data_path):
        files: list[str] = []
        for file in glob.glob(data_path + "\\*"):
            files.extend(get_all_filename(file))
        return files
    else:
        return [data_path]


class Recognition:
    unknown = "UNKNOWN??"

    class ProcessedFace:
        def __init__(self, filename, auto_save=False, create_file_if_not_found=False, **kwargs):
            self.filename = filename
            self.IDD: str = ""
            self.data: List[np.ndarray] = []
            self.VER = 1
            self.__auto_save = auto_save

            if create_file_if_not_found:
                self.IDD = ("unknown:" if kwargs.get("unknown") is not None else "") + \
                           (kwargs.get("IDD") if kwargs.get("IDD") is not None else str(uuid4().hex))
                self.save()

            self.open()

        def open(self):
            try:
                with open(self.filename, "rb") as file:
                    rawdata = pickle.load(file)
                    if rawdata.get("id") is not None and rawdata.get("data") is not None:
                        self.data = rawdata.get("data")
                        self.IDD = rawdata.get("id")
                        self.VER = rawdata.get("version")
                    else:
                        raise KeyError(f'cannot read file "{self.filename}" because \"id\" or \"data\" is not found.')
            except FileNotFoundError:
                raise FileNotFoundError(f'file named "{self.filename}" not found!')

        def save(self):
            with open(self.filename, "wb") as file:
                pickle.dump(self.to_dict(), file)

        def add(self, processed_face: Recognition.ProcessedFace):
            self.data.extend(processed_face.data)
            if self.__auto_save:
                self.save()

        def add_raw_encoding(self, raw_encoding: np.ndarray):
            self.data.append(raw_encoding)
            if self.__auto_save:
                self.save()

        def add_raw_encodings(self, raw_encodings: Union[List[np.ndarray], Tuple[np.ndarray]]):
            self.data.extend(raw_encodings)
            if self.__auto_save:
                self.save()

        @property
        def amount(self) -> int:
            return len(self.data)

        @property
        def is_unknown(self):
            return self.IDD.startswith("unknown:")

        def to_dict(self) -> Dict:
            return {"id": self.IDD, "data": self.data}

        def to_file(self, file_path):
            with open(file_path, "wb") as file:
                pickle.dump(self.to_dict(), file)

    class ProcessedFacePool:
        @staticmethod
        def from_filenames(filenames: List[str]) -> Recognition.ProcessedFacePool:
            return Recognition.ProcessedFacePool([Recognition.ProcessedFace(filename) for filename in filenames])

        def __init__(self, processed_faces: Union[List[Recognition.ProcessedFace], Tuple[Recognition.ProcessedFace]]):
            self.processed_faces: List[Recognition.ProcessedFace] = processed_faces

        def get_identities(self) -> List[str]:
            return list(set(face.IDD for face in self.processed_faces))

        def get_encodings(self) -> List[List[np.ndarray]]:
            return [i.data for i in self.processed_faces]

        def add(self, process_pool: Recognition.ProcessedFacePool):
            self.processed_faces.extend(process_pool.processed_faces)

        def add_processed_face(self, process_face: Recognition.ProcessedFace):
            if process_face.IDD in self.get_identities():
                self.processed_faces[self.index(process_face.IDD)].add_raw_encodings(process_face.data)
            else:
                self.processed_faces.append(process_face)

        def get_encoding(self, IDD):
            return self.processed_faces[self.index(IDD)]

        def index(self, IDD):
            for i, pf in enumerate(self.processed_faces):
                if pf.IDD == IDD:
                    return i

        def face_recognition(self, ref: List[np.ndarray], tolerance=0.4) -> Tuple[str, float]:
            avg: Dict[str, general.Average] = {}
            for identity in self.processed_faces:
                for new_encoding in ref:
                    matches = face_recognition.compare_faces(identity.data, new_encoding, tolerance=tolerance)

                    for idx, match in enumerate(matches):
                        if match:
                            face_distance = face_recognition.face_distance([identity.data[idx]], new_encoding)
                            if avg.get(identity.IDD) is None:
                                avg[identity.IDD] = general.Average()
                            avg[identity.IDD].add(face_distance)
            sorted_identities = sorted(avg.items(), key=lambda x: x[1].get())

            for i in sorted_identities:
                print(i[0], ": ", i[1].get())

            try:
                return sorted_identities[0][0], sorted_identities[0][1].get()
            except IndexError:
                return Recognition.unknown, 0

        def generator(self) -> collections.Iterable[Recognition.ProcessedFace]:
            for i in self.processed_faces:
                yield i

    def __init__(self, data_path, remember=False, name_map_path=None):
        self.processed_faces: Recognition.ProcessedFacePool
        self.name_map: dict[str, str] = {}
        self.name_map_path: str = name_map_path
        self.face_detection_method: str = "hog"
        self.min_confidence: float = 0.5
        self.num_jitters = 2
        self.remember: bool = remember
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
            self.processed_faces: Recognition.ProcessedFacePool = Recognition.ProcessedFacePool.from_filenames(
                general.scan_files(self.data_path, ".pkl"))
        else:
            raise TypeError("only accepts directory.")

        log("\n" +
            tabulate(
                list(
                    zip(
                        sorted(self.processed_faces.get_identities(), key=len),
                        [self.processed_faces.get_identities().count(i) for i in
                         sorted(list(set(self.processed_faces.get_identities())), key=len)],
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

        process_faces: Recognition.ProcessedFacePool = Recognition.ProcessedFacePool.from_filenames(
            general.scan_files(self.data_path, ".pkl"))
        self.processed_faces.add(process_faces)

        log("\n" +
            tabulate(
                list(
                    zip(
                        sorted(list(set(self.processed_faces.get_identities())), key=len),
                        [self.processed_faces.get_identities().count(i) for i in
                         sorted(list(set(self.processed_faces.get_identities())), key=len)],
                    )
                ),
                headers=["ID", "amount"],
                tablefmt="rounded_grid",
            ),
            color=Color.Green
            )

    def recognition(self, img):
        # return face and confidence of the face
        if not self.processed_faces.get_encodings() and not self.remember:
            print("false because i dont know")
            return (False, 0), None

        mp_face_detection = mp.solutions.face_detection
        W, H, _ = img.shape
        img = cv2.cvtColor(remove_shadow_grey(img), cv2.COLOR_GRAY2RGB)

        if not (W and H):
            print("false because w or h is 0")
            return (False, 0), None

        img = imutils.resize(img, width=230)
        W, H, _ = img.shape

        if self.face_detection_method == "mp":
            face_location = []
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

        if not self.processed_faces.get_encodings():
            return (self.unknown, 0.3), new_encoding

        name = self.unknown
        best_match_value = 0.7

        name, best_match_value = self.processed_faces.face_recognition([new_encoding], tolerance=1-self.min_confidence)

        print(f"recognized: {name}")
        if name == self.unknown and self.remember:
            return (name, 1 - best_match_value), new_encoding

        return (name, 1 - best_match_value), new_encoding
