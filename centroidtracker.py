import time

from scipy.spatial import distance as dist
from collections import OrderedDict
import topData
import numpy as np
import general
from general import log
from copy import deepcopy
import cv2
import ray
from recognition import Recognition
from uuid import uuid4
from json import loads, dumps
from pickle import load as pkl_load
from pickle import dump as pkl_dump
from os.path import exists


class CentroidTracker:
    def __init__(
            self,
            faceRecPath,
            maxDisappeared=10,
            minFaceConfidence=0.85,
            minFaceRecConfidence=0.5,
            minFaceBlur=100,
            faceCheckAmount=10,
            remember_unknown_face=False,
            otherSetting=None,
    ):
        otherSetting = {} if otherSetting is None else otherSetting
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objects_positions = topData.topData(10)
        self.objects_ID = set()
        self.objects_data = OrderedDict()
        self.disappeared = OrderedDict()
        self.faceRecPath = faceRecPath
        self.recognizer = Recognition(faceRecPath, name_map_path=otherSetting.get("name_map_path"))
        self.objects_names = general.rayDict.remote()
        self.is_checked = general.rayDict.remote()
        self.maxDisappeared = maxDisappeared
        self.minFaceConfidence = minFaceConfidence
        self.minFaceRecConfidence = minFaceRecConfidence
        self.minFaceBlur = minFaceBlur
        self.faceCheckAmount = faceCheckAmount
        self.remember_unknown_face = remember_unknown_face
        self.remember_unknown_face_maximum_confidence = 0.54
        self.last_deregister = general.rayDict.remote()
        self.recognition_progress = general.rayDict.remote()
        self.pre_face_encodings = general.rayDict.remote()
        self.recognizer.min_confidence = self.minFaceRecConfidence
        self.recognizer.remember = self.remember_unknown_face

    def register(self, centroid, data):
        self.objects[self.nextObjectID] = centroid
        self.objects_positions.add(self.nextObjectID, centroid)
        self.objects_ID.add(self.nextObjectID)
        self.objects_data[self.nextObjectID] = topData.topData(
            min_detection=self.minFaceBlur, max_size=self.faceCheckAmount
        )
        self.objects_names.set.remote(self.nextObjectID, "UNKNOWN")
        self.is_checked.set.remote(self.nextObjectID, False)
        self.disappeared[self.nextObjectID] = 0
        self.objects_data[self.nextObjectID].add_image(*data)
        self.nextObjectID += 1

    @ray.remote
    def start_recognition(self, objectID):  # version slow [for multiple face at same time]
        names: dict[list[int]] = {}
        encodings = []
        object_amount = len(self.objects_data[objectID].get())
        for j, i in enumerate(self.objects_data[objectID].get()):
            (
                name,
                recognition_confidence,
            ), face_encodings = self.recognizer.recognition(i)
            if face_encodings is not None:
                encodings.append(face_encodings)
            if names.get(name) is None:
                names[name] = []
            names[name].append(recognition_confidence)
            self.recognition_progress.set.remote(objectID, j / object_amount)
        most_common = general.Most_Common(names)

        if most_common["name"] is None:
            self.is_checked.set.remote(objectID, False)
        elif most_common["name"] is False:
            self.is_checked.set.remote(objectID, False)
            self.objects_names.set.remote(objectID, "CHECKED_UNKNOWN")
        else:
            log(names)
            log(most_common["name"])
            check_name = most_common["name"]
            if most_common["name"] == self.recognizer.unknown and self.remember_unknown_face:
                generate_name = f"unknown:{str(uuid4().hex)}"
                if self.remember_unknown_face_maximum_confidence >= most_common["confidence"]:
                    self.pre_face_encodings.set.remote(generate_name, encodings)
                    check_name = generate_name
            else:
                if self.remember_unknown_face and most_common["name"] != self.recognizer.unknown:
                    self.pre_face_encodings.set.remote(most_common["name"], encodings)
                # check_name = most_common["name"]
            self.is_checked.set.remote(objectID, True)
            self.objects_names.set.remote(objectID, check_name)

    @ray.remote
    def start_recognition_final(self, objectID):
        names: dict[list[int]] = {}
        encodings = []

        if not ray.get(self.is_checked.get.remote(objectID)):
            object_amount = len(self.objects_data[objectID].get())
            for j, i in enumerate(self.objects_data[objectID].get()):
                (
                    name,
                    recognition_confidence,
                ), face_encodings = self.recognizer.recognition(i)
                if face_encodings is not None:
                    encodings.append(face_encodings)
                if names.get(name) is None:
                    names[name] = []
                names[name].append(recognition_confidence)
                self.recognition_progress.set.remote(objectID, j / object_amount)

            most_common = general.Most_Common(names)
            if most_common["name"] == self.recognizer.unknown:
                generate_name = f"unknown:{str(uuid4().hex)}"
                if self.remember_unknown_face_maximum_confidence >= most_common["confidence"]:
                    self.pre_face_encodings.set.remote(generate_name, encodings)
                    print("generated:", generate_name)
                else:
                    print("face_encoding error")
                    pass
                    # self.pre_face_encodings.set.remote(self.recognizer.unknown, encodings)
                self.last_deregister.recursive_update.remote({"name": generate_name}, objectID)
                print(generate_name, "final_result")
            else:
                self.last_deregister.recursive_update.remote({"name": most_common["name"]}, objectID)
                print(most_common["name"], "final_result")
        else:
            most_common = {"name": ray.get(self.objects_names.get.remote(objectID))}
            self.last_deregister.recursive_update.remote({"name": most_common["name"]}, objectID)
            print(most_common["name"], "final_result")

    def deregister(self, objectID):
        self.start_recognition_final.remote(self, objectID)
        self.last_deregister.set.remote(objectID, {"img": deepcopy(self.objects_data[objectID])})
        del self.objects[objectID]
        del self.objects_data[objectID]
        del self.disappeared[objectID]
        self.objects_ID.remove(objectID)
        self.objects_names.delete.remote(objectID)
        self.is_checked.delete.remote(objectID)

    @staticmethod
    def is_blur(cv2_img, min_confidence=100, return_value=None):
        return_value = False if return_value is None else return_value
        if cv2_img.any():
            if return_value:
                return cv2.Laplacian(cv2_img, cv2.CV_64F).var()
            else:
                return not (cv2.Laplacian(cv2_img, cv2.CV_64F).var() >= min_confidence)

    def update(self, rects):
        st_all = time.time()

        data = [list(i.values())[0] for i in rects]
        rects = [list(i.keys())[0] for i in rects]

        st = time.time()
        if len(rects) == 0:
            a = self.disappeared.copy().keys()

            for objectID in a:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects
        # print(time.time() - st, "deregister faces")

        st = time.time()
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        # print(time.time() - st, "define input centroids")

        if len(self.objects) == 0:
            st = time.time()
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], data[i])
            # print(time.time() - st, "register faces")

        else:
            st = time.time()
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # timeout = 0.05
            # object_names_preload_done, _ = ray.wait([self.objects_names.get_all.remote()], timeout=timeout)
            # object_names_preload = ray.get(object_names_preload_done[0]) if object_names_preload_done else {}
            # unknown_face_encodings_done, _ = ray.wait([self.pre_face_encodings.get_all.remote()], timeout=timeout)
            # unknown_face_encodings = ray.get(unknown_face_encodings_done[0]) if unknown_face_encodings_done else {}
            # is_checked_preload_done, _ = ray.wait([self.is_checked.get_all.remote()], timeout=timeout)
            # is_checked_preload = ray.get(is_checked_preload_done[0]) if is_checked_preload_done else {}

            # print(time.time() - st, "prepare parameter")

            st = time.time()
            object_names_preload, object_names_preload_not_done = ray.wait([self.objects_names.get_all.remote()],
                                                                           timeout=0.02)
            if object_names_preload_not_done: return self.objects
            else: object_names_preload = ray.get(object_names_preload)[0]

            # print(time.time() - st, "object name")

            st = time.time()
            is_checked_preload, is_checked_preload_not_done = ray.wait([self.is_checked.get_all.remote()],
                                                                       timeout=0.02)
            if is_checked_preload_not_done: return self.objects
            else: is_checked_preload = ray.get(is_checked_preload)[0]

            # print(time.time() - st, "is checked ")

            st = time.time()
            unknown_face_encodings, unknown_face_encodings_not_done = ray.wait([self.pre_face_encodings.get_all.remote()],
                                                                              timeout=0.02)
            if unknown_face_encodings_not_done: return self.objects
            else: unknown_face_encodings = ray.get(unknown_face_encodings)[0]

            # print(time.time() - st, "encoding\n")

            st = time.time()
            for index, (row, col) in enumerate(zip(rows, cols)):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]

                if object_names_preload.get(objectID) == "CHECKED_UNKNOWN":
                    self.objects_data[objectID].clear()
                    self.objects_names.set.remote(objectID, "__UNKNOWN__")

                self.objects[objectID] = inputCentroids[col]
                self.objects_positions.add(objectID, inputCentroids[col])
                self.objects_data[objectID].add_image(*data[col])
                self.disappeared[objectID] = 0

                if unknown_face_encodings:
                    with open(self.recognizer.name_map_path, "r", encoding="utf-8") as file:
                        raw_data = file.read()

                    if raw_data:
                        information = loads(raw_data)
                    else:
                        information = {}

                    for i in unknown_face_encodings:
                        if not i:
                            continue
                        if i in self.recognizer.loaded_id and not i.startswith("unknown:"):
                            if exists(self.faceRecPath + r"/known/" + i + ".pkl"):
                                with open(self.faceRecPath + r"/known/" + i + ".pkl", "rb") as file:
                                    existed_data: dict = pkl_load(file)
                                    existed_data["id"] = i
                                    existed_data["data"].append(unknown_face_encodings[i][0])
                                    self.recognizer.loaded_encodings.append(unknown_face_encodings[i][0])
                                    self.recognizer.loaded_id.extend([i for _ in range(len(unknown_face_encodings[i]))])
                                with open(self.faceRecPath + r"/known/" + i + ".pkl", "wb") as file:
                                    pkl_dump(existed_data, file)
                                self.pre_face_encodings.delete.remote(i)
                        else:
                            # i is "unknown:[<ID>]" but : cannot be in filename
                            if i not in self.recognizer.loaded_id:
                                information[i] = f"บุคคลปริศนา [{i.split(':')[1]}]"
                            with open(
                                    self.faceRecPath + r"/unknown/" + i.split(":")[1] + ".pkl",
                                    "wb",
                            ) as file:
                                pkl_dump({"id": i, "data": unknown_face_encodings[i]}, file)
                                self.recognizer.loaded_encodings.append(unknown_face_encodings[i][0])
                                self.recognizer.loaded_id.extend([i for _ in range(len(unknown_face_encodings[i][0]))])
                                self.recognizer.name_map[i] = information[i]
                                self.pre_face_encodings.delete.remote(i)

                    dump_information = dumps(information)
                    if dump_information:
                        with open(self.recognizer.name_map_path, "w") as file:
                            file.write(dump_information)
                if (
                        self.objects_data[objectID].lowest() >= self.minFaceConfidence
                        and self.objects_data[objectID].is_full()
                        and not is_checked_preload.get(objectID)
                ):
                    self.is_checked.set.remote(objectID, True)
                    self.objects_names.set.remote(objectID, f"IN_PROCESS")
                    self.start_recognition.remote(self, objectID)

                usedRows.add(row)
                usedCols.add(col)
            # print(time.time() - st, "do da thing")

            st = time.time()
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for index, row in enumerate(unusedRows):
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], data[col])
            # print(time.time() - st, "check if any face disappear")
        # print(time.time() - st_all, "all\n")
        return self.objects
