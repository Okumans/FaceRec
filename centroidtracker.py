# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import topData
import numpy as np
import general
from general import log, Color
import cv2
import ray
from recognition import Recognition


class CentroidTracker:
    def __init__(self, maxDisappeared=10, minFaceConfidence=.85, minFaceBlur=100, faceCheckAmount=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objects_data = OrderedDict()
        self.disappeared = OrderedDict()
        self.recognizer = Recognition(r"C:\general\Science_project\Science_project_cp39\\resources")
        self.objects_names = general.rayDict.remote()
        self.is_checked = general.rayDict.remote()
        self.maxDisappeared = maxDisappeared
        self.minFaceConfidence = minFaceConfidence
        self.minFaceBlur = minFaceBlur
        self.faceCheckAmount = faceCheckAmount

    def register(self, centroid, data):
        self.objects[self.nextObjectID] = centroid
        self.objects_data[self.nextObjectID] = topData.topData(min_detection=self.minFaceBlur, max_size=self.faceCheckAmount)
        self.objects_names.set.remote(self.nextObjectID, "UNKNOWN")
        self.is_checked.set.remote(self.nextObjectID, False)
        self.disappeared[self.nextObjectID] = 0
        self.objects_data[self.nextObjectID].add(*data)
        self.nextObjectID += 1

    @ray.remote
    def start_recognition(self, objectID):  # version slow [for multiple face at same time]
        names = []
        for j, i in enumerate(self.objects_data[objectID].get()):
            names.append(self.recognizer.recognition(i))
        most_common = general.Most_Common(names)

        if most_common is None:
            self.is_checked.set.remote(objectID, False)
            self.objects_names.set.remote(objectID, "CHECKED_UNKNOWN")
        elif most_common is False:
            self.is_checked.set.remote(objectID, False)
        else:
            log(names)
            log(most_common)
            self.is_checked.set.remote(objectID, True)
            self.objects_names.set.remote(objectID, most_common)

    @ray.remote
    def __recognition(self, objectData):  # recognition worker for fast recognition method
        return self.recognizer.recognition(objectData)

    @ray.remote
    def _start_recognition(self, objectID):
        names = []
        for j, i in enumerate(self.objects_data[objectID].get()):
            names.append(self.__recognition.remote(self, i))

        names = ray.get(names)
        most_common = general.Most_Common(names)
        print(names)
        print(most_common)
        if most_common is None:
            self.is_checked.set.remote(objectID, False)
            self.objects_names.set.remote(objectID, "CHECKED_UNKNOWN")
        elif most_common is False:
            self.is_checked.set.remote(objectID, False)
            self.objects_names.set.remote(objectID, "CHECKED_UNKNOWN")
        else:
            self.is_checked.set.remote(objectID, True)
            self.objects_names.set.remote(objectID, most_common)

    @ray.remote
    def start_recognition_final(self, objectID):
        names = []
        for j, i in enumerate(self.objects_data[objectID].get()):
            names.append(self.recognizer.recognition(i))
        most_common = general.Most_Common(names)
        print(names)
        print(most_common, "final#####")

    def deregister(self, objectID):
        self.start_recognition.remote(self, objectID)
        self.start_recognition_final.remote(self, objectID)

        del self.objects[objectID]
        del self.objects_data[objectID]
        del self.disappeared[objectID]
        self.objects_names.delete.remote(objectID)
        self.is_checked.delete.remote(objectID)

    @staticmethod
    def is_blur(cv2_img, min_confidence=100, return_value=None):
        return_value = False if return_value is None else return_value
        if cv2_img.any():
            if return_value: return cv2.Laplacian(cv2_img, cv2.CV_64F).var()
            else: return not (cv2.Laplacian(cv2_img, cv2.CV_64F).var() >= min_confidence)

    def update(self, rects):
        data = [list(i.values())[0] for i in rects]
        rects = [list(i.keys())[0] for i in rects]

        if len(rects) == 0:
            a = self.disappeared.copy().keys()
            for objectID in a:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], data[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for index, (row, col) in enumerate(zip(rows, cols)):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.objects_data[objectID].add(*data[col])
                self.disappeared[objectID] = 0

                if self.objects_data[objectID].lowest() >= self.minFaceConfidence and self.objects_data[objectID].is_full() and not ray.get(self.is_checked.get.remote(objectID)):
                    self.is_checked.set.remote(objectID, True)
                    self.objects_names.set.remote(objectID, "IN_PROCESS")
                    self.start_recognition.remote(self, objectID)

                usedRows.add(row)
                usedCols.add(col)

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

        return self.objects
