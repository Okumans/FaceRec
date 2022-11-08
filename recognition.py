import glob
import os.path as path
import pickle
import face_recognition
import numpy as np
from general import log, Color
import cv2, time


def get_all_filename(data_path):
    if path.isdir(data_path):
        files = []
        for file in glob.glob(data_path+"\\*"):
            files.extend(get_all_filename(file))
        return files
    else:
        return [data_path]


class Recognition:
    def __init__(self, data_path):
        self.loaded_encodings = []
        self.loaded_id = []
        if path.isdir(data_path):
            for filename in get_all_filename(data_path):
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
        log(list(set(self.loaded_id)), color=Color.Cyan)

    def recognition(self, img):
        try:
            face_location = face_recognition.face_locations(img)
            new_encoding = face_recognition.face_encodings(img, face_location)[0]
        except IndexError: return False
        matches = face_recognition.compare_faces(self.loaded_encodings, new_encoding)
        face_distances = face_recognition.face_distance(self.loaded_encodings, new_encoding)
        best_match_index = np.argmin(face_distances)
        name = "UNKNOWN"
        if matches[best_match_index]:
            name = self.loaded_id[best_match_index]
        return name


