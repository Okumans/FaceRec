from general import scan_files
from face_recognition import face_encodings, face_distance
import pickle
import numpy as np
from typing import Any
import os.path as path


class ContaminationScanner:
    def __init__(self, target_path: str, min_face_rec_confidence: float = 0.5):
        self.target_path = target_path
        self.files = scan_files(self.target_path, ".pkl")
        self.min_face_rec_confidence = min_face_rec_confidence

    def scan(self):
        for file in self.files:
            with open(file, "rb") as f:
                raw_information: dict[str, Any] = pickle.loads(f.read())
                ID: str = raw_information["id"]
                data: list[np.ndarray] = raw_information["data"]

                print(f"SCANNING {path.basename(file)}")
                new_data: list[np.ndarray] = []
                for dat in data:
                    similarity = 1-(sum(face_distance(data, dat))/len(data))
                    if similarity < self.min_face_rec_confidence:
                        print("\tFOUND NOT QUALIFY: ", similarity)
                    else:
                        new_data.append(dat)
            new_information = {"id": ID, "data": new_data}
            if new_information != raw_information:
                print(f"\tOVERWRITE {path.basename(file)}")
                with open(file, "wb") as f:
                    f.write(pickle.dumps(new_information))

    @staticmethod
    def scan_now(encodings: list[np.ndarray], min_face_rec_confidence: float = 0.5):
        encodings_length: int = len(encodings)
        new_encodings: list[np.ndarray] = []
        for encoding in encodings:
            similarity = 1 - (sum(face_distance(encodings, encoding)) / encodings_length)
            if similarity < min_face_rec_confidence:
                print("\tFOUND NOT QUALIFY: ", similarity)
            else:
                new_encodings.append(encoding)
        return new_encodings


if __name__ == "__main__":
    cs = ContaminationScanner("resources_test_2")
    cs.scan()
