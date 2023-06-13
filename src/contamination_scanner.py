from src.general import scan_files
from face_recognition import face_encodings, face_distance
import pickle
import numpy as np
from src.DataBase import DataBase
from typing import Any
import os.path as path
from src.general import Average, msg_box


class ContaminationScanner:
    def __init__(self, target_path: str, min_face_rec_confidence: float = 0.65):
        self.target_path = target_path
        self.files = scan_files(self.target_path, ".pkl")
        self.min_face_rec_confidence = min_face_rec_confidence
        self.__db = DataBase("Students", sync_with_offline_db=True)
        self.__db.offline_db_folder_path = self.target_path

    def scan(self):
        for file in self.files:
            with open(file, "rb") as f:
                raw_information: dict[str, Any] = pickle.loads(f.read())
                ID: str = raw_information["id"]
                data: list[np.ndarray] = raw_information["data"]

                print(f"\rSCANNING QUALIFY {path.basename(file)}", end="")
                new_data: list[np.ndarray] = []
                for dat in data:
                    similarity = 1 - (sum(face_distance(data, dat)) / len(data))
                    if similarity < self.min_face_rec_confidence:
                        print("\n\tFOUND NOT QUALIFY: ", similarity)
                    else:
                        new_data.append(dat)
            new_information = {"id": ID, "data": new_data}
            if new_information != raw_information:
                print(f"\n\tOVERWRITE {path.basename(file)}")
                with open(file, "wb") as f:
                    f.write(pickle.dumps(new_information))

    def scan_duplicate(self):
        for file_consider in self.files:
            max_similarity = 0
            with open(file_consider, "rb") as f:
                raw_information_consider: dict[str, Any] = pickle.loads(f.read())
                ID_consider: str = raw_information_consider["id"]
                data_consider: list[np.ndarray] = raw_information_consider["data"]

                for file_another in self.files:
                    print(f"\rSCANNING DUPLICATE {path.basename(file_another)}::{path.basename(file_consider)}", end="")
                    with open(file_another, "rb") as f_:
                        raw_information_another: dict[str, Any] = pickle.loads(f_.read())
                        ID_another: str = raw_information_another["id"]
                        data_another: list[np.ndarray] = raw_information_another["data"]

                        if self.__db.quick_get_data(ID_consider).get("parent") is None:
                            self.__db.update(ID_consider, parent=ID_consider)

                        # print(ID_another, "debug")
                        if self.__db.quick_get_data(ID_another).get("parent") is None:
                            self.__db.update(ID_another, parent=ID_another)

                        similarity = Average()
                        for datum_consider in data_consider:
                            if data_another:
                                similarity.adds(face_distance(data_another, datum_consider))

                        max_similarity = max(max_similarity, similarity.get())
                        if max_similarity < similarity.get():
                            continue

                        if 1 - similarity.get() > .5 and similarity.length != 0 and ID_consider != ID_another:
                            if not ID_consider.startswith("unknown:"):
                                if ID_another.startswith("unknown:"):
                                    self.__db.update(ID_another, parent=ID_consider)
                                    print(f"{ID_another} is set as child of {ID_consider}"
                                          f"similarity:{(1-similarity.get())*100}%")

                            else:
                                if ID_another.startswith("unknown:") and self.__db.quick_get_data(ID_consider).get("parent") != ID_consider:
                                    self.__db.update(ID_another, parent=self.__db.quick_get_data(ID_consider).get("parent"))
                                    print(f"{ID_another} is set as child of {self.__db.quick_get_data(ID_consider).get('parent')}")

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
    cs = ContaminationScanner("resources_test_3")
    cs.scan()
