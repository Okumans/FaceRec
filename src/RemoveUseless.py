from __future__ import annotations
from src import general
from src.DataBase import DataBase
from typing import *
from src.recognition import Recognition
from src.studentSorter import Student


def add_str_nl(base, *args, sep=" ", end="\n"):
    return base + sep.join(args) + end


class RemoveUseless:
    DEFAULT_EXCEPTION = ["unknown_people.png", "encoded/", "last_update"]

    def __init__(self, face_rec_path: str, db: DataBase, recognizer: Recognition, keyword_exceptions=None, **kwargs):
        self.target: str = face_rec_path
        self.recognizer: Recognition = recognizer
        self.db: DataBase = db
        self.storage = DataBase.Storage = db.Storage(cache=kwargs.get("cache", None))
        self.keyword_exceptions = RemoveUseless.DEFAULT_EXCEPTION.copy().append(keyword_exceptions) \
            if keyword_exceptions is not None else RemoveUseless.DEFAULT_EXCEPTION
        self.recognizer.update(self.storage)
        self.files: List[str] = general.scan_files(self.target)
        self.all_identities: Set[str] = set(self.db.get_database().keys())
        self.found_identities: Set[str] = set(Recognition.ProcessedFacePool.from_filenames(self.files).get_identities())
        self.all_images: Set[str] = set(list(i.name for i in self.storage.bucket.list_blobs()))

    @property
    def not_found_in_local(self) -> Set[str]:
        return self.all_identities.difference(self.found_identities)

    @property
    def found_in_local(self) -> Set[str]:
        return self.all_identities.intersection(self.found_identities)

    @property
    def image_not_found_in_local(self) -> Set[str]:
        return self.all_images.difference(self.found_identities)

    def should_delete(self):
        idds = self.not_found_in_local
        result = ""
        amount = 0

        idd_use = []
        image_use = []
        image_cache_use = []

        result += add_str_nl(general.msg_box("Identities"))
        for idd in idds:
            amount += 1
            if idd not in self.keyword_exceptions:
                try:
                    result += "  " + add_str_nl(Student().load_from_db(self.db, idd).get_table()) + "\n"
                except TypeError:
                    result += "  " + add_str_nl(idd) + "\n"
                idd_use.append(idd)

        general.print_msg_box(result)
        result = ""

        amount_image_raw = set()
        result += add_str_nl(general.msg_box("Images"))
        for idd in idds:
            if self.storage.exists(idd) and idd not in self.keyword_exceptions:
                amount_image_raw.add(idd)
                try:
                    result += add_str_nl(Student().load_from_db(self.db, idd).get_table()) + "\n"
                except TypeError:
                    result += add_str_nl(idd) + "\n"
                image_use.append(idd)

        for idd in self.image_not_found_in_local:
            if self.storage.exists(idd) and idd not in self.keyword_exceptions:
                amount_image_raw.add(idd)
                try:
                    result += add_str_nl(Student().load_from_db(self.db, idd).get_table()) + "\n"
                except TypeError:
                    result += add_str_nl(idd) + "\n"
                image_use.append(idd)

        general.print_msg_box(result)
        result = ""

        if self.storage.cache:
            result += add_str_nl(general.msg_box("Cache Images"))
            for idd in idds:
                if self.storage.get_cache_image(idd) is not None and idd not in self.keyword_exceptions:
                    amount += 1
                    try:
                        result += "  " + add_str_nl(Student().load_from_db(self.db, idd).get_table()) + "\n"
                    except TypeError:
                        result += "  " + add_str_nl(idd) + "\n"
                    image_cache_use.append(idd)

            general.print_msg_box(result)
            result = ""

        amount += len(amount_image_raw)
        result += add_str_nl(general.msg_box(f"Total of {amount} items can be removed"))

        print(result)
        return list(set(idd_use)), list(set(image_use)), list(set(image_cache_use))

    def delete_identities(self, list_of_idd):
        print("deleting identities information")
        for idd in list_of_idd:
            self.db.delete(idd)
            print(f"deleted identity \"{idd}\"")

    def delete_images(self, list_of_image_idd):
        print("deleting images")
        for idd in list_of_image_idd:
            self.storage.delete(idd)
            print(f"deleted identity \"{idd}\"")

    def delete_cache_images(self, list_of_image_idd):
        print("deleting cache images")
        for idd in list_of_image_idd:
            self.storage.delete(idd)
            print(f"deleted identity \"{idd}\"")


if __name__ == "__main__":
    face_reg_path = "C:/general/Science_project/Science_project_cp39_refactor/recognition_resources"
    name_map_path = face_reg_path + "/name_information.json"
    cred_path = "C:/general/Science_project/Science_project_cp39_refactor/src/resources/serviceAccountKey.json"
    recognizer = Recognition(face_reg_path, False, name_map_path)
    db = DataBase("Students", certificate_path=cred_path)
    rmul = RemoveUseless(face_reg_path, db, recognizer)
    print(rmul.should_delete())

