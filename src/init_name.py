import glob
from src.DataBase import DataBase
import json
import os.path as path
import os
from datetime import datetime, timedelta
import pickle
import shelve
from src import general
from src.recognition import Recognition
from typing import *
import google.cloud.storage.blob
import google.api_core.exceptions

def scan_files(directory: str) -> list[str]:
    def dfs(_directory: str, _file_type: str) -> list:
        if path.isdir(_directory):
            _files: list[str] = []
            for _file in glob.glob(_directory + "\\*"):
                _files.extend(dfs(_file, _file_type))
            return _files
        else:
            if path.splitext(_directory)[1] == _file_type:
                return [_directory]
            return []

    if not path.exists(directory):
        return [""]

    return dfs(directory, ".pkl")


def remove_expire_unknown_faces(data_path: str, unknown_face_life_time=timedelta(hours=24)):
    files = scan_files(data_path)
    unknown_path = {}
    for file in files:
        unknown_path[path.splitext(path.basename(file))[0]] = file

    db = DataBase("Students", sync_with_offline_db=True)
    db.offline_db_folder_path = data_path
    db_data = db.get_database()
    for datum in db_data:
        if datum.startswith("unknown:"):
            if datetime.now() > (datetime.fromtimestamp(db_data[datum]["last_checked"]) + unknown_face_life_time):
                print(f"{datum} expire for "
                      f"{datetime.now()-datetime.fromtimestamp(db_data[datum]['last_checked'])+unknown_face_life_time}")
                print("\tpath:", unknown_path.get(datum.lstrip("unknown:")))

                db.delete(datum)
                if unknown_path.get(datum.lstrip("unknown:")) is not None:
                    os.remove(unknown_path.get(datum.lstrip("unknown:")))


def name_information_init(data_path: str, name_information: str, certificate_path: str = "src/serviceAccountKey.json"):

    db = DataBase("Students", sync_with_offline_db=True, certificate_path=certificate_path)
    db.offline_db_folder_path = data_path

    if db.sync_with_offline_db and db.can_connect():
        with shelve.open(db.offline_db_folder_path + "/" + db.db_name) as off_db:
            if db.latest_update_is_online() is None:
                print("firebase and offline database is the same.")
            elif not db.latest_update_is_online():
                print("syncing firebase with offline database...")
                db.set_database(dict(off_db))
            else:
                on_data = db.get_database()
                print("syncing offline database with firebase...")
                for key in off_db.keys():
                    if key not in on_data:
                        off_db.pop(key)
                off_db.update(on_data)
    else:
        if db.sync_with_offline_db:
            print("internet not found: use offline database")

    if not path.exists(name_information):
        open(name_information, "w").close()

    files = scan_files(data_path)
    name_information_data = {}
    with open(name_information, "r", encoding="utf-8") as file:
        data = file.read()
        if data:
            name_information_data = json.loads(data)

    print("syncing name information with firebase")
    for file in files:
        filename = path.basename(path.splitext(file)[0])
        if filename not in name_information_data and "unknown:" + filename not in name_information_data:
            with open(file, "rb") as f:
                try:
                    ID = pickle.loads(f.read())["id"]
                    name_information_data[ID] = filename
                except pickle.UnpicklingError:
                    raise pickle.UnpicklingError(f"unpickling error at file \"{file}\"")

        if db.get_data(filename) is None and db.get_data("unknown:" + filename) is None:
            print(path.basename(path.dirname(file)))
            if path.basename(path.dirname(file)) == "known":
                db.add_data(filename, *DataBase.default)
            else:
                db.add_data("unknown:"+filename, *DataBase.default)
        else:
            try:
                name = db.get_data(filename).get("realname") + " " + db.get_data(filename).get("surname")
            except AttributeError:
                name = db.get_data("unknown:"+filename).get("realname") + " " + db.get_data("unknown:"+filename).get("surname")
            if name != " ":
                name_information_data[filename] = name

    with open(name_information, "w", encoding="utf-8") as file:
        if name_information_data:
            file.write(json.dumps(name_information_data))


def init_shared(data_path: str, cache_path: str,  certificate_path: str = "src/serviceAccountKey.json"):
    db: DataBase = DataBase("Students", sync_with_offline_db=True, certificate_path=certificate_path)
    db.offline_db_folder_path = data_path
    storage: DataBase.Storage = db.Storage(cache=cache_path)

    files = general.scan_files(data_path, ".pkl")
    print(files)
    pfp: Recognition.ProcessedFacePool = Recognition.ProcessedFacePool.from_filenames(files)
    share_filenames = [i.name.lstrip("shared/") for i in list(storage.bucket.list_blobs(prefix="shared/"))][1:]
    share_files: List[google.cloud.storage.blob.Blob] = [i for i in list(storage.bucket.list_blobs(prefix="shared/"))][1:]

    print(share_filenames)

    for identity in pfp.get_identities():
        if identity not in share_filenames:
            if not pfp.get_encoding(identity).is_unknown:
                print("Upload", f"{identity}.pkl")
                pfp.get_encoding(identity).to_file(os.path.join(cache_path, f"{identity}_temp.pkl"))
                storage.add_encoding_file(identity, os.path.join(cache_path, f"{identity}_temp.pkl"), prefix_path="shared/")
                os.remove(os.path.join(cache_path, f"{identity}_temp.pkl"))

    identity_blob: google.cloud.storage.blob.Blob
    identity: str
    for identity,  identity_blob in zip(share_filenames, share_files):
        if identity not in pfp.get_identities():
            print("Download", os.path.basename(os.path.join(data_path, "known", f"{identity}.pkl")))
            try:
                identity_blob.download_to_filename(os.path.join(data_path, "known", f"{identity}.pkl"))
            except google.api_core.exceptions.NotFound:
                print(f"{identity}.pkl not found!")
                if os.path.exists(os.path.join(data_path, "known", f"{identity}.pkl")):
                    os.remove(os.path.join(data_path, "known", f"{identity}.pkl"))


if __name__ == "__main__":
    init_shared("../recognition_resources_temp", "../cache", "../src/resources/serviceAccountKey.json")
