import glob
from DataBase import DataBase
import json
import os.path as path
import os
import pickle
import shelve


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


def name_information_init(data_path: str, name_information: str):
    """
    :param data_path: face data directory full path
    :param name_information: name information full path
    :return: None
    """

    db = DataBase("Students", sync_with_offline_db=True)
    db.offline_db_folder_path = r"C:\general\Science_project\Science_project_cp39\resources_test_2"

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
        open(name_information, 'w').close()

    files = scan_files(data_path)
    name_information_data = {}
    with open(name_information, "r", encoding="utf-8") as file:
        data = file.read()
        if data:
            name_information_data = json.loads(data)

    print("syncing name information with firebase")
    for file in files:
        filename = path.basename(path.splitext(file)[0])
        if filename not in name_information_data and "unknown:"+filename not in name_information_data:
            with open(file, "rb") as f:
                ID = pickle.loads(f.read())["id"]
                name_information_data[ID] = filename

        if db.get_data(filename) is None:
            db.add_data(filename, *DataBase.default)
        else:
            name = db.get_data(filename).get("realname") + " " + db.get_data(filename).get("surname")
            if name != " ":
                name_information_data[filename] = name

    with open(name_information, "w", encoding="utf-8") as file:
        if name_information_data:
            file.write(json.dumps(name_information_data))


if __name__ == "__main__":
    name_information_init(r"C:\general\Science_project\Science_project_cp39\resources", r"C:\general\Science_project\Science_project_cp39\resources\name_information.json")





