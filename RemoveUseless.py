from src.RemoveUseless import RemoveUseless
from typing import *
from src.DataBase import DataBase
from src.general import print_msg_box, msg_box, MessageIO
from src.recognition import Recognition

if __name__ == "__main__":
    face_reg_path = "C:/general/Science_project/Science_project_cp39_refactor/recognition_resources"
    cred_path = "C:/general/Science_project/Science_project_cp39_refactor/src/resources/serviceAccountKey.json"
    if MessageIO.ask_y_n(
            "This process is very risk. it will delete every unused information. and all can not be recovered.\n"
            "are you sure to process?", ignore_case=True, return_boolean=True):
        if not MessageIO.ask_y_n(f"Is the target of this process (recognition_resources) \"{face_reg_path}\"",
                                 ignore_case=True, return_boolean=True):
            face_reg_path = MessageIO.ask_until_sure("new path to recognition_resources", str)
        name_map_path = face_reg_path + "/name_information.json"
        if not MessageIO.ask_y_n(f"Is the cred of the target database\"{cred_path}\"",
                                 ignore_case=True, return_boolean=True):
            cred_path = MessageIO.ask_until_sure("new path to cred", str)

        print_msg_box("initializing Recognition model\n" + msg_box(
            f"recognition_resource: \"{face_reg_path}\"\n"
            "remember_unknown_face: False\n"
            f"name_map_path: \"{name_map_path}\""), title="Info")

        recognizer = Recognition(face_reg_path, False, name_map_path)

        print_msg_box("initializing Database\n" + msg_box(
            f"database name: \"Students\"\n"
            "sync_with_offline_database: False\n"
            f"cred_path: \"{cred_path}\""), title="Info")

        db = DataBase("Students", certificate_path=cred_path)

        print_msg_box("initializing RemoveUseless module\n" +
                      msg_box(
                          f"recognition_resource: \"{face_reg_path}\"\n" +
                          msg_box(
                              f"database name: \"Students\"\n"
                              "sync_with_offline_database: False\n"
                              f"cred_path: \"{cred_path}\"", title="Database: ") + "\n" +
                          msg_box(
                              f"recognition_resource: \"{face_reg_path}\"\n"
                              "remember_unknown_face: False\n"
                              f"name_map_path: \"{name_map_path}\"", title="Recognition module: ") + "\n" +
                          f"cred_path: \"{cred_path}\""), title="Info")

        rmul = RemoveUseless(face_reg_path, db, recognizer)

        idd, images, cache_images = rmul.should_delete()

        if MessageIO.ask_y_n("The shown identities will be deleted.\n are you sure", ignore_case=False,
                             return_boolean=True,
                             upper_y=True):
            print(idd)
            rmul.delete_identities(idd)

        if MessageIO.ask_y_n("The shown Images idd will be deleted.\n are you sure", ignore_case=False,
                             return_boolean=True,
                             upper_y=True):
            print(images)
            rmul.delete_images(images)
