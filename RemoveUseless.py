from src.RemoveUseless import RemoveUseless
from typing import *
from src.DataBase import DataBase
from src.general import print_msg_box, msg_box


def ask(message: str, input_type: Callable) -> Any:
    print_msg_box(message)
    return input_type(input(": "))


def show(message: str):
    print_msg_box(message)


def choice(message: str, choices: List, ignore_case=False) -> str:
    data = msg_box(message) + "\n"
    for i in choices:
        data += f"   ◆ {i}\n"
    print_msg_box(data)

    ch = input(": ")
    choices = list(map(lambda a: a.lower(), choices)) if ignore_case is True else choices
    while (ch.strip() if ignore_case is False else ch.strip().lower()) not in choices:
        ch = input(": ")
    return ch


def ask_y_n(message: str, ignore_case=False, return_boolean=False, upper_y=False):
    print_msg_box(message + " (y/n)")
    ch = input(": ")
    choices = ("y", "n") if upper_y is False else ("Y", "n")
    while (ch.strip() if ignore_case is False else ch.strip().lower()) not in choices:
        ch = input(": ")
    return (ch == ("y" if upper_y is False else "Y")) if return_boolean is True else ch


def ask_until_sure(message: str, input_type: Callable):
    ans: str = ""
    sure: bool = False
    while not sure:
        ans = ask(message, str)
        sure = ask_y_n(f"are you sure? [{ans}]", ignore_case=True, return_boolean=True)
    return input_type(ans)


if __name__ == "__main__":
    face_reg_path = "C:/general/Science_project/Science_project_cp39_refactor/recognition_resources"
    cred_path = "C:/general/Science_project/Science_project_cp39_refactor/src/resources/serviceAccountKey.json"
    if ask_y_n("This process is very risk. it will delete every unused information. and all can not be recovered.\n"
               "are you sure to process?", ignore_case=True, return_boolean=True):
        if not ask_y_n(f"Is the target of this process (recognition_resources) \"{face_reg_path}\"",
                       ignore_case=True, return_boolean=True):
            face_reg_path = ask_until_sure("new path to recognition_resources", str)
        name_map_path = face_reg_path + "/name_information.json"
        if not ask_y_n(f"Is the cred of the target database\"{cred_path}\"",
                       ignore_case=True, return_boolean=True):
            cred_path = ask_until_sure("new path to cred", str)

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

        if ask_y_n("The shown identities will be deleted.\n are you sure", ignore_case=False, return_boolean=True,
                   upper_y=True):
            print(idd)
            rmul.delete_identities(idd)

        if ask_y_n("The shown Images idd will be deleted.\n are you sure", ignore_case=False, return_boolean=True,
                   upper_y=True):
            print(images)
            rmul.delete_images(images)

