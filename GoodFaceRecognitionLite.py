import json
import logging
import os
import os.path as path
import sys
import time
from copy import deepcopy
from datetime import datetime
from json import dumps, loads
from threading import Thread
from typing import *

import cv2
import mediapipe as mp
import numpy as np
import ray
from PyQt5 import QtGui
from PyQt5.QtCore import (
    pyqtSignal,
    pyqtSlot,
    Qt,
    QThread,
    QSize,
    QRect,
    QCoreApplication,
    QVariantAnimation,
    pyqtBoundSignal,
)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QScrollArea,
    QSpacerItem,
    QMainWindow,
    QBoxLayout,
    QSplitter
)
from ray.exceptions import GetTimeoutError

from src import general
from src import ui_popupLite  # for setting popup
from src import ui_popup2  # for infobox popup
from src.DataBase import DataBase
from src.FaceAlignment import face_alignment
from src.ShadowRemoval import remove_shadow_grey
from src.attendant_graph import AttendantGraph, Arrange
from src.centroidtracker import CentroidTracker
from src.general import Color, get_from_percent, RepeatedTimer
from src.init_name import name_information_init, init_shared
from src.recognition import Recognition
from src.scrollbar_style import scrollbar_style
from src.studentSorter import Student

try:
    from picamera2 import Picamera2
except ImportError:
    pass


# -----------------setting-----------------
try:
    with open("settings.json", "r") as f:
        setting = json.load(f)
except FileNotFoundError:
    print("setting not found!, please run setup.py first.")
    quit()

use_folder = [setting['project_path'], setting["face_reg_path"], setting["face_reg_path"] + r"/unknown",
              setting["face_reg_path"] + r"/known", setting['project_path'] + r"/cache"]

RAY_TIMEOUT = 0.005

# -------------global variable--------------
if __name__ == "__main__":
    print(f"project path set to \"{setting['project_path']}\"")

    for folder_path in use_folder:
        if not path.exists(folder_path):
            os.mkdir(folder_path)

    name_information_init(setting["face_reg_path"], setting["name_map_path"], certificate_path=setting["db_cred_path"])
    init_shared(setting["face_reg_path"], setting["cache_path"], certificate_path=setting["db_cred_path"])
    # remove_expire_unknown_faces(setting["face_reg_path"])
    # ContaminationScanner(setting["face_reg_path"], .65).scan()
    # ContaminationScanner(setting["face_reg_path"], .8).scan_duplicate()

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    # emotion_recognizer = FER()
    ct = CentroidTracker(
        faceRecPath=setting["face_reg_path"],
        maxDisappeared=setting["face_max_disappeared"],
        minFaceBlur=setting["min_faceBlur_detection"],
        minFaceConfidence=setting["min_detection_confidence"],
        minFaceRecConfidence=setting["min_recognition_confidence"],
        faceCheckAmount=setting["face_check_amount"],
        remember_unknown_face=setting["remember_unknown_face"],
        otherSetting=setting,
    )
    ct.recognizer.face_detection_method = "hog"
    (H, W) = (None, None)
    text_color = (31, 222, 187)
    prev_frame_time = 0  # fps counter
    new_frame_time = 0
    last_id = -1
    now_frame = 0  # cv2 mat
    already_check = {}
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image_error = cv2.imread(setting["project_path"] + "/src/resources/image_error.png")
    unknown_image = cv2.imread(setting["project_path"] + "/src/resources/unknown_people.png")


class VideoThread(QThread):
    change_pixmap_signal: pyqtBoundSignal = pyqtSignal(np.ndarray)
    change_infobox_message_signal: pyqtBoundSignal = pyqtSignal(dict)

    def __init__(self):
        global H, W
        super().__init__()

        (H, W) = (None, None)
        self.video_type: str = setting["video_source"]
        self.db: DataBase = DataBase("Students",
                                     sync_with_offline_db=True)
        self.db.offline_db_folder_path = setting["face_reg_path"]
        self.run: bool = True
        self.frame_index: int = 0
        self.objname_map: Dict[int, any] = {}
        self.avg_fps: general.Average = general.Average([0], calculate_amount=100)

        # for windows
        if setting.get("platform", "win") == "win":
            self.cap = cv2.VideoCapture(0)
        elif setting.get("platform", "win") == "rpi":
            self.cap = Picamera2()
            self.cap.create_video_configuration({"size": (640, 480)})
            self.cap.start()

    def release_cam(self):
        if setting.get("platform", "win") == "win":
            self.cap.release()
        elif setting.get("platform", "win") == "rpi":
            self.cap.close()
        self.run = False

    def run(self):
        global H, W, prev_frame_time, new_frame_time, last_id, now_frame
        with mp_face_detection.FaceDetection(min_detection_confidence=setting["min_detection_confidence"],
                                             model_selection=1) as face_detection:
            while self.run or self.cap.isOpened():
                # success, image = True, self.cap.capture_array()
                success, image = self.cap.read()
                new_frame_time = time.time()

                if not success or image is None:
                    logging.info(f"Ignoring empty camera frame.")
                    continue

                if H is None or W is None:
                    (H, W) = image.shape[:2]
                    setting["base_resolution"] = (W, H)

                for i in range(setting["rotate_frame"]):
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                ct.maxDisappeared = setting["face_max_disappeared"]
                ct.minFaceBlur = setting["min_faceBlur_detection"]
                ct.minFaceConfidence = setting["min_detection_confidence"]
                ct.faceCheckAmount = setting["face_check_amount"]
                ct.recognizer.min_confidence = setting["min_recognition_confidence"]
                ct.recognizer.remember = setting["remember_unknown_face"]

                now_frame = image
                image.flags.writeable = False
                image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)

                # change the brightness, contrast, sharpness, gray_mode as it assign to the settings
                if setting["autoBrightnessContrast"]:
                    image = general.change_brightness_to(image, setting["autoBrightnessValue"])
                    image = general.change_contrast_to(image, setting["autoContrastValue"])

                if setting["sharpness_filter"]:
                    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

                # for night vision or bad lighting condition
                if setting["gray_mode"]:
                    image = cv2.cvtColor(remove_shadow_grey(image), cv2.COLOR_GRAY2RGB)
                    general.putBorderText(
                        image,
                        "NIGHT MODE",
                        (W - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Color.Violet,
                        Color.Black,
                        2,
                        3
                    )

                # st = time.time()
                results: NamedTuple = face_detection.process(image)
                # print(time.time()-st, 1/((time.time()-st) if time.time()-st != 0 else -1), "process..")

                image.flags.writeable = True
                image_use = deepcopy(image)
                image = cv2.resize(image, (W, H))
                rects = []

                # st = time.time()
                if results.detections:
                    for detection in results.detections:
                        x_min = detection.location_data.relative_bounding_box.xmin * W * setting["resolution"]
                        y_min = detection.location_data.relative_bounding_box.ymin * H * setting["resolution"]
                        x_max = x_min + detection.location_data.relative_bounding_box.width * W * setting["resolution"]
                        y_max = y_min + detection.location_data.relative_bounding_box.height * H * setting["resolution"]
                        face_height = y_max - y_min
                        box = (x_min, y_min, x_max, y_max)

                        face_image = deepcopy(
                            image_use[
                                int((int(box[1]) - get_from_percent(face_height, 20)) / setting["resolution"]):
                                int((int(box[3]) + get_from_percent(face_height, 20)) / setting["resolution"]),
                                int((int(box[0]) - get_from_percent(face_height, 20)) / setting["resolution"]):
                                int((int(box[2]) + get_from_percent(face_height, 20)) / setting["resolution"]),
                            ]
                        )
                        (int(box[0] / setting["resolution"]), int(box[1] / setting["resolution"])),
                        (int(box[2] / setting["resolution"]), int(box[3] / setting["resolution"])),
                        # align face if face_alignment is on
                        if setting["face_alignment"]:
                            try:
                                face_image = face_alignment(face_image, detection)
                            except TypeError:
                                pass

                        distance: float = ((H * setting["resolution"] * W * setting["resolution"]) / 518400) * (
                                ((face_image.shape[1] + face_image.shape[0]) / 2) / 220.39
                        )
                        distance = distance ** (1 / -0.949) if distance != 0 else 1000000000000000
                        # check if face data is enough for face recognizing
                        if face_height >= 60:
                            rects.append({box: (detection.score[0], face_image)})

                        # display debug message in image if debug is on
                        if setting["debug"]:
                            general.putBorderText(
                                image,
                                f"confident: {round(detection.score[0], 2)}% "
                                f"blur {CentroidTracker.is_blur(face_image, setting['min_faceBlur_detection'])} ",
                                (int(box[0]), int(box[1]) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                (0, 0, 0),
                                2,
                                3,
                            )
                            general.putBorderText(
                                image,
                                f"brightness: {round(general.brightness(face_image), 2)} "
                                f"contrast: {round(general.contrast(face_image), 2)}",
                                (int(box[0]), int(box[1]) + 38),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                (0, 0, 0),
                                2,
                                3,
                            )

                            general.putBorderText(
                                image,
                                f"size(WxH): {face_image.shape[1]}, {face_image.shape[0]} distance-predict: {distance}",
                                (int(box[0]), int(box[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                (0, 0, 0),
                                2,
                                3,
                            )

                            general.putBorderText(
                                image,
                                f"Not supported",
                                (int(box[0]), int(box[1]) - 48),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                (0, 0, 0),
                                2,
                                3,
                            )

                        cv2.rectangle(
                            image,
                            (int(box[0] / setting["resolution"]), int(box[1] / setting["resolution"])),
                            (int(box[2] / setting["resolution"]), int(box[3] / setting["resolution"])),
                            (0, 0, 0),
                            5,
                        )

                        cv2.rectangle(
                            image,
                            (int(box[0] / setting["resolution"]), int(box[1] / setting["resolution"])),
                            (int(box[2] / setting["resolution"]), int(box[3] / setting["resolution"])),
                            text_color,  # if 0.3 <= distance <= 3.5 and face_height >= 60 else (255, 0, 0),
                            3,
                        )

                objects = ct.update(rects)

                temp = last_id
                objects_ids = [i[0] for i in objects.items()]
                if objects_ids:
                    last_id = max(objects_ids)

                for i in objects_ids:
                    if i > temp:
                        already_check[i] = False

                update_current_identity = True

                try:
                    names = ray.get(ct.objects_names.get_all.remote(), timeout=RAY_TIMEOUT)
                    progresses = ray.get(ct.recognition_progress.get_all.remote(), timeout=RAY_TIMEOUT)
                except GetTimeoutError:
                    update_current_identity = False

                if update_current_identity:
                    for i in objects_ids:
                        name = names.get(i)
                        progress = progresses.get(i)

                        if name == "IN_PROCESS":
                            self.change_infobox_message_signal.emit(
                                {
                                    "name": name,
                                    "ID": i,
                                    "image": ct.objects_data[i].get()[0],
                                    "progress": progress,
                                }
                            )
                        else:
                            if (name not in [
                                "UNKNOWN",
                                "CHECKED_UNKNOWN",
                                "__UNKNOWN__",
                                None,
                                False,
                                ""
                            ] and already_check[i] is False):

                                already_check[i] = True
                                ct.last_deregister.delete.remote(i)

                                if name != ct.recognizer.unknown and not name.startswith("attacked:"):
                                    data = self.db.get_data(name)
                                    if data is None:
                                        self.db.add_data(name, *DataBase.default)
                                        data = self.db.get_data(name)

                                    now_time = time.time()
                                    graph_info = data.get("graph_info") if data.get("graph_info") is not None else []
                                    graph_info.append(now_time)
                                    self.db.update(name, last_checked=now_time, graph_info=graph_info)

                                self.change_infobox_message_signal.emit(
                                    {
                                        "name": name,
                                        "ID": i,
                                        "image": ct.objects_data[i].get()[0],
                                        "progress": 0.9999,
                                    }
                                )

                    update_identity_gui = True

                    try:
                        last_deregister = ray.get(ct.last_deregister.get_all.remote(), timeout=RAY_TIMEOUT)
                    except GetTimeoutError:
                        update_identity_gui = False

                    if update_identity_gui:
                        for i in last_deregister.items():
                            last_objects_names = i[1].get("name")
                            progress = progresses.get(i[0])
                            if last_objects_names is not None and already_check[i[0]] is False:
                                already_check[i[0]] = True
                                ct.last_deregister.delete.remote(i[0])
                                try:
                                    last_objects_data = i[1]["img"].get()[0]
                                except IndexError:
                                    last_objects_data = image_error

                                if last_objects_names not in [False, "UNKNOWN??", ""] and \
                                        not last_objects_names.startswith("attacked:"):
                                    data = self.db.get_data(last_objects_names)
                                    if data is None:
                                        self.db.add_data(last_objects_names, *DataBase.default)
                                        data = self.db.get_data(last_objects_names)

                                    now_time = time.time()
                                    graph_info = data.get("graph_info") if data.get("graph_info") is not None else []
                                    graph_info.append(now_time)
                                    self.db.update(last_objects_names, last_checked=now_time, graph_info=graph_info)

                                self.change_infobox_message_signal.emit(
                                    {
                                        "name": last_objects_names,
                                        "ID": i[0],
                                        "image": last_objects_data,
                                        "last": True,
                                        "progress": 0.9999,
                                    }
                                )

                            elif already_check[i[0]] is False:
                                self.change_infobox_message_signal.emit(
                                    {
                                        "name": last_objects_names,
                                        "ID": i[0],
                                        "progress": progress,
                                    }
                                )

                for (objectID, centroid) in objects.items():

                    if not update_current_identity:
                        name = "IN_PROGRESS"
                    else:
                        name = (
                            general.Most_Common(names.get(objectID))
                            if type(names.get(objectID)) == list
                            else names.get(objectID)
                        )

                    if self.objname_map.get(objectID) is None and update_current_identity:
                        self.objname_map[objectID] = name

                    text = "ID [{}]".format(objectID)

                    general.putBorderText(
                        image,
                        text,
                        (centroid[0] / setting["resolution"] - 10, centroid[1] / setting["resolution"] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        (0, 0, 0),
                        2,
                        3,
                    )

                    if name not in ["IN_PROCESS",
                                    "UNKNOWN",
                                    "__UNKNOWN__",
                                    "CHECKED_UNKNOWN",
                                    "UNKNOWN??", ""] and not name.startswith("attacked:") and update_current_identity:

                        if self.db.quick_get_data(name).get("parent") is None:
                            self.db.update(name, parent=name)

                        name = self.db.quick_get_data(name).get("parent")

                    general.putBorderText(
                        image,
                        "IN_PROCESS" if name == "__UNKNOWN__" else name,
                        (centroid[0] / setting["resolution"] - 10, centroid[1] / setting["resolution"] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        (0, 0, 0),
                        2,
                        3,
                    )

                    cv2.circle(
                        image,
                        (int(centroid[0] / setting["resolution"]), int(centroid[1] / setting["resolution"])),
                        4,
                        text_color,
                        -1,
                    )

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                total_time = new_frame_time - prev_frame_time
                self.frame_index += 1

                if setting["fps_show"]:
                    if setting["average_fps"]:
                        if total_time < 100:
                            self.avg_fps.add(total_time)
                            total_time = self.avg_fps.get()

                    fps = int(1 / total_time) if total_time != 0 else -1

                    cv2.putText(
                        image,
                        str(fps),
                        (7, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (100, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                else:
                    general.putBorderText(
                        image,
                        datetime.now().strftime("%H:%M:%S"),
                        (7, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        Color.White,
                        Color.Black,
                        3,
                        4,
                        cv2.LINE_AA,
                    )

                prev_frame_time = new_frame_time

                self.change_pixmap_signal.emit(image)

        if setting.get("platform", "win") == "win":
            self.cap.release()
        elif setting.get("platform", "win") == "rpi":
            self.cap.close()


class App(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent=parent)
        self.spacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.last_progress_ = {}
        self.info_boxes = {}
        self.info_boxes_ID = []
        self.info_boxes_attacked = []
        self.id_navigation = {}
        self.db = DataBase("Students", sync_with_offline_db=True)
        self.db.offline_db_folder_path = setting["db_path"]
        self.storage = self.db.Storage(cache=setting.get("cache_path"))
        parent.setWindowTitle("GoodFaceRecognition")
        parent.resize(672, 316)
        parent.setStyleSheet("background-color: #0b1615;")

        self.centralwidget = QSplitter(Qt.Horizontal)

        self.image_label = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMaximumWidth(int(2 * parent.width() / 3))
        self.image_label.setMinimumWidth(int(480 / 4))

        self.image_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            "background: qlineargradient( x1:0 y1:0, x2:0 y2:1, stop:0 rgb(32, 45, 47), stop:.5 #3d5c57, stop:1 rgb("
            "32, 45, 47)); "
            "border-radius: 10px;"
        )
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.setting_button = general.PushButton()
        self.setting_button.setIcon(QIcon(setting["project_path"] + "/src/resources/setting.png"))
        self.setting_button.setIconSize(QSize(15, 15))
        self.setting_button.clicked.connect(self.handle_dialog)
        self.setting_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.face_data_manager_button = general.PushButton()
        self.face_data_manager_button.setIcon(QIcon(setting["project_path"] + "/src/resources/manager.png"))
        self.setting_button.setIconSize(QSize(15, 15))
        self.face_data_manager_button.clicked.connect(self.start_face_data_manager)
        self.face_data_manager_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        button_group_layout = QHBoxLayout()
        button_group_layout.addWidget(self.setting_button)
        button_group_layout.addWidget(self.face_data_manager_button)
        button_group = QWidget()
        button_group.setLayout(button_group_layout)
        button_group.setStyleSheet(
            "background: rgba(255, 255, 255, 0.1);"
            "border-radius: 10px;"
        )
        button_group_stretch_layout = QHBoxLayout()
        button_group_stretch_layout.addWidget(button_group)
        button_group_stretch_layout.addStretch()

        self.verticalLayout_1 = QVBoxLayout()
        self.verticalLayout_1.addWidget(self.image_label)
        self.verticalLayout_1.addLayout(button_group_stretch_layout)
        self.verticalLayout_1.addItem(QSpacerItem(10, 10, QSizePolicy.Preferred, QSizePolicy.Expanding))

        self.scrollArea = QScrollArea()
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setAcceptDrops(False)
        self.scrollArea.setAutoFillBackground(False)
        self.scrollArea.setStyleSheet("background-color: #142523;\n" "border-radius: 10px;")
        self.scrollArea.verticalScrollBar().setStyleSheet(scrollbar_style)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)

        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 563, 539))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.addStretch()
        self.verticalLayout.setDirection(QBoxLayout.BottomToTop)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.raise_()
        self.image_label.raise_()
        self.cameraVerticalLayout = QWidget()
        self.cameraVerticalLayout.setLayout(self.verticalLayout_1)
        self.cameraVerticalLayout.setMaximumWidth(int(2 * parent.width() / 3))

        self.centralwidget.addWidget(self.cameraVerticalLayout)
        self.centralwidget.addWidget(self.scrollArea)
        self.centralwidget.setSizes([parent.width() // 2, parent.width() // 2])

        parent.setCentralWidget(self.centralwidget)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_infobox_message_signal.connect(self.update_infobox)
        self.thread.start()

    @staticmethod
    def start_face_data_manager():
        print(f"open file \"{setting['project_path'] + '/FaceDataManager.py'}\"")
        Thread(target=os.system, args=(f"python {setting['project_path'] + '/FaceDataManager.py'}",)).start()

    def handle_dialog(self):
        dlg = ui_popupLite.Ui_Dialog(self, default_setting=setting, image=deepcopy(now_frame))

        if dlg.exec():
            print("Success!")
            if setting.get("video_change") is True:
                self.thread.release_cam()
                self.thread = VideoThread()
                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.change_infobox_message_signal.connect(self.update_infobox)
                self.thread.start()
                setting["video_change"] = None
        else:
            print("Cancel!")

    def info_box_popup(self, index: int, box: QLabel, cv_image: np.ndarray):
        dlg = ui_popup2.Ui_Dialog(self)

        avg_color = list(map(int, np.average(np.average(cv_image, axis=0), axis=0)))

        dlg.Image_box.setStyleSheet(
            "border-radius: 10px;\n"
            "border: 2px solid rgb(202, 229, 229);"
            f"background-color: rgb({avg_color[2]}, {avg_color[1]}, {avg_color[0]});\n"
        )

        data = box.text().lstrip("<font size=5><b>").rstrip("</font>").split("</b></font><br><font size=3>")
        data = [*data[0].split(": "), *data[1].split(" ")]
        name = data[0]
        dlg.name = name
        date_: str = data[2]
        time_: str = data[3]

        if self.info_boxes_ID[index] == ct.recognizer.unknown:
            return

        personal_data = Student().load_from_db(self.db, self.info_boxes_ID[index])
        dlg.lineEdit.setText(personal_data.realname)

        if name.startswith("attacked:"):
            return

        try:
            IDD: str = personal_data.IDD
        except ValueError:
            IDD = name
            ct.recognizer.name_map[IDD] = name

        dlg.Image_box.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    general.generate_profile(
                        IDD,
                        image_source=setting["project_path"] + "/src/resources/unknown_people.png",
                        font_path=setting["project_path"] + "/src/resources/Kanit-Medium.ttf",
                    ),
                    dlg.Image_box.size().width() - 10,
                    dlg.Image_box.size().height() - 10,
                ),
                10,
            )
        )

        Thread(target=lambda: self.__load_image_passive(imageBox=dlg.Image_box, ID=IDD)).start()

        if IDD in []:
            return

        if personal_data is not None:
            info_data_graph_info = personal_data.student_attendant_graph_data
            data_x, data_y = AttendantGraph(today=datetime.today()).load_datetimes(info_data_graph_info).data_in_week()
            dlg.plot_graph(data_x, data_y)
            raw_info_data_except_graph_info = {}
            key_queue = [
                Student.FIRSTNAME,
                Student.LASTNAME,
                Student.NICKNAME,
                Student.STUDENT_ID,
                Student.STUDENT_CLASS,
                Student.STUDENT_CLASS,
                Student.LAST_CHECKED,
            ]
            for key in key_queue:
                value = personal_data.to_dict()[key]
                if key != Student.STUDENT_ATTENDANT_GRAPH_DATA:
                    if key == Student.LAST_CHECKED:
                        value = datetime.fromtimestamp(value).strftime("%d %b %Y %X")
                    raw_info_data_except_graph_info[key] = value

            raw_info_data_except_graph_info["active_days"] = len(
                Arrange(AttendantGraph().load_datetimes(info_data_graph_info).dates).arrange_in_all_as_day()
            )
            print(raw_info_data_except_graph_info)
            dlg.add_data(raw_info_data_except_graph_info)

        dlg.ID.setText(IDD)
        dlg.exec()
        print("IDD:", IDD)

        if dlg.name != name:
            with open(setting["name_map_path"], "r") as file:
                raw_data = file.read()

            if raw_data:
                information = loads(raw_data)
            else:
                information = {}

            IDD_old = IDD
            if IDD.startswith("unknown:"):
                del information[IDD]
                del ct.recognizer.name_map[IDD]
                IDD = IDD.lstrip("unknown:")
                for index, load_id_id in enumerate(ct.recognizer.loaded_id):
                    if load_id_id == IDD_old:
                        ct.recognizer.loaded_id[index] = IDD

                processed_face = Recognition.ProcessedFace(
                    ct.faceRecPath + r"\unknown\{}.pkl".format(IDD))

                processed_face.IDD = IDD
                processed_face.filename = ct.faceRecPath + r"\known\{}.pkl".format(IDD)
                processed_face.save()
                os.remove(ct.faceRecPath + r"\unknown\{}.pkl".format(IDD))

                if self.db.get_data(IDD_old) is not None:
                    db_data = self.db.get_data(IDD_old)
                    self.db.delete(IDD_old)
                    self.db.add_data(
                        IDD,
                        realname=db_data.get("realname", ""),
                        surname=db_data.get("surname", ""),
                        nickname=db_data.get("nickname", ""),
                        student_id=db_data.get("student_id", 0),
                        student_class=db_data.get("student_class", ""),
                        class_number=db_data.get("class_number", 0),
                        active_days=db_data.get("active_days", 0),
                        last_checked=db_data.get("last_checked", 0),
                        graph_info=db_data.get("graph_info", []),
                        check_name=dlg.name,
                    )

            information[IDD] = dlg.name
            with open(ct.recognizer.name_map_path, "w") as file:
                dump_information = dumps(information)
                if dump_information:
                    file.write(dump_information)
                    ct.recognizer.name_map[IDD] = dlg.name

            for i in range(index + 1):
                if self.info_boxes_ID[i] == IDD or self.info_boxes_ID[i] == IDD_old:
                    textbox: QLabel = self.id_navigation[i]["message_box"]
                    textbox.setText(
                        f"<font size=5><b>{dlg.name}: {index}</b></font><br><font size=3>{date_} {time_}</font>"
                    )

    def new_info_box(self, message, cv_image, ID) -> (QLabel, QLabel, QHBoxLayout):
        _translate = QCoreApplication.translate
        horizontalLayout = QHBoxLayout(self.scrollAreaWidgetContents)
        box = QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(box.sizePolicy().hasHeightForWidth())
        box.setSizePolicy(sizePolicy)
        box.setMinimumSize(QSize(0, 80))
        box.setMaximumSize(QSize(16777215, 160))
        box.setFont(QFont(setting["font"]))
        box.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb(62, 83, 87), stop: 1 rgb(32, 45, 47));"
            "color: #8ba0a3;"
            "padding-left: 10px;"
            "border-radius: 10px;"
        )
        box.setText(_translate("MainWindow", message))
        box.setTextFormat(Qt.RichText)
        box.mousePressEvent = lambda _: self.info_box_popup(ID, box, deepcopy(cv_image))
        img_box = QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(img_box.sizePolicy().hasHeightForWidth())
        img_box.setSizePolicy(sizePolicy)
        img_box.setMinimumSize(QSize(80, 80))
        img_box.setMaximumSize(QSize(80, 80))
        img_box.setStyleSheet("background-color: #114f46;" "border-radius: 10px;" "border: 3px solid #0a402c;")
        if cv_image is None or not cv_image.any():
            cv_image = image_error
        img_box.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR),
                    img_box.size().width() - 10,
                    img_box.size().height() - 10,
                ),
                10,
            )
        )
        img_box.setAlignment(Qt.AlignCenter)
        horizontalLayout.addWidget(img_box)
        horizontalLayout.addWidget(box)

        return img_box, box, horizontalLayout

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        avg_color_top = list(map(int, np.average(np.average(cv_img[0: int(cv_img.shape[0] / 4)], axis=0), axis=0)))
        avg_color_bottom = list(
            map(
                int, np.average(np.average(cv_img[int(3 * cv_img.shape[0] / 4): int(cv_img.shape[0])], axis=0), axis=0)
            )
        )

        pixmap = general.convert_cv_qt(
            cv_img,
            self.image_label.size().width() - 10,
            self.image_label.size().height() - 10,
        )

        self.image_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            f"background: qlineargradient( x1:0 y1:0, x2:0 y2:1, "
            f"stop:0 rgb({avg_color_top[2]}, {avg_color_top[1]}, {avg_color_top[0]}),"
            f"stop:.5 rgb({avg_color_bottom[2]}, {avg_color_bottom[1]}, {avg_color_bottom[0]}), stop:.9 rgba( "
            "255, 255, 255, 0)); "
            "border-radius: 10px;"
        )

        rounded = general.round_Pixmap(pixmap, 10)
        self.image_label.setPixmap(rounded)

    def __load_image_passive(self, index: int = None, imageBox: QWidget = None, ID: str = None):
        if imageBox is None and index is not None:
            imageBox = self.id_navigation[index]["img_box"]
            load_image = self.storage.smart_get_image(self.info_boxes_ID[index])

        elif index is None:
            load_image = self.storage.smart_get_image(ID)

        else:
            load_image = unknown_image

        if not (load_image is not None and load_image is not False and load_image.any()):
            return

        pixmap = general.convert_cv_qt(
            load_image,
            imageBox.size().width() - 10,
            imageBox.size().height() - 10,
        )
        rounded = general.round_Pixmap(pixmap, 10)
        imageBox.setPixmap(rounded)

    def set_infobox_progress(self, progress, index, special_state=False, name=None):
        textBox: QLabel = self.id_navigation[index]["message_box"]
        imageBox: QLabel = self.id_navigation[index]["img_box"]

        def _animate(value):
            grad = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 #1b8c7b, stop: {value} #1bb77b, " \
                   f"stop: {value + 0.001} rgb(62, 83, 87), stop: 1 rgb(32, 45, 47)); padding-left: 10px;"
            textBox.setText(f"<font size=5><b>{round(value * 100)}</b></font>%")
            textBox.setStyleSheet(grad)

        def __animate(value):
            grad = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, " \
                   f"stop: 0 rgb{value.red(), value.green(), value.blue()}, " \
                   f"stop: 1 rgb{value.red(), value.green() - 34, value.blue()}); padding-left: 10px;"
            textBox.setStyleSheet(grad)

        if progress == self.last_progress_[index]:
            return
        if progress == 0.9999:  # the last update (update the name)
            animation = QVariantAnimation(self)
            animation.valueChanged.connect(_animate)
            animation.setStartValue(0.001 if self.last_progress_[index] == 0 else self.last_progress_[index])
            animation.setEndValue(0.999)
            animation.setDuration(500)
            if special_state:
                animation1 = QVariantAnimation(self)
                animation1.valueChanged.connect(__animate)
                animation1.setStartValue(QtGui.QColor(27, 183, 123))
                animation1.setEndValue(QtGui.QColor(183, 160, 27))
                animation1.setDuration(500)
                animation.finished.connect(lambda: animation1.start())
                if name is not None:
                    animation_imgbox = QVariantAnimation(self)
                    animation_imgbox.valueChanged.connect(
                        lambda value: imageBox.setStyleSheet(
                            f"border: 3px solid rgb({value.red()}, {value.green()}, {value.blue()});"
                        )
                    )
                    animation_imgbox.setStartValue(QtGui.QColor(imageBox.palette().window().color().rgb()))
                    animation_imgbox.setEndValue(QtGui.QColor(183, 160, 27))
                    animation_imgbox.setDuration(500)
                    animation_imgbox.start()

                    animation1.finished.connect(lambda: textBox.setText(name))
            else:
                if name is not None:
                    animation_imgbox = QVariantAnimation(self)
                    animation_imgbox.valueChanged.connect(
                        lambda value: imageBox.setStyleSheet(
                            f"border: 3px solid rgb({value.red()}, {value.green()}, {value.blue()});"
                        )
                    )
                    animation_imgbox.setStartValue(QtGui.QColor(imageBox.palette().window().color().rgb()))
                    animation_imgbox.setEndValue(QtGui.QColor(34, 212, 146))
                    animation_imgbox.setDuration(500)

                    if self.info_boxes_attacked[index]:
                        animation1 = QVariantAnimation(self)
                        animation1.valueChanged.connect(__animate)
                        animation1.setStartValue(QtGui.QColor(27, 183, 123))
                        animation1.setEndValue(QtGui.QColor(183, 84, 27))
                        animation1.setDuration(500)
                        animation_imgbox.setEndValue(QtGui.QColor(183, 84, 27))
                        animation1.finished.connect(lambda: animation_imgbox.start())
                        animation.finished.connect(lambda: animation1.start())
                        print("it run??")
                    else:
                        animation_imgbox.start()

                    # print("yes i am", progress, self.last_progress_[index], special_state)
                    animation.finished.connect(lambda: textBox.setText(name))

                    if len(self.info_boxes_ID) > index:
                        if self.info_boxes_ID[index] is not False:
                            Thread(target=lambda: self.__load_image_passive(index)).start()

            animation.start()

        elif progress < 0.9999:  # update status of people (not finished; update the progress bar)
            animation = QVariantAnimation(self)
            animation.valueChanged.connect(_animate)
            animation.setStartValue(min(self.last_progress_[index], progress))
            animation.setEndValue(max(self.last_progress_[index], progress))
            animation.setDuration(500)
            animation.start()
        else:
            textBox.setStyleSheet(
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb(62, 83, 87),"
                " stop: 1 rgb(32, 45, 47)); padding-left: 10px;"
            )
        self.last_progress_[index] = progress

    @pyqtSlot(dict)
    def update_infobox(self, data: dict):
        _translate = QCoreApplication.translate
        name = data.get("name")
        ID = data.get("ID")
        image = data.get("image")
        last = data.get("last")
        progress = data.get("progress")
        progress = 0 if progress is None else progress
        state = False

        if self.info_boxes.get(ID) is None:
            self.info_boxes[ID] = True
            self.verticalLayout.removeItem(self.spacer)
            img_box, message_box, layout = self.new_info_box(f"<font size=5><b>Finding face..</b></font>", image, ID)
            self.id_navigation[ID] = {"img_box": img_box, "message_box": message_box}
            self.verticalLayout.addLayout(layout)

        else:
            if name is None:
                self.info_boxes_attacked.append(False)
            elif name is not False and name.startswith("attacked:"):
                self.info_boxes_attacked.append(True)
            else:
                self.info_boxes_attacked.append(False)

            if name == "IN_PROCESS" or name == "__UNKNOWN__":
                message = None
            elif (last is True and name is False) or name == "":
                message = f'<font size=5><b>FAILED: {ID}</b></font><br><font size=3>"' \
                          f'{time.strftime("%D/%M %H:%M:%S", time.localtime())}</font>'
                state = True
                self.info_boxes_ID.append(False)
            elif name is None:
                message = f"<font size=5>...</font>"
            elif name == ct.recognizer.unknown:
                self.info_boxes_ID.append(ct.recognizer.unknown)
                message = f'<font size=5><b>UNKNOWN: {ID}</b></font><br><font size=3>"' \
                          f'{time.strftime("%D/%M %H:%M:%S", time.localtime())}</font>'
            else:
                if name not in ["IN_PROCESS",
                                "UNKNOWN",
                                "__UNKNOWN__",
                                "CHECKED_UNKNOWN",
                                "UNKNOWN??",
                                ""] and not name.startswith("attacked:"):

                    if self.db.get_data(name).get("parent") is None:
                        self.db.update(name, parent=name)

                    name = self.db.get_data(name)["parent"]

                mapped_name = ct.recognizer.name_map.get(name)
                self.info_boxes_ID.append(name)
                if mapped_name is None:
                    mapped_name = name

                if len(mapped_name) > 20:
                    mapped_name = mapped_name[:20] + "..."

                message = f"<font size=5><b>{mapped_name}: {ID}</b></font><br><font size=3>" \
                          f"{time.strftime('%D/%M %H:%M:%S', time.localtime())}</font>"

            if self.last_progress_.get(ID) is None:
                self.last_progress_[ID] = 0.001

            if len(self.scrollAreaWidgetContents.children()) <= ID:
                return

            imgbox: QLabel = self.id_navigation[ID]["img_box"]

            if image is not None and image.any():
                pixmap = general.convert_cv_qt(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    imgbox.size().width() - 10,
                    imgbox.size().height() - 10,
                )
                rounded = general.round_Pixmap(pixmap, 10)
                imgbox.setPixmap(rounded)

            self.set_infobox_progress(progress, ID, special_state=state, name=message)

            if last is True:
                del self.last_progress_[ID]


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    a = App(MainWindow)
    MainWindow.show()

    RepeatedTimer(60 * 10, lambda: ct.recognizer.update(a.storage))
    sys.exit(app.exec_())
