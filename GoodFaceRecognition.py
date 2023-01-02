import cProfile
import sys
import numpy as np
import time
import triple_gems
from copy import deepcopy
from shutil import move
from datetime import datetime
from pyautogui import size
from os.path import exists
from os import mkdir
from PyQt5 import QtGui
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
)
from PyQt5.QtGui import QPixmap, QFont, QIcon
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
    QPropertyAnimation,
)
from scrollbar_style import scrollbar_style
import Logger
import cv2
import ray
import mediapipe as mp
from mss import mss
import ui_popup  # for setting popup
import ui_popup2  # for infobox popup
import general
from general import Color, get_from_percent
from ShadowRemoval import remove_shadow_grey
from centroidtracker import CentroidTracker
from FaceAlignment import face_alignment
from json import dumps, loads

# -----------------setting-----------------
setting = dict()
setting["video_source"] = 0
setting["min_detection_confidence"] = 0.7
setting["min_recognition_confidence"] = 0.55
setting["min_faceBlur_detection"] = 24  # low = rgb(175, 0, 0)blur, high = not blur
setting["autoBrightnessContrast"] = False
setting["autoBrightnessValue"] = 80  # from 0 to 255
setting["autoContrastValue"] = 30  # from 0 to 255
setting["face_check_amount"] = 3
setting["face_max_disappeared"] = 10
setting["night_mode_brightness"] = 40
setting["sharpness_filter"] = False
setting["gray_mode"] = False
setting["debug"] = False
setting["fps_show"] = False
setting["average_fps"] = True
setting["cpu_amount"] = 16
setting["resolution"] = 1
setting["base_resolution"] = (0, 0)
setting["remember_unknown_face"] = True
setting["face_reg_path"] = r"C:\general\Science_project\Science_project_cp39\resources"
setting["name_map_path"] = r"C:\general\Science_project\Science_project_cp39\resources\name_information.json"
setting["font"] = "Kanit"
setting["face_alignment"] = True
use_folder = [setting["face_reg_path"] + r"\unknown", setting["face_reg_path"] + r"\known"]

# -------------global variable--------------
if __name__ == "__main__":
    # check if using folder is available
    for folder_path in use_folder:
        if not exists(folder_path):
            mkdir(folder_path)

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
    logger = Logger.Logger()
    (H, W) = (None, None)
    text_color = (0, 255, 255)
    prev_frame_time = 0  # fps counter
    new_frame_time = 0
    last_id = -1
    now_frame = 0  # cv2 mat
    already_check = {}
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image_error = cv2.imread("image_error.png")


class VideoThread(QThread):
    change_pixmap_signal: pyqtBoundSignal = pyqtSignal(np.ndarray)
    change_infobox_message_signal: pyqtBoundSignal = pyqtSignal(dict)

    def __init__(self):
        global H, W
        super().__init__()
        self.video_type = setting["video_source"]
        self.run = True
        self.avg_fps = general.Average([0], calculate_amount=100)
        if self.video_type == "screen":
            self.sct = mss()
            self.screen_size = size()
        else:
            self.cap = cv2.VideoCapture(self.video_type)
        (H, W) = (None, None)

    def release_cam(self):
        if self.video_type != "screen":
            self.cap.release()
        self.run = False

    def run(self):
        global H, W, prev_frame_time, new_frame_time, last_id, now_frame
        with mp_face_detection.FaceDetection(min_detection_confidence=0.75, model_selection=1) as face_detection:
            while self.run or self.cap.isOpened():
                if self.video_type == "screen":
                    monitor = {
                        "top": 40,
                        "left": 0,
                        "width": self.screen_size[0],
                        "height": self.screen_size[1],
                    }
                    image = np.array(self.sct.grab(monitor))
                    success = True
                else:
                    success, image = self.cap.read()

                new_frame_time = time.time()

                image = cv2.resize(image, (0, 0), fx=setting["resolution"], fy=setting["resolution"])
                ct.maxDisappeared = setting["face_max_disappeared"]
                ct.minFaceBlur = setting["min_faceBlur_detection"]
                ct.minFaceConfidence = setting["min_detection_confidence"]
                ct.faceCheckAmount = setting["face_check_amount"]
                ct.recognizer.min_confidence = setting["min_recognition_confidence"]
                ct.recognizer.remember = setting["remember_unknown_face"]

                if not success or image is None:
                    logger.log(f"Ignoring empty camera frame.")
                    continue

                if H is None or W is None:
                    (H, W) = image.shape[:2]
                    setting["base_resolution"] = (W, H)

                now_frame = image
                image.flags.writeable = False
                image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)

                if setting["autoBrightnessContrast"]:
                    image = general.change_brightness_to(image, setting["autoBrightnessValue"])
                    image = general.change_contrast_to(image, setting["autoContrastValue"])

                if setting["sharpness_filter"]:
                    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

                if setting["gray_mode"]:
                    image = cv2.cvtColor(remove_shadow_grey(image), cv2.COLOR_GRAY2RGB)  # for night vision
                    general.putBorderText(
                        image,
                        "NIGHT MODE",
                        (W - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Color.Violet,
                        Color.Black,
                        2,
                        3,
                    )

                results = face_detection.process(image)

                image.flags.writeable = True
                image_use = deepcopy(image)
                image = cv2.resize(image, (W, H))
                rects = []

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
                            int(box[1])
                            - get_from_percent(face_height, 20): int(box[3])
                                                                 + get_from_percent(face_height, 20),
                            int(box[0])
                            - get_from_percent(face_height, 20): int(box[2])
                                                                 + get_from_percent(face_height, 20),
                            ]
                        )

                        if setting["face_alignment"]:
                            face_image = face_alignment(face_image, detection)
                        rects.append({box: (detection.score[0], face_image)})

                        # face_image = deepcopy(image[int(box[1]) - get_from_percent(face_height, 20):int(box[3]) +
                        # get_from_percent(face_height, 20), int(box[0]) - get_from_percent(face_height, 20):int(box[
                        # 2]) + get_from_percent(face_height, 20)])
                        # detected_emotions = emotion_recognizer.detect_emotions(face_image) print("emotion:",
                        # sorted(detected_emotions[0]["emotions"].items(), key=lambda x: x[1], reverse=True)[0] if
                        # detected_emotions else "None") cv2.imshow("test", cv2.cvtColor(face_image,
                        # cv2.COLOR_RGB2BGR))

                        if setting["debug"]:
                            general.putBorderText(
                                image,
                                f"confident: {round(detection.score[0], 2)}% blur {CentroidTracker.is_blur(face_image, setting['min_faceBlur_detection'])} ",
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
                                f"brightness: {round(general.brightness(face_image), 2)} contrast: {round(general.contrast(face_image), 2)}",
                                (int(box[0]), int(box[1]) + 38),
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
                            text_color,
                            3,
                        )

                objects = ct.update(rects)

                temp = last_id
                objects_ids = [i[0] for i in objects.items()]
                if objects_ids:
                    last_id = max(objects_ids)

                names = ray.get(ct.objects_names.get_all.remote())
                progresses = ray.get(ct.recognition_progress.get_all.remote())

                for i in objects_ids:
                    name = names.get(i)
                    progress = progresses.get(i)
                    if i > temp:
                        already_check[i] = False
                    elif name == "IN_PROCESS":
                        self.change_infobox_message_signal.emit(
                            {
                                "name": name,
                                "ID": i,
                                "image": ct.objects_data[i].get()[0],
                                "progress": progress,
                            }
                        )
                    else:
                        if (
                                name
                                not in [
                            "UNKNOWN",
                            "CHECKED_UNKNOWN",
                            "__UNKNOWN__",
                            None,
                            False,
                        ]
                                and already_check[i] is False
                        ):
                            already_check[i] = True
                            ct.last_deregister.delete.remote(i)
                            self.change_infobox_message_signal.emit(
                                {
                                    "name": name,
                                    "ID": i,
                                    "image": ct.objects_data[i].get()[0],
                                    "progress": 0.9999,
                                }
                            )

                for i in ray.get(ct.last_deregister.get_all.remote()).items():
                    last_objects_names = i[1].get("name")
                    progress = progresses.get(i[0])
                    if last_objects_names is not None and already_check[i[0]] is False:
                        already_check[i[0]] = True
                        ct.last_deregister.delete.remote(i[0])
                        try:
                            last_objects_data = i[1]["img"].get()[0]
                        except IndexError:
                            last_objects_data = image_error
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
                    text = "ID [{}]".format(objectID)
                    name = (
                        general.Most_Common(names.get(objectID))
                        if type(names.get(objectID)) == list
                        else names.get(objectID)
                    )
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
                    """ 
                    cv2.arrowedLine(
                        image,
                        ct.objects_positions.data[objectID][0],
                        ct.objects_positions.data[objectID][-1],
                        Color.Violet,
                        6,
                        tipLength=0.5,
                    )
                    """

                total_time = new_frame_time - prev_frame_time

                if setting["average_fps"]:
                    if total_time < 100:
                        self.avg_fps.add(total_time)
                        total_time = self.avg_fps.get()

                fps = int(1 / total_time) if total_time != 0 else -1
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if setting["fps_show"]:
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

        if self.video_type != "screen":
            self.cap.release()


class App(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent=parent)
        self.spacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.last_progress_ = {}
        self.info_boxes = {}
        parent.setWindowTitle("Qt live label demo")
        parent.resize(1336, 553)
        parent.setStyleSheet("background-color: #0b1615;")

        self.centralwidget = QWidget(parent)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)

        self.image_label = QLabel(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMinimumSize(QSize(640, 480))
        self.image_label.setMaximumSize(QSize(640 * 2, 480 * 2))
        self.image_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            "background: qlineargradient( x1:0 y1:0, x2:0 y2:1, stop:0 rgb(32, 45, 47), stop:.5 #3d5c57, stop:1 rgb("
            "32, 45, 47)); "
            "border-radius: 10px;"
        )
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setting_button = general.PushButton(self.centralwidget)
        self.setting_button.setIcon(QIcon("setting.png"))
        self.setting_button.setIconSize(QSize(30, 30))
        self.setting_button.setMaximumSize(QSize(60, 99999999))
        # self.setting_button.setFixedSize(QSize(100, 100))

        self.setting_button.clicked.connect(self.handle_dialog)
        self.setting_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.verticalLayout_1 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_1.addWidget(self.image_label)
        self.verticalLayout_1.addWidget(self.setting_button)
        self.verticalLayout_1.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.horizontalLayout.addLayout(self.verticalLayout_1)

        self.scrollArea = QScrollArea(self.centralwidget)
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
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.scrollArea.raise_()
        self.image_label.raise_()
        parent.setCentralWidget(self.centralwidget)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_infobox_message_signal.connect(self.update_infobox)
        self.thread.start()

    def handle_dialog(self):
        dlg = ui_popup.Ui_Dialog(self, default_setting=setting, image=deepcopy(now_frame))

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

    def info_box_popup(self, ID: int, box: QLabel, cv_image: np.ndarray):
        dlg = ui_popup2.Ui_Dialog(self)
        dlg.Image_box.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR),
                    dlg.Image_box.size().width() - 10,
                    dlg.Image_box.size().height() - 10,
                ),
                10,
            )
        )

        avg_color = list(map(int, np.average(np.average(cv_image, axis=0), axis=0)))
        dlg.Image_box.setStyleSheet(
            "border-radius: 10px;\n"
            "border: 2px solid rgb(202, 229, 229);"
            f"background-color: rgb({avg_color[2]}, {avg_color[1]}, {avg_color[0]});\n"
        )

        data = box.text().lstrip("<font size=8><b>").rstrip("</font>").split("</b></font><br><font size=4>")
        data = [*data[0].split(": "), *data[1].split(" ")]
        name = data[0]
        dlg.name = name
        dat: str = data[2]
        time_: str = data[3]
        dlg.lineEdit.setText(name)

        try:
            IDD: str = list(ct.recognizer.name_map.keys())[list(ct.recognizer.name_map.values()).index(name)]
        except ValueError:
            IDD = name
            ct.recognizer.name_map[IDD] = name

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

            if IDD.startswith("unknown:"):
                del information[IDD]
                del ct.recognizer.name_map[IDD]
                IDD_old = IDD
                IDD = IDD.lstrip("unknown:")
                for index, load_id_id in enumerate(ct.recognizer.loaded_id):
                    if load_id_id == IDD_old:
                        ct.recognizer.loaded_id[index] = IDD
                move(ct.faceRecPath + r"\unknown\{}.pkl".format(IDD), ct.faceRecPath + r"\known\{}.pkl".format(IDD))

            information[IDD] = dlg.name
            with open(ct.recognizer.name_map_path, "w") as file:
                dump_information = dumps(information)
                if dump_information:
                    file.write(dump_information)
                    ct.recognizer.name_map[IDD] = dlg.name

    def new_info_box(self, message, cv_image, ID):
        _translate = QCoreApplication.translate
        horizontalLayout = QHBoxLayout(self.scrollAreaWidgetContents)
        box = QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(box.sizePolicy().hasHeightForWidth())
        box.setSizePolicy(sizePolicy)
        box.setMinimumSize(QSize(0, 160))
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
        img_box.setMinimumSize(QSize(160, 160))
        img_box.setMaximumSize(QSize(160, 160))
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

        return horizontalLayout

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
            f"background: qlineargradient( x1:0 y1:0, x2:0 y2:1, stop:0 rgb({avg_color_top[0]}, {avg_color_top[1]}, {avg_color_top[2]}),"
            f"stop:.5 rgb({avg_color_bottom[0]}, {avg_color_bottom[1]}, {avg_color_bottom[2]}), stop:.9 rgba( "
            "255, 255, 255, 0)); "
            "border-radius: 10px;"
        )

        rounded = general.round_Pixmap(pixmap, 10)
        self.image_label.setPixmap(rounded)

    def set_infobox_progress(self, progress, index, special_state=False, name=None):
        textBox: QLabel = self.scrollAreaWidgetContents.children()[index]
        imageBox: QLabel = self.scrollAreaWidgetContents.children()[index + 1]

        def _animate(value):
            grad = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 #1b8c7b, stop: {value} #1bb77b, stop: {value + 0.001} rgb(62, 83, 87), stop: 1 rgb(32, 45, 47)); padding-left: 10px;"
            textBox.setText(f"<font size=8><b>{round(value * 100)}</b></font>%")
            textBox.setStyleSheet(grad)

        def __animate(value):
            grad = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb{value.red(), value.green(), value.blue()}, stop: 1 rgb{value.red(), value.green() - 34, value.blue()}); padding-left: 10px;"
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
                    animation_imgbox.start()
                    # print("yes i am", progress, self.last_progress_[index], special_state)
                    animation.finished.connect(lambda: textBox.setText(name))
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
                f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb(62, 83, 87), stop: 1 rgb(32, 45, 47)); padding-left: 10px;"
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
            self.verticalLayout.addLayout(self.new_info_box(f"<font size=8><b>ค้นหาใบหน้า</b></font>", image, ID))
            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
            self.verticalLayout.addItem(self.spacer)

        else:
            if name == "IN_PROCESS" or name == "__UNKNOWN__":
                message = None
            elif last is True and name is False:
                message = f'<font size=8><b>FAILED: {ID}</b></font><br><font size=4>{time.strftime("%D/%M %H:%M:%S", time.localtime())}</font>'
                state = True
            elif name is None:
                message = f"<font size=8>...</font>"
            else:
                mapped_name = ct.recognizer.name_map.get(name)
                if mapped_name is None:
                    mapped_name = name

                message = f"<font size=8><b>{mapped_name}: {ID}</b></font><br><font size=4>{time.strftime('%D/%M %H:%M:%S', time.localtime())}</font>"

            if self.last_progress_.get(((ID + 1) * 2) - 1) is None:
                self.last_progress_[((ID + 1) * 2) - 1] = 0.001

            if len(self.scrollAreaWidgetContents.children()) <= ((ID + 1) * 2):
                # print(
                #     len(self.scrollAreaWidgetContents.children()),
                #     ((ID + 1) * 2),
                #     ((ID + 1) * 2) - 1,
                # )
                return

            self.set_infobox_progress(progress, ((ID + 1) * 2) - 1, special_state=state, name=message)
            textbox: QLabel = self.scrollAreaWidgetContents.children()[((ID + 1) * 2) - 1]
            imgbox: QLabel = self.scrollAreaWidgetContents.children()[((ID + 1) * 2)]

            if image is not None and image.any():
                pixmap = general.convert_cv_qt(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    imgbox.size().width() - 10,
                    imgbox.size().height() - 10,
                )
                rounded = general.round_Pixmap(pixmap, 10)
                imgbox.setPixmap(rounded)

            self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

            if last is True:
                del self.last_progress_[((ID + 1) * 2) - 1]


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    a = App(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())