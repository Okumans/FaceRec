from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QScrollArea, \
    QSpacerItem, QMenu, QMenuBar, QStatusBar, QMainWindow
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QSize, QRect, QCoreApplication
import numpy as np
import time
import urllib
from matplotlib.pyplot import gray

import Logger
from ShadowRemoval import remove_shadow_grey
from centroidtracker import CentroidTracker
import numpy as np
import time
import general
from general import log, Color, get_from_percent
from copy import deepcopy
import cv2
from FaceAlignment import face_alignment
import ray
import mediapipe as mp

# -----------------setting-----------------
video_source = 'http://192.168.1.44:8080/video'
min_detection_confidence = .75
min_faceBlur_detection = 24  # low = blur, high = not blur
autoBrightnessContrast = True
autoBrightnessValue = 80  # from 0 - 255
autoContrastValue = 30  # from 0 - 255
face_check_amount = 3
face_max_disappeared = 20
night_mode_brightness = 40
sharpness_filter = False
gray_mode = False
debug = False
fps_show = False
cpu_amount = 8
face_reg_path = r"C:\general\Science_project\Science_project_cp39\\resources"

# -------------global variable--------------
ray.init(num_cpus=cpu_amount)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
ct = CentroidTracker(face_reg_path, maxDisappeared=face_max_disappeared, minFaceBlur=min_faceBlur_detection,
                     minFaceConfidence=min_detection_confidence, faceCheckAmount=face_check_amount)
logger = Logger.Logger()
(H, W) = (None, None)
text_color = (0, 255, 255)
prev_frame_time = 0  # fps counter
new_frame_time = 0
last_id = -1
already_check = {}
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])


class info:
    def __init__(self, string, image):
        self.string = string
        self.image = image


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_infobox_message_signal = pyqtSignal(info)

    def run(self):
        global H, W, prev_frame_time, new_frame_time, last_id
        cap = cv2.VideoCapture(video_source)
        face_mesh = mp_face_mesh.FaceMesh()
        with mp_face_detection.FaceDetection(min_detection_confidence=0.55, model_selection=1) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                new_frame_time = time.time()

                if not success:
                    logger.log("Ignoring empty camera frame.")
                    continue

                if H is None or W is None:
                    (H, W) = image.shape[:2]

                image.flags.writeable = False
                image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
                if general.brightness(image) < night_mode_brightness or gray_mode:
                    image = cv2.cvtColor(remove_shadow_grey(image), cv2.COLOR_GRAY2RGB)  # for night vision
                    general.putBorderText(image, "NIGHT MODE",
                                          (W - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Color.Violet, Color.Black, 2, 3)
                if sharpness_filter: image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                rects = []
                if results.detections:
                    for detection in results.detections:
                        x_min = detection.location_data.relative_bounding_box.xmin * W
                        y_min = detection.location_data.relative_bounding_box.ymin * H
                        x_max = x_min + detection.location_data.relative_bounding_box.width * W
                        y_max = y_min + detection.location_data.relative_bounding_box.height * H
                        face_height = y_max - y_min
                        box = (x_min, y_min, x_max, y_max)
                        face_image = face_alignment(deepcopy(image[int(box[1]) - get_from_percent(face_height, 20):
                                                                   int(box[3]) + get_from_percent(face_height, 20),
                                                             int(box[0]) - get_from_percent(face_height, 20):
                                                             int(box[2]) + get_from_percent(face_height, 20)]),
                                                    face_mesh)
                        #  face_image = deepcopy(image[int(box[1]) - get_from_percent(face_height, 20):int(box[3]) + get_from_percent(face_height, 20), int(box[0]) - get_from_percent(face_height, 20):int(box[2]) + get_from_percent(face_height, 20)])
                        rects.append({box: (detection.score[0], face_image)})

                        if autoBrightnessContrast:
                            face_image = general.change_brightness_to(face_image, autoBrightnessValue)
                            face_image = general.change_contrast_to(face_image, autoContrastValue)
                        # cv2.imshow("test", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

                        general.putBorderText(image,
                                              f"confident: {round(detection.score[0], 2)}% blur {CentroidTracker.is_blur(face_image, min_faceBlur_detection)} ",
                                              (int(box[0]), int(box[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                              (255, 0, 0),
                                              (0, 0, 0), 2, 3)
                        if debug:
                            general.putBorderText(image,
                                                  f"brightness: {round(general.brightness(face_image), 2)} contrast: {round(general.contrast(face_image), 2)}",
                                                  (int(box[0]), int(box[1]) + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                  (255, 0, 0),
                                                  (0, 0, 0), 2, 3)
                        # f"brightness: {round(general.brightness(face_image), 2)} contrast: {round(general.contrast(face_image), 2)}"
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 0), 5)
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), text_color, 3)

                objects = ct.update(rects)

                temp = last_id
                if [i[0] for i in objects.items()]:
                    last_id = max([i[0] for i in objects.items()])
                for i in [i[0] for i in objects.items()]:
                    if i > temp:
                        already_check[i] = False
                        print(f"update id {i - 1} -> {i}")
                    else:
                        name = ray.get(ct.objects_names.get.remote(i))
                        if name not in ["UNKNOWN", "IN_PROCESS", "CHECKED_UNKNOWN", None, "False"] and already_check[i] is False:
                            already_check[i] = True
                            print(f"update id {i - 1} -> {i}: '{name}'")
                            self.change_infobox_message_signal.emit(
                                info(f"<font size=8><b>{name}</b></font><br><font size=4>{time.strftime('%D/%M %H:%M:%S', time.localtime())}</font>",
                                     ct.objects_data[i].get()[0]))

                for i in ray.get(ct.last_deregister.get_all.remote()).items():
                    last_objects_names = i[1].get("name")
                    if last_objects_names is not None and already_check[i[0]] is False:
                        already_check[i[0]] = True
                        last_objects_data = i[1]["img"].get()[0]
                        self.change_infobox_message_signal.emit(
                            info(f"<font size=8><b>{last_objects_names}</b></font><br><font size=4>{time.strftime('%D/%M %H:%M:%S', time.localtime())}</font>", last_objects_data))

                for (objectID, centroid) in objects.items():
                    text = "ID [{}]".format(objectID)
                    # noinspection PyUnresolvedReferences
                    general.putBorderText(image, text, (centroid[0] - 10, centroid[1] - 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, (0, 0, 0), 2, 3)
                    general.putBorderText(image,
                                          general.Most_Common(ray.get(ct.objects_names.get.remote(objectID))) if type(
                                              ray.get(ct.objects_names.get.remote(objectID))) == list else ray.get(
                                              ct.objects_names.get.remote(objectID)),
                                          (centroid[0] - 10, centroid[1] - 40),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, (0, 0, 0), 2, 3)

                    cv2.circle(image, (centroid[0], centroid[1]), 4, text_color, -1)

                # Flip the image horizontally for a selfie-view display.
                total_time = new_frame_time - prev_frame_time
                fps = int(1 / total_time) if total_time != 0 else -1
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if fps_show:
                    cv2.putText(image, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
                prev_frame_time = new_frame_time
                self.change_pixmap_signal.emit(image)
        cap.release()


class App(QWidget):
    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        MainWindow.setWindowTitle("Qt live label demo")
        MainWindow.resize(1336, 553)
        MainWindow.setStyleSheet("background-color: rgb(53, 74, 76);")

        self.centralwidget = QWidget(MainWindow)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.centralwidget.setObjectName("centralwidget")

        self.image_label = QLabel(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMinimumSize(QSize(640, 480))
        self.image_label.setMaximumSize(QSize(640 * 2, 480 * 2))
        self.image_label.setStyleSheet("color: rgb(240, 240, 240);\n"
                                       "padding-top: 15px;\n"
                                       "background-color:rgba(32, 45, 47, 200);\n"
                                       "border-radius: 10px;")
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.verticalLayout_1 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_1.addWidget(self.image_label)
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
        self.scrollArea.setStyleSheet("background-color: rgba(239, 239, 239, 30);\n"
                                      "border-radius: 10px;")
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)

        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 563, 539))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.spacerItem = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.verticalLayout.addItem(self.spacerItem)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.scrollArea.raise_()
        self.image_label.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 1151, 22))
        self.menuSetting = QMenu(self.menubar)
        font = QtGui.QFont()
        font.setFamily("Prompt")
        self.menuSetting.setFont(font)
        self.menuSetting.setObjectName("menuSetting")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuSetting.menuAction())

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_infobox_message_signal.connect(self.test)
        # start the thread
        self.thread.start()

    def new_info_box(self, message, cv_image):
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
        box.setStyleSheet("background-color: rgba(138, 209, 235, 120);\n"
                          "color: rgb(54, 54, 54);\n"
                          "padding-left: 10px;\n"
                          "font: bold \"Roboto\";\n"
                          "border-radius: 10px;")
        box.setText(_translate("MainWindow", message))
        box.setTextFormat(Qt.RichText)
        img_box = QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(img_box.sizePolicy().hasHeightForWidth())
        img_box.setSizePolicy(sizePolicy)
        img_box.setMinimumSize(QSize(160, 160))
        img_box.setMaximumSize(QSize(160, 160))
        img_box.setStyleSheet("background-color: rgba(178, 249, 255, 120);\n"
                              "border-radius: 10px;")
        img_box.setPixmap(self.convert_cv_qt(cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR), img_box.size().width()-20, img_box.size().height()-20))
        img_box.setAlignment(Qt.AlignCenter)
        horizontalLayout.addWidget(img_box)
        horizontalLayout.addWidget(box)
        return horizontalLayout

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, self.image_label.size().width() - 30, self.image_label.size().height() - 30)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(info)
    def test(self, data: info):
        message = data.string
        image = data.image
        self.verticalLayout.removeItem(self.spacerItem)
        self.verticalLayout.addLayout(self.new_info_box(message, image))
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())
        self.verticalLayout.addItem(self.spacerItem)

    @staticmethod
    def convert_cv_qt(cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    a = App()
    a.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
