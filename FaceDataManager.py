import datetime
from pprint import pprint

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
    QFileDialog,
    QMainWindow,
    QLineEdit,
    QInputDialog,
    QBoxLayout,
    QMessageBox
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
import sys
import os.path as path
from os import remove
import general
import pickle
import cv2
from copy import deepcopy
import numpy as np
from attendant_graph import AttendantGraph, Arrange
from DataBase import DataBase
from shutil import move
from uuid import uuid4
from threading import Thread
import pickle


class App(QMainWindow):
    def resizeEvent(self, event):
        self.add_button.move(self.size().width() - 120, self.size().height() - 120)
        QMainWindow.resizeEvent(self, event)

    def __init__(self, target_directory: str):
        super().__init__()
        self.target_directory = target_directory
        self.db = DataBase("Students")
        self.db.offline_db_folder_path = self.target_directory
        self.result = {}
        self.current: str = ""  # current person data (ID)
        self.current_filename: str = ""  # current filename for image (ndarray)
        self.id_navigation: dict = {}
        self.face_data_info_loaded: dict = {}
        self.created_face_not_saved: dict = {}

        self.setWindowTitle("Qt live label demo")
        self.resize(1336, 553)
        self.setStyleSheet("background-color: #0b1615; color: white;")

        self.spacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.centralwidget = QWidget(self)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)

        self.id_information_label = QLabel(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.id_information_label.sizePolicy().hasHeightForWidth())
        self.id_information_label.setSizePolicy(sizePolicy)
        self.id_information_label.setMinimumSize(QSize(640, 660))
        self.id_information_label.setMaximumSize(QSize(640, 660))
        self.id_information_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            "background: qlineargradient( x1:0 y1:0, x2:0 y2:1, stop:0 rgb(32, 45, 47), stop:.5 #3d5c57, stop:1 rgb("
            "32, 45, 47)); "
            "border-radius: 10px;"
        )

        outer_layout = QHBoxLayout()
        layout_left = QVBoxLayout()
        layout_right = QVBoxLayout()

        self.name_line_edit = QLineEdit()
        self.name_line_edit.setPlaceholderText("ชื่อ-นามสกุล")
        element_font = QtGui.QFont()
        element_font.setFamily("Kanit")
        element_font.setPointSize(36)
        element_font.setBold(True)
        element_font.setWeight(75)
        self.name_line_edit.setFont(element_font)
        self.name_line_edit.setStyleSheet("color: #9eb5b3;\n" "background: transparent;\n" "border: 0;\n" "")
        layout_left.addWidget(self.name_line_edit)

        self.id_label = QLabel()
        element_font = QtGui.QFont()
        element_font.setFamily("Kanit")
        element_font.setPointSize(14)
        element_font.setBold(True)
        self.id_label.setFont(element_font)
        self.id_label.setStyleSheet(
            "background-color: #142523;\n" "border-radius: 10px;\n" "border-bottom: 2px solid #305752;color:#9eb5b3;"
            "padding-bottom: 5px; color: rgba(158, 181, 179, 80)"
        )
        self.id_label.mousePressEvent = self.add_data_to_id
        layout_left.addWidget(self.id_label)

        self.data_label = QLabel()
        self.data_label.setMinimumHeight(480)
        self.data_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            "background: qlineargradient( x1:0 y1:0, x2:0 y2:1, stop:0 #0e1a18, stop:.5 #142523, stop:1 #0e1a18);"
            "border-radius: 10px;"
        )
        self.data_label_layout = QVBoxLayout(self.data_label)

        layout_left.addWidget(self.data_label)
        layout_left.addSpacerItem(self.spacer)
        outer_layout.addLayout(layout_left)

        self.image_label = QLabel()
        self.image_label.setMinimumWidth(200)
        self.image_label.setMinimumHeight(240)
        self.image_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            "background: qlineargradient( x1:0 y1:0, x2:1 y2:1, stop:0 #1c2e2c, stop:.5 #2b4240, stop:1 #1c2e2c);"
            "border-radius: 10px;"
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_select

        layout_right.addWidget(self.image_label)
        layout_right.addSpacerItem(self.spacer)

        outer_layout.addLayout(layout_right)

        self.id_information_label.setLayout(outer_layout)
        self.id_information_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self.verticalLayout_1 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_1.addWidget(self.id_information_label)
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
        self.verticalLayout.addStretch()
        self.verticalLayout.setDirection(QBoxLayout.BottomToTop)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.scrollArea.raise_()
        self.id_information_label.raise_()
        self.setCentralWidget(self.centralwidget)
        self.init_face_ids()
        self.add_input("ชื่อเล่น", "nickname")
        self.add_input("รหัสประจำตัว", "student_id")
        self.add_input("ห้อง", "student_class")
        self.add_input("เลขที่", "class_number")
        self.add_input("จำนวนวันที่มาโรงเรียน", "active_days", enable=False)
        self.add_input("เช็คชื่อครั้งล่าสุด", "last_checked", enable=False)
        self.data_label_layout.addStretch()
        self.save_button = general.PushButton(style_sheet="""background-color: %s;
                                                       border: none;
                                                       border-radius: 5px;
                                                       color: %s;
                                                       padding: 16px 32px;
                                                       text-align: center;
                                                       text-decoration: none;
                                                       font: bold \"Kanit\";
                                                       font-size: {self.font_size}px;
                                                       margin: 4px 2px;""",
                                              base_color="#637173",
                                              foreground_base_color="black",
                                              foreground_changed_color="black"
                                              )
        self.save_button.setText("บันทึก")
        self.save_button.clicked.connect(self.save_data)

        self.delete_button = general.PushButton(style_sheet="""background-color: %s;
                                                       border: none;
                                                       border-radius: 5px;
                                                       color: %s;
                                                       padding: 16px 32px;
                                                       text-align: center;
                                                       text-decoration: none;
                                                       font: bold \"Kanit\";
                                                       font-size: {self.font_size}px;
                                                       margin: 4px 2px;""",
                                                base_color="#735a5d",
                                                changed_color="#b71b32",
                                                foreground_base_color="black",
                                                foreground_changed_color="black"
                                                )
        self.delete_button.setText("ลบ")
        self.delete_button.clicked.connect(self.delete_data)

        layout_button = QHBoxLayout()
        layout_button.addWidget(self.save_button)
        layout_button.addWidget(self.delete_button)

        self.data_label_layout.addLayout(layout_button)

        self.add_button = general.PushButton(self.centralwidget,
                                             style_sheet="""background-color: %s;
                                                        border: none;
                                                        border-radius: 45px;
                                                        color: %s;
                                                        padding: 16px 32px;
                                                        text-align: center;
                                                        text-decoration: none;
                                                        font: bold \"Kanit\";
                                                        font-size: {self.font_size}px;
                                                        margin: 4px 2px;""",
                                             base_color="#1bb77b",
                                             foreground_base_color="black",
                                             changed_color="#166346",
                                             foreground_changed_color="black"
                                             )
        self.add_button.setIcon(QIcon("add.png"))
        self.add_button.move(self.size().width() - 130, self.size().height())
        self.add_button.setIconSize(QSize(30, 30))
        self.add_button.setFixedSize(QSize(100, 100))
        self.add_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.add_button.clicked.connect(self.new_person)

    def load_image_passive(self):
        for index, ID in enumerate(deepcopy(self.face_data_info_loaded)):
            load_image = self.db.Storage().get_image(ID)
            if load_image is not None and load_image is not False and load_image.any():
                self.id_navigation[ID]["image_box"].setPixmap(
                    general.round_Pixmap(
                        general.convert_cv_qt(
                            load_image,
                            self.id_navigation[ID]["image_box"].size().width() - 10,
                            self.id_navigation[ID]["image_box"].size().height() - 10,
                        ),
                        10,
                    )
                )

    def init_face_ids(self):
        files = general.scan_files(self.target_directory)
        face_data_loaded = {}
        for file in files:
            print("seeking.", path.basename(file))
            with open(file, "rb") as f:
                existed_data: dict = pickle.load(f)
                if existed_data:
                    data_amount = len(existed_data["data"])
                    ID = existed_data["id"]
                    face_data_loaded[ID] = {"path": file, "data_amount": data_amount}
                else:
                    print("error bro")

            image = general.generate_profile(ID)
            layout, img_box, message_box = self.new_info_box(
                f"<font size=8><b>{ID} [{face_data_loaded[ID]['data_amount']}]</font>",
                image,
                ID)
            self.id_navigation[ID] = {"image_box": img_box, "message_box": message_box, "layout": layout}
            self.verticalLayout.addLayout(layout)
            self.face_data_info_loaded = face_data_loaded
            Thread(target=self.load_image_passive).start()

    def image_select(self, e):
        filename = QFileDialog.getOpenFileName(
            self, "Image file", filter="Image files (*.jpg *.JPG *.png *.jpeg *.PNG *.JPEG)"
        )
        if not filename[0]:
            return

        image = cv2.imread(filename[0])
        if image is None or not image.any():
            image = cv2.imread("image_error.png")
        self.current_filename = filename[0]
        self.image_label.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    image,
                    self.image_label.size().width() - 10,
                    self.image_label.size().height() - 10,
                ),
                10,
            )
        )

    def add_data_to_id(self, e):
        if self.current is None:
            return

        if self.face_data_info_loaded[self.current]["data_amount"] != 0:
            return

        mode, done = QInputDialog.getItem(self, 'Input Dialog', 'choose mode:', ["load from image files",
                                                                                 "scan from camera",
                                                                                 "load from trained file"])
        if not done:
            return

        if not preload_face_reg_model:
            from FaceTrainer_new import VideoFaceTrainer, FileFaceTrainer

        if mode == "load from image files":
            dial = QFileDialog()
            dial.setStyleSheet("background-color: white;")
            filenames = dial.getOpenFileNames(self, "Image files",
                                              filter="Image files (*.jpg *.JPG *.png *.jpeg *.PNG *.JPEG)")
            if filenames:
                filenames = filenames[0]
                fft = FileFaceTrainer(ID=self.current,
                                      output_path=self.target_directory + "/known")
                Thread(target=lambda: fft.train_now_normal(filenames)).start()

        elif mode == "load from trained file":
            dial = QFileDialog()
            dial.setStyleSheet("background-color: white;")
            filenames = dial.getOpenFileNames(self, "Trained files",
                                              filter="Pickle files (*.pkl, *.pickle, *.PKL)")
            if filenames:
                filename = filenames[0][0]
                print(filename)
                information = {}
                with open(filename, "rb") as file:
                    information = pickle.loads(file.read())
                    information["id"] = self.current
                with open(self.target_directory + "/known/" + self.current + ".pkl", "wb") as file:
                    file.write(pickle.dumps(information))

        elif mode == "scan from camera":
            vft = VideoFaceTrainer(ID=self.current,
                                   output_path=self.target_directory + "/known")
            Thread(target=lambda: (vft.run(), vft.write_data_normal_gray())).start()

    def delete_data(self):
        ID = self.current

        con = QMessageBox().question(self, "?", f"Are you sure to delete \"{ID}\". You cannot recover this.")
        if con == QMessageBox.Yes:
            if not self.created_face_not_saved.get(ID):
                remove(self.face_data_info_loaded[ID]["path"])
                self.db.delete(ID)
                del self.face_data_info_loaded[ID]

            self.__box_delete(self.id_navigation[ID]["layout"])
            self.unload_data()

            del self.id_navigation[ID]
            del self.created_face_not_saved[ID]

    def unload_data(self):
        self.current = ""
        self.current_filename = ""
        self.result["nickname"].setText("")
        self.result["student_id"].setText("0")
        self.result["student_class"].setText("")
        self.result["class_number"].setText("0")
        self.result["active_days"].setEnabled(True)
        self.result["active_days"].setText("-")
        self.result["active_days"].setEnabled(False)
        self.result["last_checked"].setEnabled(True)
        self.result["last_checked"].setText("-")
        self.result["last_checked"].setEnabled(False)
        self.image_label.clear()
        self.name_line_edit.setText("")
        self.id_label.setText("")

    def save_data(self):
        ID = self.current
        fullname = self.name_line_edit.text().split()
        realname = fullname[0]

        if len(fullname) < 2:
            surname = ""
        elif len(fullname) > 2:
            realname = " ".join(fullname[:-1])
            surname = fullname[-1]
        else:
            surname = fullname[1]

        nickname = self.result["nickname"].text()
        student_id = self.result["student_id"].text()
        student_class = self.result["student_class"].text()
        class_number = self.result["class_number"].text()

        if self.current.startswith("unknown:"):
            print("yee")
            ID_old = ID
            ID = ID.lstrip("unknown:")
            self.current = ID
            face_data = {}
            with open(self.face_data_info_loaded[ID_old]["path"], "rb") as file:
                face_data = pickle.loads(file.read())
                face_data["id"] = ID

            with open(self.face_data_info_loaded[ID_old]["path"], "wb") as file:
                file.write(pickle.dumps(face_data))

            move(self.target_directory + r"\unknown\{}.pkl".format(ID),
                 self.target_directory + r"\known\{}.pkl".format(ID))
            self.face_data_info_loaded[ID_old]["path"] = self.target_directory + r"\known\{}.pkl".format(ID)

            if self.db.get_data(ID_old) is not None:
                self.db.delete(ID_old)
                db_data = self.db.get_data(ID_old)
            else:
                db_data = {
                    "realname": "",
                    "surname": "",
                    "nickname": "",
                    "student_id": 0,
                    "student_class": "",
                    "class_number": 0,
                    "active_days": 0,
                    "last_checked": 0,
                    "graph_info": [],
                    "last_update": 0,
                }

            self.db.add_data(ID,
                             realname=db_data.get("realname", ""),
                             surname=db_data.get("surname", ""),
                             nickname=db_data.get("nickname", ""),
                             student_id=db_data.get("student_id", 0),
                             student_class=db_data.get("student_class", ""),
                             class_number=db_data.get("class_number", 0),
                             active_days=db_data.get("active_days", 0),
                             last_checked=db_data.get("last_checked", 0),
                             graph_info=db_data.get("graph_info", []),
                             )

            self.id_navigation[ID] = self.id_navigation[ID_old]
            self.face_data_info_loaded[ID] = self.face_data_info_loaded[ID_old]
            del self.id_navigation[ID_old]
            del self.face_data_info_loaded[ID_old]
            del self.created_face_not_saved[ID_old]
            self.id_navigation[ID]["message_box"].setText(f"<font size=8><b>{ID} "
                                                          f"[{self.face_data_info_loaded[ID]['data_amount']}]</font>")
            self.id_navigation[ID]["message_box"].mousePressEvent = lambda _: self.info_box_popup(ID)

        if self.db.get_data(ID) is None:
            if self.created_face_not_saved[ID]:
                with open(self.target_directory + "/known/" + ID + ".pkl", "wb") as file:
                    data = {"id": ID, "data": []}
                    file.write(pickle.dumps(data))
                    self.face_data_info_loaded[ID] = {"path": self.target_directory + "/known/" + ID + ".pkl",
                                                      "data_amount": 0}
            self.db.add_data(ID, *self.db.default)

        self.db.update(ID=ID,
                       realname=realname,
                       surname=surname,
                       nickname=nickname,
                       student_id=student_id,
                       student_class=student_class,
                       class_number=class_number)

        if self.current_filename:
            self.db.Storage().add_image(ID=ID,
                                        filename=self.current_filename,
                                        resize=(86, 86))

        if self.current_filename:
            self.id_navigation[self.current]["image_box"].setPixmap(
                general.round_Pixmap(
                    general.convert_cv_qt(
                        cv2.imread(self.current_filename),
                        self.id_navigation[self.current]["image_box"].size().width() - 10,
                        self.id_navigation[self.current]["image_box"].size().height() - 10,
                    ),
                    10,
                )
            )

    def current_box_animation_enter(self, message_box: QWidget):
        def __animate(value):
            grad = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb{value.red(), value.green(), value.blue()}, stop: 1 rgb{value.red(), value.green() - 34, value.blue()}); padding-left: 10px; color: rgb(32, 45, 47);"
            message_box.setStyleSheet(grad)

        animation1 = QVariantAnimation(self)
        animation1.valueChanged.connect(__animate)
        animation1.setStartValue(QtGui.QColor(62, 83, 87))
        animation1.setEndValue(QtGui.QColor(27, 183, 123))
        animation1.setDuration(500)
        animation1.start()

    def current_box_animation_left(self, message_box: QWidget):
        def __animate(value):
            grad = f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb{value.red(), value.green(), value.blue()}, stop: 1 rgb(32, 45, 47)); padding-left: 10px; color: #8ba0a3;"
            message_box.setStyleSheet(grad)

        animation1 = QVariantAnimation(self)
        animation1.valueChanged.connect(__animate)
        animation1.setStartValue(QtGui.QColor(27, 183, 123))
        animation1.setEndValue(QtGui.QColor(62, 83, 87))
        animation1.setDuration(500)
        animation1.start()

    def add_input(self, message, data_name, enable=True, value=None):
        horizontalLayout = QHBoxLayout()

        fontl = QFont()
        fontl.setFamily("Kanit")
        fontl.setPointSize(14)
        fontl.setBold(True)
        name_label = QLabel()
        name_label.setText(message)
        name_label.setFont(fontl)
        name_label.setStyleSheet(
            "color: #9eb5b3;"
        )

        input_lineedit = QLineEdit()
        input_lineedit.setText(str(value))
        input_lineedit.setStyleSheet(
            "background-color: rgba(140, 163, 161, 50);"
            "border-radius: 10px;"
            "border-bottom: 2px solid #507a75;"
            "padding-left: 10px;"
            "padding-bottom: 5px;"
            "color: #9eb5b3;"
        )
        input_lineedit.setEnabled(enable)
        fontl = QFont()
        fontl.setFamily("Kanit")
        fontl.setPointSize(14)
        fontl.setBold(False)
        input_lineedit.setFont(fontl)

        horizontalLayout.addWidget(name_label)
        horizontalLayout.addWidget(input_lineedit)

        self.data_label_layout.addLayout(horizontalLayout)
        self.result[data_name] = input_lineedit

    def __box_delete(self, box):
        def delete_items_of_layout(layout):
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                    else:
                        delete_items_of_layout(item.layout())
        for i in range(self.verticalLayout.count()):
            layout_item = self.verticalLayout.itemAt(i)
            if layout_item.layout() == box:
                delete_items_of_layout(layout_item.layout())
                self.verticalLayout.removeItem(layout_item)
                break

    def info_box_popup(self, ID):
        def load_image_later():
            load_image = self.db.Storage().get_image(ID)
            if load_image is not None and load_image is not False and load_image.any():
                image = load_image
                self.image_label.setPixmap(
                    general.round_Pixmap(
                        general.convert_cv_qt(
                            image,
                            self.image_label.size().width() - 10,
                            self.image_label.size().height() - 10,
                        ),
                        10,
                    )
                )

        data = self.db.get_data(ID)
        if data is None:
            if self.created_face_not_saved.get(ID) is not None:
                data = {
                    "realname": "",
                    "surname": "",
                    "nickname": "",
                    "student_id": 0,
                    "student_class": "",
                    "class_number": 0,
                    "active_days": 0,
                    "last_checked": 0,
                    "graph_info": [],
                    "last_update": 0,
                }
            else:
                self.db.add_data(ID, *self.db.default)
                data = self.db.get_data(ID)

        if self.current:
            self.current_box_animation_left(self.id_navigation[self.current]["message_box"])
        self.current_box_animation_enter(self.id_navigation[ID]["message_box"])

        realname = data["realname"] if data["surname"] is not None else ""
        surname = data["surname"] if data["surname"] is not None else ""
        name = realname + " " + surname
        nickname = data["nickname"]
        student_id = data["student_id"]
        student_class = data["student_class"]
        class_number = data["class_number"]

        active_days = len(Arrange(AttendantGraph().load_floats(data.get("graph_info")).dates).arrange_in_all_as_day()) \
            if data.get("graph_info") is not None else "-"
        last_checked = data["last_checked"]
        last_checked = datetime.datetime.fromtimestamp(last_checked).strftime("%d %b %Y %X") if last_checked != 0 else \
            "-"

        name = ID if name == " " else name

        self.name_line_edit.setText(name)
        self.name_line_edit.setCursorPosition(0)
        self.id_label.setText(ID)

        self.image_label.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    general.generate_profile(ID),
                    self.image_label.size().width() - 10,
                    self.image_label.size().height() - 10,
                ),
                10,
            )
        )
        Thread(target=load_image_later).start()
        self.result["nickname"].setText(str(nickname))
        self.result["student_id"].setText(str(student_id))
        self.result["student_class"].setText(str(student_class))
        self.result["class_number"].setText(str(class_number))
        self.result["active_days"].setEnabled(True)
        self.result["active_days"].setText(str(active_days))
        self.result["active_days"].setEnabled(False)
        self.result["last_checked"].setEnabled(True)
        self.result["last_checked"].setText(str(last_checked))
        self.result["last_checked"].setEnabled(False)

        self.current = ID
        self.current_filename = ""

    def new_person(self):
        generated_name: str = uuid4().hex
        self.result["nickname"].setText("")
        self.result["student_id"].setText("0")
        self.result["student_class"].setText("")
        self.result["class_number"].setText("0")
        self.result["active_days"].setEnabled(True)
        self.result["active_days"].setText("-")
        self.result["active_days"].setEnabled(False)
        self.result["last_checked"].setEnabled(True)
        self.result["last_checked"].setText("-")
        self.result["last_checked"].setEnabled(False)
        self.name_line_edit.setText(generated_name)
        self.id_label.setText(generated_name)
        self.image_label.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    general.generate_profile(generated_name),
                    self.image_label.size().width() - 10,
                    self.image_label.size().height() - 10,
                ),
                10,
            )
        )

        layout, img_box, message_box = self.new_info_box(f"<font size=8><b>{generated_name} [0]</font>",
                                                         None,
                                                         generated_name)

        if self.current:
            self.current_box_animation_left(self.id_navigation[self.current]["message_box"])
        self.current_box_animation_enter(message_box)

        self.created_face_not_saved[generated_name] = True
        self.id_navigation[generated_name] = {"image_box": img_box, "message_box": message_box, "layout": layout}
        self.current = generated_name
        self.current_filename = ""

        self.verticalLayout.addLayout(layout)
        self.verticalLayout.addStretch()

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
        box.setFont(QFont(font))
        box.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop: 0 rgb(62, 83, 87), stop: 1 rgb(32, 45, 47));"
            "color: #8ba0a3;"
            "padding-left: 10px;"
            "border-radius: 10px;"
        )
        box.setText(_translate("MainWindow", message))
        box.setTextFormat(Qt.RichText)
        box.mousePressEvent = lambda _: self.info_box_popup(ID)
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
            cv_image = general.generate_profile(ID)
        img_box.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    cv_image,
                    img_box.size().width() - 10,
                    img_box.size().height() - 10,
                ),
                10,
            )
        )
        img_box.setAlignment(Qt.AlignCenter)
        horizontalLayout.addWidget(img_box)
        horizontalLayout.addWidget(box)

        return horizontalLayout, img_box, box


if __name__ == "__main__":
    preload_face_reg_model = False
    # True: load model before starting program -> use a lot of memory bet best for training new face
    # False: load model when start training -> use less memory but slow when start training
    if preload_face_reg_model:
        from FaceTrainer_new import VideoFaceTrainer, FileFaceTrainer

    image_error = cv2.imread("image_error.png")
    unknown_image = cv2.imread("unknown_people.png")
    font = "Kanit"
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    a = App(r"C:\general\Science_project\Science_project_cp39\resources_test_2")
    a.show()
    sys.exit(app.exec_())
