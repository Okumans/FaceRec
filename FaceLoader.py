"""
Structure:
    image: image_data
    realname:
    surname:
    nickname:
    student-id:
    class:
    class-number:
    active-days:
    last_checked:
    graph-info: ...
"""
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox
from PyQt5.QtCore import QCoreApplication, Qt, QRect
from PyQt5.QtGui import QFont
import sys
import cv2
import ui_popup2
import general
import pickle
import numpy as np
from DataBase import DataBase


def edit_image_event(e):
    global image_filename
    filename = QFileDialog.getOpenFileName(
        MainWindow, "Image file", filter="Image files (*.jpg *.JPG *.png *.jpeg " "*.PNG *.JPEG)"
    )
    if not filename:
        return

    image = cv2.imread(filename[0])
    if image is None or not image.any():
        image = cv2.imread("image_error.png")
    image_filename = filename[0]
    a.Image_box.setPixmap(
        general.round_Pixmap(
            general.convert_cv_qt(
                image,
                a.Image_box.size().width() - 10,
                a.Image_box.size().height() - 10,
            ),
            10,
        )
    )
    avg_color = list(map(int, np.average(np.average(image, axis=0), axis=0)))
    a.Image_box.setStyleSheet(
        "border-radius: 10px;\n"
        "border: 2px solid rgb(202, 229, 229);"
        f"background-color: rgb({avg_color[2]}, {avg_color[1]}, {avg_color[0]});\n"
    )

    print(avg_color)


def edit_id_event(e):
    global IDD
    filename = QFileDialog.getOpenFileName(MainWindow, "Pickle file", filter="Pickle files (*.pkl *.pickle *.PKL)")

    if not filename and not filename[0]:
        return

    ID: str = ""
    data_amount: int = 0

    with open(filename[0], "rb") as file:
        existed_data: dict = pickle.load(file)
        if existed_data:
            data_amount = len(existed_data["data"])
            ID = existed_data["id"]
        else:
            return

    a.ID.setStyleSheet(
        "background-color: #385c58;\n" "border-radius: 10px;\n" "border-bottom: 2px solid #305752;color:#d3e0df;"
    )
    a.ID.setText(f"{ID} [{data_amount}]")
    IDD = ID

    db_data = db.get_data(ID)
    print(ID, db_data)
    if db_data is not None:
        a.lineEdit.setText(db_data["realname"] + " " + db_data["surname"])
        data["nickname"].setText(db_data["nickname"])
        data["student_id"].setText(str(db_data["student_id"]))
        data["student_class"].setText(db_data["student_class"])
        data["class_number"].setText(str(db_data["class_number"]))

    image = db.Storage().get_image(ID)
    if image.any():
        a.Image_box.setPixmap(
            general.round_Pixmap(
                general.convert_cv_qt(
                    image,
                    a.Image_box.size().width() - 10,
                    a.Image_box.size().height() - 10,
                ),
                10,
            )
        )
        avg_color = list(map(int, np.average(np.average(image, axis=0), axis=0)))
        a.Image_box.setStyleSheet(
            "border-radius: 10px;\n"
            "border: 2px solid rgb(202, 229, 229);"
            f"background-color: rgb({avg_color[2]}, {avg_color[1]}, {avg_color[0]});\n"
    )


def add_input(name, data_name):
    horizontalLayout = QHBoxLayout()

    name_label = QLabel()
    name_label.setText(name)
    name_label.setFont(font)
    name_label.setStyleSheet(
        "color: #9eb5b3;"
    )

    input_lineedit = QLineEdit()
    input_lineedit.setStyleSheet(
            "background-color: #8ca3a1;"
            "border-radius: 10px;"
            "border-bottom: 2px solid #305752;"
            "padding-left: 10px;"
            "color: #142523;"
        )
    fontl = QFont()
    fontl.setFamily("Kanit")
    fontl.setPointSize(16)
    fontl.setBold(False)
    input_lineedit.setFont(fontl)

    horizontalLayout.addWidget(name_label)
    horizontalLayout.addWidget(input_lineedit)

    verticalLayout.addLayout(horizontalLayout)
    data[data_name] = input_lineedit


def submit_event(e):
    if not (a.lineEdit.text() and a.ID.text()):
        warning = QMessageBox()
        warning.setIcon(QMessageBox.Warning)
        warning.setText(f"name or ID not found.")
        warning.setInformativeText('please check that all box was filled')
        warning.setWindowTitle("Warning!")
        warning.exec_()
        return

    fullname = a.lineEdit.text().split()
    realname = fullname[0]
    if len(fullname) < 2:
        surname = "-"
    elif len(fullname) > 2:
        realname = " ".join(fullname[:-1])
        surname = fullname[-1]
    else:
        surname = fullname[1]

    for key, value in data.items():
        if not value.text():
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Warning)
            warning.setText(f"{key} not found.")
            warning.setInformativeText('please check that all box was filled')
            warning.setWindowTitle("Warning!")
            warning.exec_()
            return

    print(image_filename, a.ID.text(), realname, surname, data["nickname"], data["student_id"], data["student_class"], data["class_number"], 0, 0, [])

    if image_filename:
        db.Storage().add_image(IDD, image_filename, (80, 80))

    if db.get_data(IDD) is None:
        db.add_data(
            IDD,
            realname=realname,
            surname=surname,
            nickname=data["nickname"].text(),
            student_id=data["student_id"].text(),
            student_class=data["student_class"].text(),
            class_number=data["class_number"].text(),
            active_days=0,
            last_checked=0,
            graph_info=[]
        )
    else:
        db.update(
            IDD,
            realname=realname,
            surname=surname,
            nickname=data["nickname"].text(),
            student_id=data["student_id"].text(),
            student_class=data["student_class"].text(),
            class_number=data["class_number"].text()
        )


if __name__ == "__main__":
    data = {}
    db = DataBase("Students")
    font = QFont()
    font.setFamily("Kanit")
    font.setPointSize(18)
    font.setBold(True)
    IDD = ""
    image_filename = ""
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    a = ui_popup2.Ui_Dialog(MainWindow)
    a.resize(540, 514//2)

    verticalLayout = QVBoxLayout()
    add_input("ชื่อเล่น\t ", "nickname")
    add_input("รหัสนักเรียน", "student_id")
    add_input("ห้อง\t ", "student_class")
    add_input("เลขที่\t ", "class_number")
    a.label.setLayout(verticalLayout)

    submit = general.PushButton(a)
    submit.font_size = 12
    submit.setGeometry(QRect(400, 250, 120, 30))
    submit.setFont(font)
    submit.setText("เสร็จสิ้น")
    submit.clicked.connect(submit_event)

    a.Image_box_2.hide()
    a.Image_box.mousePressEvent = edit_image_event
    a.ID.mousePressEvent = edit_id_event

    sys.exit(a.exec_())
