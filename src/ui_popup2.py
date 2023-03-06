import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Ui_Dialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setObjectName("Dialog")
        self.name = ""
        self.resize(673, 494)
        self.setStyleSheet("background-color: #0b1615;")
        self.Image_box = QtWidgets.QLabel(self)
        self.Image_box.setGeometry(QtCore.QRect(20, 20, 191, 251))
        self.Image_box.setStyleSheet(
            "background-color: #142523;\n" "border-radius: 10px;\n" "border: 2px solid #305752; "
        )
        self.Image_box.setObjectName("Image_box")
        self.Image_box.setAlignment(Qt.AlignCenter)
        self.ID = QtWidgets.QLabel(self)
        self.ID.setGeometry(QtCore.QRect(230, 70, 291, 31))
        font = QtGui.QFont()
        font.setFamily("Kanit")
        font.setPointSize(14)
        self.ID.setFont(font)
        self.ID.setStyleSheet(
            "background-color: #142523;\n" "border-radius: 10px;\n" "border-bottom: 2px solid #305752;color:#9eb5b3;"
        )
        self.ID.setObjectName("ID")

        self.Image_box_2 = QtWidgets.QLabel(self)
        self.Image_box_2.setGeometry(QtCore.QRect(20, 280, 631, 220))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Image_box_2.sizePolicy().hasHeightForWidth())
        self.Image_box_2.setSizePolicy(sizePolicy)
        self.Image_box_2.setStyleSheet("background-color: #142523;\n" "border-radius: 10px;")
        self.Image_box_2.setText("")
        self.Image_box_2.setObjectName("Image_box_2")
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(230, 10, 291, 61))
        font = QtGui.QFont()
        font.setFamily("Kanit")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit.setFont(font)
        self.lineEdit.setStyleSheet("color: #9eb5b3;\n" "background: transparent;\n" "border: 0;\n" "")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.textChanged.connect(self.event_handler)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(230, 110, 291, 161))
        self.label.setStyleSheet(
            "background-color: #142523;\n"
            "color: rgb(61, 84, 79);\n"
            "border-radius: 10px;\n"
            "selection-background-color: rgb(154, 213, 202);\n"
            "selection-color: rgb(27, 13, 62);\n"
            'font: 63 8pt "Kanit SemiBold";'
        )
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setObjectName("label")

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def event_handler(self, text):
        self.name = text

    def add_data(self, data: dict):
        layout = QVBoxLayout()
        for key, values in data.items():
            data_label = QtWidgets.QLabel(f"{key}: {values}")
            layout.addWidget(data_label)
        self.label.setLayout(layout)

    def plot_graph(self, data_x, data_y):
        canvas = MplCanvas(
            self, width=self.Image_box_2.size().width() / 100, height=self.Image_box_2.size().height() / 100, dpi=100
        )

        canvas.axes.yaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        canvas.axes.plot_date(data_x, data_y, color="#c8f7e8", linestyle="solid")
        canvas.axes.axhline(y=np.nanmean(data_y), color="#c8f7e8", linestyle="dotted")

        canvas.axes.patch.set_alpha(0)
        canvas.figure.autofmt_xdate()
        canvas.axes.tick_params(axis="both", colors="#9eb5b3", labelsize=8)
        canvas.axes.spines["left"].set_linewidth(1.2)
        canvas.axes.spines["bottom"].set_linewidth(1.2)
        canvas.axes.spines[["right", "top"]].set_visible(False)
        canvas.axes.spines["bottom"].set_color("#9eb5b3")
        canvas.axes.spines["top"].set_color("#9eb5b3")
        canvas.axes.spines["right"].set_color("#9eb5b3")
        canvas.axes.spines["left"].set_color("#9eb5b3")
        canvas.figure.patch.set_alpha(0)

        # self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(self.toolbar)
        layout.addWidget(canvas)
        self.Image_box_2.setLayout(layout)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.ID.setText(_translate("Dialog", "ID"))
        self.lineEdit.setPlaceholderText(_translate("Dialog", "NAME"))
