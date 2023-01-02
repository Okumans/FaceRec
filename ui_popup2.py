from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QInputDialog, QFileDialog, QDialog
from PyQt5.QtCore import Qt


class Ui_Dialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setObjectName("Dialog")
        self.name = ""
        self.resize(673, 494)
        self.setStyleSheet("background-color: #0b1615;")
        self.Image_box = QtWidgets.QLabel(self)
        self.Image_box.setGeometry(QtCore.QRect(20, 20, 191, 221))
        self.Image_box.setStyleSheet(
            "background-color: #142523;\n"
            "border-radius: 10px;\n"
            "border: 2px solid #305752; "
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
        self.calendarWidget = QtWidgets.QCalendarWidget(self)
        self.calendarWidget.setGeometry(QtCore.QRect(20, 280, 241, 191))
        font = QtGui.QFont()
        font.setFamily("Kanit SemiBold")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(7)
        self.calendarWidget.setFont(font)
        self.calendarWidget.setStyleSheet(
            "background-color: rgb(202, 229, 229);\n"
            "color: rgb(61, 84, 79);\n"
            "border: 6px solid transparent;\n"
            "border-radius: 10px;\n"
            "selection-background-color: rgb(154, 213, 202);\n"
            "selection-color: rgb(27, 13, 62);\n"
            'font: 63 8pt "Kanit SemiBold";'
        )
        self.calendarWidget.setObjectName("calendarWidget")
        self.Image_box_2 = QtWidgets.QLabel(self)
        self.Image_box_2.setGeometry(QtCore.QRect(270, 280, 381, 191))
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
        self.label.setGeometry(QtCore.QRect(230, 110, 291, 131))
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

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.ID.setText(_translate("Dialog", "ID"))
        self.lineEdit.setPlaceholderText(_translate("Dialog", "NAME"))
