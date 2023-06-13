from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QInputDialog, QFileDialog, QDialog, QApplication, QMainWindow
from src import general, ShadowRemoval
import numpy as np
import multiprocessing
from copy import deepcopy
import face_recognition
import sys


class Ui_Dialog(QDialog):
    minConfi: QLabel

    def __init__(self, parent=None, default_setting=None, image=None):
        super().__init__(parent=parent)
        image = cv2.imread("kaopan.jpg") if image is None else image
        self.setObjectName("Dialog")
        self.resize(int(437/1.5), int(514/1.5))
        self.setStyleSheet("background-color: white;")
        self.setting = default_setting
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.imageBox = QtWidgets.QLabel(self)
        self.fpsShow = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.rememberUnknownFace = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.debugMode = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.grayMode = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.sharpFilter = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.autoBright = QtWidgets.QCheckBox(self.horizontalLayoutWidget_2)
        self.mxFaceCap = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Resolution = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.nightModeVal = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.mxfaceTimeDiss = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.autoConVal = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.autoBrightVal = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.faceBlurVal = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.minConfi = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.minConfiRec = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.vdoSrc = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.mxFaceCap_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.resolution_box = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.nightModeVal_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.mxfaceTimeDiss_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.autoConVal_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.autoBrightVal_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.faceBlurVal_box = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.minConfi_box = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.minConfiRec_box = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.vdoSrc_box = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.image = 0
        self.face_location = ()

        self.imageBox.setGeometry(QtCore.QRect(50, 30, 381//2, 221//2))
        self.imageBox.setAlignment(Qt.AlignCenter)
        self.imageBox.setStyleSheet("background-color: rgb(211, 211, 211);")
        self.imageBox.setObjectName("label_14")

        self.image = cv2.cvtColor(
            general.convertQImageToMat(
                self.convert_cv_qt(image, self.imageBox.size().width(), self.imageBox.size().height()).toImage()
            ),
            cv2.COLOR_BGRA2BGR,
        )
        self.face_location = face_recognition.face_locations(self.image)

        self.buttonBox.setGeometry(QtCore.QRect(160, 300, 121, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")

        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 160, 180, 180))
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.vdoSrc_box.activated.connect(self.vdoSrc_event)
        if self.setting["video_source"] != 0:
            self.vdoSrc_box.addItem(str(self.setting["video_source"]))
        self.vdoSrc_box.addItem("0")
        self.vdoSrc_box.addItem("screen")
        self.vdoSrc_box.addItem("file")
        self.vdoSrc_box.addItem("online")
        self.verticalLayout.addWidget(self.vdoSrc_box)

        self.minConfi_box.valueChanged.connect(self.minConfi_event)
        self.minConfi_box.setSingleStep(0.05)
        self.minConfi_box.setRange(0, 100)
        self.minConfi_box.setValue(self.setting["min_detection_confidence"])
        self.minConfi_box.setSuffix(" %")
        self.verticalLayout.addWidget(self.minConfi_box)

        self.minConfiRec_box.valueChanged.connect(self.minConfi_event)
        self.minConfiRec_box.setSingleStep(0.05)
        self.minConfiRec_box.setRange(0, 100)
        self.minConfiRec_box.setValue(self.setting["min_recognition_confidence"])
        self.minConfiRec_box.setSuffix(" %")
        self.verticalLayout.addWidget(self.minConfiRec_box)

        self.faceBlurVal_box.valueChanged.connect(self.faceBlurVal_event)
        self.faceBlurVal_box.setRange(0, 1000)
        self.faceBlurVal_box.setValue(self.setting["min_faceBlur_detection"])
        self.verticalLayout.addWidget(self.faceBlurVal_box)

        if not self.setting["autoBrightnessContrast"]:
            self.autoBrightVal_box.setValue(int(general.brightness(self.image)))
            self.autoConVal_box.setValue(int(general.contrast(self.image)))
        else:
            self.autoBrightVal_box.setValue(self.setting["autoBrightnessValue"])
            self.autoConVal_box.setValue(self.setting["autoContrastValue"])

        self.autoBrightVal_box.valueChanged.connect(self.autoBrightVal_event)
        self.autoBrightVal_box.setRange(0, 255)
        self.autoBrightVal_box.setEnabled(self.setting["autoBrightnessContrast"])
        self.verticalLayout.addWidget(self.autoBrightVal_box)

        self.autoConVal_box.valueChanged.connect(self.autoConVal_event)
        self.autoConVal_box.setRange(0, 255)
        self.autoConVal_box.setEnabled(self.setting["autoBrightnessContrast"])
        self.verticalLayout.addWidget(self.autoConVal_box)

        self.mxfaceTimeDiss_box.valueChanged.connect(self.mxfaceTimeDiss_event)
        self.mxfaceTimeDiss_box.setSuffix(" ms")
        self.mxfaceTimeDiss_box.setValue(self.setting["face_max_disappeared"])
        self.mxfaceTimeDiss_box.setRange(0, 10000)
        self.verticalLayout.addWidget(self.mxfaceTimeDiss_box)

        self.nightModeVal_box.valueChanged.connect(self.nightModeVal_event)
        self.nightModeVal_box.setRange(0, 255)
        self.nightModeVal_box.setValue(self.setting["night_mode_brightness"])
        self.verticalLayout.addWidget(self.nightModeVal_box)

        self.resolution_box.valueChanged.connect(self.Resolution_box_event)
        self.resolution_box.setRange(0.01, 4)
        self.resolution_box.setSingleStep(0.1)
        self.resolution_box.setValue(1)
        self.verticalLayout.addWidget(self.resolution_box)

        self.mxFaceCap_box.valueChanged.connect(self.mxFaceCap_event)
        self.mxFaceCap_box.setValue(self.setting["face_check_amount"])
        self.verticalLayout.addWidget(self.mxFaceCap_box)

        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3.addWidget(self.vdoSrc)
        self.verticalLayout_3.addWidget(self.minConfi)
        self.verticalLayout_3.addWidget(self.minConfiRec)
        self.verticalLayout_3.addWidget(self.faceBlurVal)
        self.verticalLayout_3.addWidget(self.autoBrightVal)
        self.verticalLayout_3.addWidget(self.autoConVal)
        self.verticalLayout_3.addWidget(self.mxfaceTimeDiss)
        self.verticalLayout_3.addWidget(self.nightModeVal)
        self.verticalLayout_3.addWidget(self.Resolution)
        self.verticalLayout_3.addWidget(self.mxFaceCap)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(190, 160, 180, 130))
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.autoBright.stateChanged.connect(self.autoBright_event)
        self.autoBright.setText("auto brightness")
        self.autoBright.setChecked(self.setting["autoBrightnessContrast"])
        self.verticalLayout_2.addWidget(self.autoBright)

        self.sharpFilter.stateChanged.connect(self.sharpFilter_event)
        self.sharpFilter.setText("sharpness filter")
        self.sharpFilter.setChecked(self.setting["sharpness_filter"])
        self.verticalLayout_2.addWidget(self.sharpFilter)

        self.grayMode.stateChanged.connect(self.grayMode_event)
        self.grayMode.setText("gray mode")
        self.grayMode.setChecked(self.setting["gray_mode"])
        self.verticalLayout_2.addWidget(self.grayMode)

        self.debugMode.stateChanged.connect(self.debugMode_event)
        self.debugMode.setText("debug")
        self.debugMode.setChecked(self.setting["debug"])
        self.verticalLayout_2.addWidget(self.debugMode)

        self.fpsShow.stateChanged.connect(self.fpsShow_event)
        self.fpsShow.setText("fps")
        self.fpsShow.setChecked(self.setting["fps_show"])
        self.verticalLayout_2.addWidget(self.fpsShow)

        self.rememberUnknownFace.stateChanged.connect(self.rememberUnknownFace_event)
        self.rememberUnknownFace.setText("remember Unknown face")
        self.rememberUnknownFace.setChecked(self.setting["remember_unknown_face"])
        self.verticalLayout_2.addWidget(self.rememberUnknownFace)

        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.retranslateUi()
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.submitclose)

    def submitclose(self):
        self.accept()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Dialog"))
        self.vdoSrc.setText(_translate("Dialog", "video source"))
        self.minConfi.setText(_translate("Dialog", "min detection confidence"))
        self.minConfiRec.setText(_translate("Dialog", "min recognition confidence"))
        self.faceBlurVal.setText(_translate("Dialog", "min face blur"))
        self.autoBrightVal.setText(_translate("Dialog", "auto brightness val"))
        self.autoConVal.setText(_translate("Dialog", "auto contrast val"))
        self.mxfaceTimeDiss.setText(_translate("Dialog", "max face disapeared"))
        self.nightModeVal.setText(_translate("Dialog", "night mode brightness"))
        self.Resolution.setText(_translate("Dialog", "resolution"))
        self.mxFaceCap.setText(_translate("Dialog", "face check amount"))
        self.autoBright.setText(_translate("Dialog", "auto brightness"))
        self.sharpFilter.setText(_translate("Dialog", "sharpness filter"))
        self.grayMode.setText(_translate("Dialog", "gray mode"))
        self.debugMode.setText(_translate("Dialog", "debug"))
        self.fpsShow.setText(_translate("Dialog", "fps"))

    def set_value_now(self):
        image_now = general.change_brightness_to(self.image, self.autoBrightVal_box.value())
        image_now = general.change_contrast_to(image_now, self.autoConVal_box.value())
        return image_now

    def setter(self):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image_now = deepcopy(self.image)

        if self.autoBright.isChecked():
            image_now = general.change_brightness_to(image_now, self.autoBrightVal_box.value())
            image_now = general.change_contrast_to(image_now, self.autoConVal_box.value())

        if self.resolution_box.value() > 0:
            image_now = cv2.resize(image_now, (0, 0), fx=self.resolution_box.value(), fy=self.resolution_box.value())

        if self.sharpFilter.isChecked():
            image_now = cv2.filter2D(src=image_now, ddepth=-1, kernel=kernel)

        if self.grayMode.isChecked() or self.nightModeVal_box.value() > self.autoBrightVal_box.value():
            image_now = cv2.cvtColor(ShadowRemoval.remove_shadow_grey(image_now), cv2.COLOR_GRAY2RGB)
            if self.nightModeVal_box.value() > self.autoBrightVal_box.value():
                general.putBorderText(
                    image_now,
                    "NIGHT MODE",
                    (180, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    (0, 0, 0),
                    2,
                    3,
                )

        if self.fpsShow.isChecked():
            cv2.putText(
                image_now,
                "fps:-",
                (7, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if self.debugMode.isChecked():
            general.putBorderText(
                image_now,
                f"confident: {self.minConfi_box.value()}% blur {self.faceBlurVal_box.value()} ",
                (100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                (0, 0, 0),
                1,
                2,
            )
            general.putBorderText(
                image_now,
                f"brightness: {self.autoBrightVal_box.value()} contrast: {self.autoConVal_box.value()}",
                (100, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                (0, 0, 0),
                1,
                2,
            )

        if not self.face_location:
            cv2.rectangle(image_now, (100, 50), (102, 110), (0, 0, 0), 5)
            cv2.rectangle(image_now, (100, 50), (102, 110), (255, 255, 0), 3)
        else:
            top, right, bottom, left = self.face_location[0]
            cv2.rectangle(image_now, (left, top), (right, bottom), (0, 0, 0), 5)
            cv2.rectangle(image_now, (left, top), (right, bottom), (255, 255, 0), 3)

        self.imageBox.setPixmap(
            self.convert_cv_qt(image_now, self.imageBox.size().width(), self.imageBox.size().height())
        )

    def vdoSrc_event(self, value):
        source = 0
        if value == self.vdoSrc_box.count() - 1:
            text, ok = QInputDialog.getText(self, "online", "internet ip.")
            if ok:
                source = text
            else:
                self.vdoSrc_box.setCurrentIndex(0)
        elif value == self.vdoSrc_box.count() - 2:
            file = QFileDialog.getOpenFileName(self, "Open a Video File", "", filter="Video File (*.mov *.mp4)")
            if file[0]:
                source = file[0]
            else:
                self.vdoSrc_box.setCurrentIndex(0)
        else:
            if self.vdoSrc_box.currentText().isnumeric():
                source = int(self.vdoSrc_box.currentText())
            else:
                source = self.vdoSrc_box.currentText()

        if self.setting["video_source"] != source:
            self.setting["video_source"] = source
            self.setting["video_change"] = True

    def minConfi_event(self, value):
        self.setter()
        self.setting["min_detection_confidence"] = value

    def faceBlurVal_event(self, value):
        self.setter()
        self.setting["min_faceBlur_detection"] = value

    def autoBrightVal_event(self, value):
        self.setter()
        self.setting["autoBrightnessValue"] = value

    def autoConVal_event(self, value):
        self.setter()
        self.setting["autoContrastValue"] = value

    def mxfaceTimeDiss_event(self, value):
        self.setter()
        self.setting["face_max_disappeared"] = value

    def nightModeVal_event(self, value):
        self.setter()
        self.setting["night_mode_brightness"] = value

    def Resolution_box_event(self, value):
        self.setter()
        self.Resolution.setText(
            f"resolution [{int(self.setting['base_resolution'][0]*self.resolution_box.value())}x{int(self.setting['base_resolution'][1]*self.resolution_box.value())}]"
        )
        self.setting["resolution"] = value

    def mxFaceCap_event(self, value):
        self.setting["face_check_amount"] = value

    def autoBright_event(self, value):
        self.setting["autoBrightnessContrast"] = bool(value)
        self.autoBrightVal_box.setEnabled(bool(value))
        self.autoConVal_box.setEnabled(bool(value))
        if not bool(value):
            self.autoBrightVal_box.setValue(int(general.brightness(self.image)))
            self.autoConVal_box.setValue(int(general.contrast(self.image)))

    def sharpFilter_event(self, value):
        self.setter()
        self.setting["sharpness_filter"] = bool(value)

    def grayMode_event(self, value):
        self.setter()
        self.setting["gray_mode"] = bool(value)

    def debugMode_event(self, value):
        self.setter()
        self.setting["debug"] = bool(value)

    def fpsShow_event(self, value):
        self.setter()
        self.setting["fps_show"] = bool(value)

    def rememberUnknownFace_event(self, value):
        self.setting["remember_unknown_face"] = bool(value)

    @staticmethod
    def convert_cv_qt(cv_img, width, height) -> QPixmap:
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class CustomDialog(QDialog):
    @staticmethod
    def convert_cv_qt(cv_img, width, height) -> QPixmap:
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("HELLO!")

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QtWidgets.QVBoxLayout()
        message = QLabel("Something happened, is that OK?")
        picture = QLabel(self)
        picture.setPixmap(self.convert_cv_qt(cv2.imread("image_error.png"), 200, 200))
        self.layout.addWidget(message)
        self.layout.addWidget(picture)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


if __name__ == "__main__":
    import json
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    with open("../settings.json", "r", encoding="utf-8") as file:
        settings = json.load(file)
        settings["base_resolution"] = (640, 480)
        app = QApplication(sys.argv)
        MainWindow = QMainWindow()
        dlg = Ui_Dialog(MainWindow, image=cv2.imread("resources/kaopan.jpg"), default_setting=settings)
        dlg.show()
        sys.exit(app.exec_())
