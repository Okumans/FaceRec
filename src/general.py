"""
Filename: general.py
Author: Jeerabhat Supapinit
"""
from __future__ import annotations
import ray
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import Qt
import cv2
import os.path as path
from collections import Iterable
from PIL import Image, ImageStat, ImageEnhance, ImageDraw, ImageFont
import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import ctypes
import platform
from PyQt5.QtGui import QPixmap, QFont
import imutils
from typing import *
import glob
import time
import threading


class MessageIO:
    @staticmethod
    def ask(message: str, input_type: Callable) -> Any:
        print_msg_box(message)
        return input_type(input(": "))

    @staticmethod
    def show(message: str):
        print_msg_box(message)

    @staticmethod
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

    @staticmethod
    def ask_y_n(message: str, ignore_case=False, return_boolean=False, upper_y=False):
        print_msg_box(message + " (y/n)")
        ch = input(": ")
        choices = ("y", "n") if upper_y is False else ("Y", "n")
        while (ch.strip() if ignore_case is False else ch.strip().lower()) not in choices:
            ch = input(": ")
        return (ch == ("y" if upper_y is False else "Y")) if return_boolean is True else ch

    @staticmethod
    def ask_until_sure(message: str, input_type: Callable):
        ans: str = ""
        sure: bool = False
        while not sure:
            ans = MessageIO.ask(message, str)
            sure = MessageIO.ask_y_n(f"are you sure? [{ans}]", ignore_case=True, return_boolean=True)
        return input_type(ans)


class AnimateProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximum(100)


def contrast(im_data_rgb):
    if im_data_rgb.any():
        return cv2.cvtColor(im_data_rgb, cv2.COLOR_RGB2GRAY).std()
    else:
        return -1


def brightness(im_data_rgb):
    if im_data_rgb.any():
        im = Image.fromarray(im_data_rgb)
        stat = ImageStat.Stat(im)
        r, g, b = stat.rms
        return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
    else:
        return -1


def convert_cv_qt(cv_img, width, height):
    """Convert from an opencv image to QPixmap"""
    cv_img = imutils.resize(cv_img, width=width)
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    return QPixmap.fromImage(convert_to_Qt_format)


def msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    return box


def print_msg_box(msg, indent=1, width=None, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)


def generate_profile(
        name: str, image_source: str = "resources/unknown_people.png", font_path: str = "Kanit-Medium.ttf"
) -> np.ndarray:
    if name is False:
        return

    img = Image.open(image_source)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 50)
    name = name[:2]
    offset_x = font.getlength(name)
    height, width = img.height, img.width
    draw.text((int(width / 2 - (offset_x / 2)), height // 2 - 55), name, (203, 203, 203), font=font)
    return np.array(img.convert("RGB"))


class FloatingButtonWidget(QtWidgets.QPushButton):
    def __init__(self, parent):
        super().__init__(parent)
        self.paddingLeft = 5
        self.paddingTop = 5

    def update_position(self):
        if hasattr(self.parent(), "viewport"):
            parent_rect = self.parent().viewport().rect()
        else:
            parent_rect = self.parent().rect()

        if not parent_rect:
            return

        x = parent_rect.width() - self.width() - self.paddingLeft
        y = self.paddingTop
        self.setGeometry(x, y, self.width(), self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_position()

    def mousePressEvent(self, event):
        self.parent().floatingButtonClicked.emit()


class Average:
    def __init__(self, values=None, calculate_amount=None):
        self.length = 0 if values is None else len(values)
        self.values = 0 if values is None else sum(values)
        self.result = 0 if self.length == 0 else self.values / self.length
        self.calculate_amount = calculate_amount

    def add(self, number):
        if self.calculate_amount is not None and self.length >= self.calculate_amount:
            self.length = 0
            self.values = 0

        self.length += 1
        self.values += number
        self.result = self.values / self.length

    def adds(self, numbers: list):
        if self.calculate_amount is not None and self.length >= self.calculate_amount:
            self.length = 0
            self.values = 0

        self.length += len(numbers)
        self.values += sum(numbers)
        self.result = self.values / self.length

    def get(self):
        return self.result

    def clear(self):
        self.length = 0
        self.values = 0
        self.result = 0


def Most_Common(dtc) -> Dict:
    max_con = {"name": "", "confidence": -100000}
    for i in dtc:
        avg = sum(dtc[i]) / len(dtc[i])
        if avg > max_con["confidence"]:
            max_con["confidence"] = avg
            max_con["name"] = i
    return max_con


def get_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def convertQImageToMat(incomingImage):
    incomingImage = incomingImage.convertToFormat(4)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    return arr


def putBorderText(
        img,
        text,
        org,
        fontFace,
        fontScale,
        fg_color,
        bg_color,
        thickness,
        border_thickness=2,
        lineType=None,
        bottomLeftOrigin=None,
):
    cv2.putText(
        img=img,
        text=str(text),
        org=tuple(map(int, org)),
        fontFace=fontFace,
        fontScale=fontScale,
        color=bg_color,
        thickness=thickness + border_thickness,
        lineType=lineType,
        bottomLeftOrigin=bottomLeftOrigin,
    )
    cv2.putText(
        img=img,
        text=str(text),
        org=tuple(map(int, org)),
        fontFace=fontFace,
        fontScale=fontScale,
        color=fg_color,
        thickness=thickness,
        lineType=lineType,
        bottomLeftOrigin=bottomLeftOrigin,
    )


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


def get_from_percent(integer, percent, rounding=True):
    return round((percent / 100) * integer) if rounding else (percent / 100) * integer


def log(message, destination=None, end="\n", sep=" ", color=(255, 255, 255)):
    if destination is not None and path.exists(destination):
        with open(destination, "a+") as file:
            if isinstance(message, Iterable):
                message = sep.join(message)
            file.write(message + end)
    else:
        print(colored(color[0], color[1], color[2], message), sep=sep, end=end)


def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


def change_brightness_to(img, value):
    bright = brightness(img)
    if bright != -1:
        value -= bright
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        # cv2.imshow("2", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(5)
    return img


def change_contrast_to(img, value):
    con = contrast(img)
    if con != -1:
        factor = value / con
        img = np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor))
    return img


@ray.remote
class rayDict:
    def __init__(self, default_value: dict = None):
        self.default_value = {} if default_value is None else default_value

    def get_all(self):
        return self.default_value

    def set(self, key, value):
        self.default_value[key] = value

    def get(self, key):
        return self.default_value.get(key)

    def clear(self):
        self.default_value = {}

    def delete(self, key):
        if self.default_value.get(key) is not None:
            del self.default_value[key]

    def recursive_get(self, *keys):
        result = self.default_value
        for i in keys:
            result = result[i]
        return result

    def recursive_update(self, update_dict: dict, *keys):
        result = self.default_value
        for i in keys:
            result = result[i]
        result.update(update_dict)


@ray.remote
class rayList:
    def __init__(self, default_value: list = None):
        self.default_value = [] if default_value is None else default_value

    def get_all(self):
        return self.default_value

    def set(self, index, value):
        self.default_value[index] = value

    def append(self, value):
        self.default_value.append(value)

    def extend(self, value):
        self.default_value.extend(value)

    def sort(self, reverse=False):
        self.default_value.sort(reverse=reverse)

    def get(self, index):
        return self.default_value[index]

    def delete(self, index):
        self.default_value.pop(index)

    def clear(self):
        self.default_value = []


class Direction:
    Undefined = -1
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Forward = 4

    def __init__(self, degree_X_Y_Z: tuple[float], error_rate: Union[tuple[float], float] = None, name: str = None):
        self.degree_x: float = degree_X_Y_Z[0]
        self.degree_y: float = degree_X_Y_Z[1]
        self.degree_z: float = degree_X_Y_Z[2]
        self.name = "" if name is None else name

        error_rate = (0, 0, 0) if error_rate is None else error_rate
        if type(error_rate) == tuple:
            self.error_rate_x: float = error_rate[0]
            self.error_rate_y: float = error_rate[1]
            self.error_rate_z: float = error_rate[2]
        else:
            self.error_rate_x: float = error_rate
            self.error_rate_y: float = error_rate
            self.error_rate_z: float = error_rate

    def __hash__(self):
        return hash((self.degree_x, self.degree_y, self.degree_z))

    def __eq__(self, other):
        return (self.degree_x, self.degree_y, self.degree_z) == (other.degree_x, other.degree_y, other.degree_z)

    def maximum_error(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        max_error_left: Callable[[float], float] = (
            lambda degree, error_rate: degree * ((100 - error_rate) / 100) - error_rate / 10
        )
        max_error_right: Callable[[float], float] = (
            lambda degree, error_rate: degree * ((100 + error_rate) / 100) + error_rate / 10
        )

        return (
            (
                max_error_left(self.degree_x, self.error_rate_x),
                max_error_left(self.degree_y, self.error_rate_y),
                max_error_left(self.degree_z, self.error_rate_z),
            ),
            (
                max_error_right(self.degree_x, self.error_rate_x),
                max_error_right(self.degree_y, self.error_rate_y),
                max_error_right(self.degree_z, self.error_rate_z),
            ),
        )

    def direction(self) -> Tuple[float, float, float]:
        return self.degree_x, self.degree_y, self.degree_z

    def main_direction(self):
        degree_x = self.degree_x
        degree_y = self.degree_y
        degree_z = self.degree_z
        if degree_x >= degree_y and degree_x >= degree_z:
            return "x"
        if degree_y >= degree_x and degree_y >= degree_z:
            return "y"
        if degree_z >= degree_x and degree_z >= degree_y:
            return "z"

    def is_same(self, direction_: Direction) -> bool:
        max_error_left = (
            lambda degree, error_rate: (degree * ((100 - error_rate) / 100))
                                       + (-error_rate if degree > 0 else +error_rate) / 10
        )
        max_error_right = (
            lambda degree, error_rate: degree * ((100 + error_rate) / 100)
                                       + (error_rate if degree > 0 else -error_rate) / 10
        )
        min_x = min(max_error_left(self.degree_x, self.error_rate_x), max_error_right(self.degree_x, self.error_rate_x))
        max_x = max(max_error_left(self.degree_x, self.error_rate_x), max_error_right(self.degree_x, self.error_rate_x))
        min_y = min(max_error_left(self.degree_y, self.error_rate_y), max_error_right(self.degree_y, self.error_rate_y))
        max_y = max(max_error_left(self.degree_y, self.error_rate_y), max_error_right(self.degree_y, self.error_rate_y))
        min_z = min(max_error_left(self.degree_z, self.error_rate_z), max_error_right(self.degree_z, self.error_rate_z))
        max_z = max(max_error_left(self.degree_z, self.error_rate_z), max_error_right(self.degree_z, self.error_rate_z))
        print(min_x, max_x, min_y, max_y, min_z, max_z)
        if min_x <= direction_.degree_x <= max_x:
            if min_y <= direction_.degree_y <= max_y:
                if min_z <= direction_.degree_z <= max_z:
                    return True
        return False


class Color:
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Blue = (0, 0, 255)
    Yellow = (255, 255, 0)
    Cyan = (0, 255, 255)
    Violet = (255, 0, 255)
    Black = (0, 0, 0)
    White = (255, 255, 255)


def round_Pixmap(pixmap, radius):
    rounded = QtGui.QPixmap(pixmap.size())
    rounded.fill(QtGui.QColor("transparent"))
    painter = QtGui.QPainter(rounded)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setBrush(QtGui.QBrush(pixmap))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(pixmap.rect(), radius, radius)
    painter.end()
    return rounded


def make_dpi_aware():
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)


def scan_files(directory: str, extension: str = ".pkl") -> List[str]:
    def dfs(_directory: str, _file_type: str) -> List[str]:
        if path.isdir(_directory):
            _files: List[str] = []
            for _file in glob.glob(path.join(_directory, "*")):
                _files.extend(dfs(_file, _file_type))
            return _files
        else:
            if path.splitext(_directory)[1] == _file_type:
                return [_directory]
            return []

    if not path.exists(directory):
        return [""]

    return dfs(directory, extension)


class PushButton(QtWidgets.QPushButton):
    def __init__(
            self,
            parent=None,
            base_color="#0b1615",
            changed_color="#1bb77b",
            style_sheet="""background-color: %s;
            border: none;
            border-radius: 10px;
            color: %s;
            padding: 16px 32px;
            text-align: center;
            text-decoration: none;
            font: bold \"Kanit\";
            font-size: {self.font_size}px;
            margin: 4px 2px;""",
            foreground_base_color="#637173",
            foreground_changed_color="black",
    ):
        super().__init__(parent)
        self._animation = QtCore.QVariantAnimation(
            startValue=QtGui.QColor(changed_color),
            endValue=QtGui.QColor(base_color),
            valueChanged=self._on_value_changed,
            duration=400,
        )
        self.base_color = base_color
        self.changed_color = changed_color
        self.style_sheet = style_sheet
        self.foreground_base_color = foreground_base_color
        self.foreground_changed_color = foreground_changed_color
        self.font_size = 16
        font = QFont("Kanit", self.font_size)
        font.setBold(True)
        self.setFont(font)
        self._update_stylesheet(QtGui.QColor(self.base_color), QtGui.QColor(self.foreground_base_color))
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def _on_value_changed(self, color):
        foreground = (
            QtGui.QColor(self.foreground_base_color)
            if self._animation.direction() == QtCore.QAbstractAnimation.Forward
            else QtGui.QColor(self.foreground_changed_color)
        )
        self._update_stylesheet(color, foreground)

    def _update_stylesheet(self, background, foreground):
        self.setStyleSheet(
            f"""
        QPushButton{{
            {self.style_sheet}
            font-size: {self.font_size}px;
        }}
        """
            % (background.name(), foreground.name())
        )

    def enterEvent(self, event):
        self._animation.setDirection(QtCore.QAbstractAnimation.Backward)
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.setDirection(QtCore.QAbstractAnimation.Forward)
        self._animation.start()
        super().leaveEvent(event)


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
