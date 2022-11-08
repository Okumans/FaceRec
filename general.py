"""
Filename: general.py
Author: Jeerabhat Supapinit
"""

from collections import Counter
import ray
from PyQt5.QtWidgets import QProgressBar
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from enum import Enum
import cv2
import os.path as path
from collections import Iterable
from PIL import Image, ImageStat, ImageEnhance
import math
import numpy as np


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


def Most_Common(lst):
    if lst is not None:
        data = Counter(lst)
        return data.most_common(1)[0][0]


def putBorderText(img, text, org, fontFace, fontScale, fg_color, bg_color, thickness, border_thickness=2, lineType=None,
                  bottomLeftOrigin=None):
    cv2.putText(img=img, text=str(text), org=org, fontFace=fontFace, fontScale=fontScale, color=bg_color,
                thickness=thickness + border_thickness, lineType=lineType, bottomLeftOrigin=bottomLeftOrigin)
    cv2.putText(img=img, text=str(text), org=org, fontFace=fontFace, fontScale=fontScale, color=fg_color,
                thickness=thickness, lineType=lineType, bottomLeftOrigin=bottomLeftOrigin)


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
        factor = value/con
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


class direction(Enum):
    Undefined = -1
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Forward = 4


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
