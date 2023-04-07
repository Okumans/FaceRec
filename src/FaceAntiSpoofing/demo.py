import torch
from src.FaceAntiSpoofing import models
import cv2
import numpy as np
import imutils
import time
from imutils.video import VideoStream
import os.path as path

model_name = "MyresNet18"
load_model_path = path.dirname(__file__) + "/a8.pth"
model = getattr(models, model_name)().eval()
model.load(load_model_path)
model.train(False)

ATTACK = 1
GENUINE = 0


def demo(img):
    data = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
    data = data[np.newaxis, :]
    data = torch.FloatTensor(data)
    with torch.no_grad():
        outputs = model(data)
        outputs = torch.softmax(outputs, dim=-1)
        predictions = outputs.to('cpu').numpy()
        attack_prob = predictions[:, ATTACK]
    return attack_prob

