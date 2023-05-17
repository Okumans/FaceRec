import cv2
import numpy as np
from os import listdir
import os.path as path
import warnings
try:
    from src.FaceAntiSpoofing.anti_spoof_predict import AntiSpoofPredict
    from src.FaceAntiSpoofing.utility import parse_model_name
except ModuleNotFoundError:
    from FaceAntiSpoofing.anti_spoof_predict import AntiSpoofPredict
    from FaceAntiSpoofing.utility import parse_model_name

warnings.filterwarnings("ignore", category=UserWarning)


class LivenessDetection:
    def __init__(self):
        self.attack_list = ['other attack', 'no attack', '2D attack']
        self.model_test = AntiSpoofPredict(0)

    @staticmethod
    def determine(result):
        model1_result = -1 if result["model1"]["state"] in ['other attack', '2D attack'] else 1
        model2_result = -1 if result["model2"]["state"] in ['other attack', '2D attack'] else 1

        return (model1_result*result["model1"]["value"] + model2_result*result["model2"]["value"])/2

    def predict(self, face_image):
        prediction_values = []
        prediction_labels = []
        model_count = 1

        for model_name in listdir(path.dirname(__file__) + "/FaceAntiSpoofing/anti_spoof_models"):
            prediction = np.zeros((1, 3))
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            resized_image = cv2.resize(face_image, (w_input, h_input))
            prediction += self.model_test.predict_onnx(resized_image, model_count)
            prediction += self.model_test.predict_onnx(resized_image, model_count)
            prediction_values.append(prediction[0][np.argmax(prediction)])
            prediction_labels.append(np.argmax(prediction))
            model_count += 1

        return {"model1": {"state": self.attack_list[prediction_labels[0]], "value": prediction_values[0]},
                "model2": {"state": self.attack_list[prediction_labels[1]], "value": prediction_values[1]}}


def predict(image):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([image], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if success:
            cv2.imshow("win", frame)
            cv2.waitKey(1)
