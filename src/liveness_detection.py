import cv2
import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms as T
from skimage.feature import local_binary_pattern


class LivenessDetection:
    def __init__(self, model_path: str):
        providers = ['CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)

    def predict(self, img):
        if img.shape != (112, 112, 3):
            img = cv2.resize(img, (112, 112))

        dummy_face = np.expand_dims(np.array(img, dtype=np.float32), axis=0) / 255.
        onnx_predict = self.model.run(['activation_5'], {"input": dummy_face})
        liveness_score = list(onnx_predict[0][0])[1]

        return liveness_score


class FaceLivenessDetector:
    def __init__(self, radius=3, n_points=8 * 3, threshold=0.5):
        self.radius = radius
        self.n_points = n_points
        self.threshold = threshold

    def extract_lbp_features(self, image):
        # Convert the image to grayscale
        image = np.array(Image.fromarray(image).convert('L'))

        # Extract LBP features
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, self.n_points + 3), density=True)
        return hist

    def is_real_face(self, image):
        # Extract LBP features from the input image
        hist = self.extract_lbp_features(image)

        # Compute the LBP distance between the input image and a reference image
        # with known texture characteristics (e.g., a printed photo)
        reference_hist = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0])
        distance = np.sum((hist - reference_hist) ** 2)

        # Return True if the LBP distance is below the threshold (i.e., the input image is a real face)
        # Return False otherwise
        if distance < self.threshold:
            return True
        else:
            return False


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if success:
            cv2.imshow("win", frame)
            cv2.waitKey(1)
