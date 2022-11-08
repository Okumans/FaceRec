"""
Filename: HeadPoseEstimation.py
Author: Jeerabhat Supapinit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from trainer import spilt_chunk, resize_by_height, split_chunks_of
import ray
from ShadowRemoval import remove_shadow_grey
from general import direction, putBorderText, change_brightness, get_from_percent
from topData import topData
from copy import deepcopy
import pickle
from FaceAlignment import face_alignment
import face_recognition


def grid_images(imgs, width, each_image_size=(100, 100)):
    horizontals = []
    imgs = [cv2.resize(raw_img, each_image_size) for raw_img in imgs]
    for img_chunk in split_chunks_of(imgs, width):
        if img_chunk:
            base = np.zeros((each_image_size[1], each_image_size[0] * width, 3), dtype=np.uint8)
            horizon_img = np.concatenate(img_chunk, axis=1)
            base[0:horizon_img.shape[0], 0:horizon_img.shape[1]] = horizon_img
            horizontals.append(base)
    return np.concatenate(horizontals, axis=0)


@ray.remote
def process_image(info, resize_size_y=200):
    print("processing image....")
    face_encodings = []
    for img in info:
        face_location = face_recognition.face_locations(img)
        if face_location:
            face_encoding = face_recognition.face_encodings(img, face_location)
            if face_encoding:
                # print("succc")
                face_encodings.append(face_encoding[0])
        else:
            pass
            # print("unsuccc")
    return face_encodings


def face_not_found(img):
    putBorderText(img, "Face not found (T-T) ", (int(W / 2) - 250, int(H / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                  (255, 145, 30), (0, 0, 0), 3, 5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = change_brightness(img, -10)
    cv2.setWindowProperty('win', cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow('win', img)
    if cv2.waitKey(5) & 0xFF == 27:
        quit()


def process_images_and_write_images(data, id):
    max_image_amount = len(data)
    cv2.imshow("win", cv2.cvtColor(grid_images(data, 12), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    a = [process_image.remote(chunk) for chunk in spilt_chunk(data, cores)]
    result = []
    success_images = 0
    for i in ray.get(a):
        for j in i:
            success_images += 1
            result.append(j)
    print(f"Success {success_images}/{max_image_amount} {round((success_images / max_image_amount) * 100, 2)}%")

    with open(f"{id}.pkl", "wb") as f:
        pickle.dump({"id": id, "data": result}, f)
    cv2.destroyAllWindows()
    print("finished..")


mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh()
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.55, model_selection=0)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 0))
to_be_encode = {direction.Forward: topData(max_size=20),
                direction.Left: topData(),
                direction.Right: topData(),
                direction.Up: topData(max_size=20)}  # direction.Down: topData()
(H, W) = (None, None)
cap = cv2.VideoCapture('http://192.168.1.102:8080/video')
face_direction = direction.Undefined
to_check_direction = direction.Forward
min_detection_score = .85
cores = 8
ray.init()

if __name__ == "__main__":
    ID = input("please enter id: ")
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        if H is None or W is None:
            (H, W) = image.shape[:2]

        start = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_mesh = face_mesh.process(image)
        results_detection = face_detection.process(image)
        image.flags.writeable = True

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if to_check_direction == direction.Undefined:
            putBorderText(image, "Please wait... (*_*)", (int(W / 2) - 250, int(H / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                          (255, 0, 255), (0, 0, 0), 3, 5)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = change_brightness(image, -10)
            cv2.imshow('win', image)
            cv2.setWindowProperty("win", cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            break

        if results_detection.detections is not None:
            if results_detection.detections[0] is None:
                face_not_found(image)
                continue
            if len(results_detection.detections) > 1:
                face_not_found(image)
                continue
        else:
            face_not_found(image)
            continue

        detection = results_detection.detections[0]
        x_min = detection.location_data.relative_bounding_box.xmin * W
        y_min = detection.location_data.relative_bounding_box.ymin * H
        x_max = x_min + detection.location_data.relative_bounding_box.width * W
        y_max = y_min + detection.location_data.relative_bounding_box.height * H
        face_width = x_max - x_min
        face_height = y_max - y_min
        box = (x_min, y_min, x_max, y_max)
        now_frame = face_alignment(deepcopy(image[int(box[1]) - get_from_percent(face_height, 20):
                                            int(box[3]) + get_from_percent(face_height, 20),
                                            int(box[0]) - get_from_percent(face_height, 20):
                                            int(box[2]) + get_from_percent(face_height, 20)]), face_mesh)
        # print(detection.score[0])

        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -3:
                    face_direction = direction.Left
                    if to_check_direction == direction.Left:
                        if to_be_encode[direction.Left].lowest() >= min_detection_score and \
                                to_be_encode[direction.Left].is_full():
                            to_check_direction = direction.Right
                        else:
                            to_be_encode[direction.Left].add(detection.score[0], now_frame)

                elif y > 3:
                    face_direction = direction.Right
                    if to_check_direction == direction.Right:
                        if to_be_encode[direction.Right].lowest() >= min_detection_score and to_be_encode[
                                direction.Right].is_full():
                            to_check_direction = direction.Up
                        else:
                            to_be_encode[direction.Right].add(detection.score[0], now_frame)

                elif x < -6:
                    face_direction = direction.Down

                elif x > 7:
                    face_direction = direction.Up
                    if to_check_direction == direction.Up:
                        if to_be_encode[direction.Up].lowest() >= min_detection_score and to_be_encode[
                                direction.Up].is_full():
                            to_check_direction = direction.Undefined
                        else:
                            to_be_encode[direction.Up].add(detection.score[0], now_frame)

                else:
                    face_direction = direction.Forward
                    if to_check_direction == direction.Forward:
                        if to_be_encode[direction.Forward].lowest() >= min_detection_score and to_be_encode[
                                direction.Forward].is_full():
                            to_check_direction = direction.Left
                        else:
                            to_be_encode[direction.Forward].add(detection.score[0], now_frame)

                # print(face_direction, len(to_be_encode[direction.Forward].get()), detection.score[0], min_detection_score, to_check_direction)
                text = f"Looking {face_direction.name}"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                putBorderText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), (0, 0, 0), 2, 3)
                putBorderText(image, f"please look {to_check_direction.name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                              (0, 255, 255), (0, 0, 0), 3, 4)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            end = time.time()
            totalTime = end - start
            if totalTime > 0:
                fps = 1 / totalTime
            else:
                fps = -1

            cv2.putText(image, f'FPS: {int(fps)} Confidence: {round(detection.score[0], 2)}', (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('win', image)
        cv2.setWindowProperty("win", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    rgb_data = []
    data_grey = []

    for key in to_be_encode:
        for img in to_be_encode[key].get():
            img = resize_by_height(img, 400)
            rgb_data.append(img)
            data_grey.append(cv2.cvtColor(remove_shadow_grey(img), cv2.COLOR_GRAY2RGB))

    process_images_and_write_images(rgb_data, ID)
    process_images_and_write_images(data_grey, ID+"_GREY")
