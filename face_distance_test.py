from general import scan_files
from recognition import Recognition
from typing import Callable
import cv2
import os.path as path
from face_recognition import face_locations
from imutils import resize
import shelve

find_distance: Callable[[int, int, float, float, float], float] = lambda W, H, res, fw, fh: (((H * res) * (W * res)) / 518400
                                                                                        ) * (((fw + fh) / 2) / 220.39
                                                                                             ) ** (1 / -0.949)
find_face_size: Callable[[int, int, float, float], float] = lambda W, H, res, d: (((H * res) * (W * res)) / 518400
                                                                                        ) * (220.39 * (d ** -0.949))

for i in sorted([i/10 for i in range(10, 105, 5)]):
    print(round(find_face_size(640, 480, 1, i)))
if __name__ == "__main__":
    path_to_faces: str = "angle_result"
    path_to_data: str = "angle_result_trained"
    files = scan_files(path_to_faces, ".png")
    recognizer: Recognition = Recognition(path_to_data, True)
    distance_test = sorted([i/10 for i in range(10, 105, 5)], reverse=True)
    resolution = (1920, 1080)
    name_map = {}
    result = {}

    with open(r"C:\general\Science_project\Science_project_cp39\angle_result_trained\name_mapping.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                value, key = line.strip().split(":")
                name_map[key.strip()] = value.strip()

    for distance in distance_test:
        for index, file in enumerate(files):
            degree_raw = path.basename(path.dirname(file))

            face_name = path.basename(path.dirname(path.dirname(file)))
            da_name = name_map[path.basename(path.dirname(path.dirname(file)))]

            if degree_raw == "0xy degree":
                image = cv2.imread(file)
                face_size = find_face_size(resolution[0], resolution[1], 1, distance)
                box = face_locations(image)
                if box:
                    top, right, bottom, left = box[0]
                    image = image[top:bottom, left:right]
                    height = abs(bottom-top)
                    width = abs(left-right)
                    image = resize(image, width=round(face_size))
                    height, width, _ = image.shape

                    if width*height >= 0:
                        dist = find_distance(resolution[0], resolution[1], 1, width, height)
                        print(face_size, width, height, dist)

                        name = recognizer.recognition(image)
                        print(name[0][0], da_name, (name[0][0] == da_name))

                        if result.get(distance) is None:
                            result[distance] = {}

                        if result.get(distance).get(face_name) is None:
                            result[distance][face_name] = []

                        result[distance][face_name].append(name[0][0] == da_name)

                        # with shelve.open('distance_result_result_1') as db:
                        #     db["result"] = result

                        image = cv2.putText(image, f"{round(dist, 2)}M {index}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("win", image)
                cv2.waitKey(1)
