import glob
import general
import cv2
import os.path as path
from face_recognition import face_locations
from recognition import Recognition
import shelve


if __name__ == '__main__':
    path_to_faces: str = "angle_result"
    images_by_angle: dict[tuple[str, str], list] = {}
    folders: list[str] = glob.glob(f"{path_to_faces}/*")
    path_to_data: str = "angle_result_trained"
    recognizer: Recognition = Recognition(path_to_data, True)
    result = {}
    name_map = {}
    with open(r"C:\general\Science_project\Science_project_cp39\angle_result_trained\name_mapping.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                value, key = line.strip().split(":")
                name_map[key.strip()] = value.strip()

    print(name_map)
    """
    angle_result -|
                  |- folder [{number}_{uuid}]-|
                                              |- folder [{angle}{axis} degree}]-|
                                                                                |- .png images [{image index}]
    
    """

    for folder in sorted(folders, key=lambda a: int(path.basename(a).split("_")[0])):
        number: str = path.basename(folder).split("_")[0]
        uuid: str = path.basename(folder).split("_")[1]

        for degree_folder in glob.glob(f"{folder}/*"):
            axis: str = "".join(["" if i.isnumeric() or i == "-" else i for i in path.basename(degree_folder).split()[0]])
            degree: str = "".join([i if i.isnumeric() or i == "-" else "" for i in path.basename(degree_folder).split()[0]])

            for image_path in glob.glob(f"{degree_folder}/*"):
                if images_by_angle.get((axis, degree)) is None:
                    images_by_angle[(axis, degree)] = []
                images_by_angle[(axis, degree)].append(image_path)

    print(images_by_angle.keys())

    counter = 0
    counter_all = 0
    for angle_folder in images_by_angle:
        angle_counter_all = 0
        angle_counter_success = 0

        print(angle_folder)
        for image in images_by_angle[angle_folder]:
            angle_counter_all += 1
            counter_all += 1

            name = path.basename(path.dirname(path.dirname(image)))
            if result.get(angle_folder) is None:
                result[angle_folder] = {}

            if result.get(angle_folder).get(name) is None:
                result[angle_folder][name] = []

            image = cv2.imread(image)

            # box = face_locations(image, model="hog")
            # if not box:
            #     continue
            #
            # top, right, bottom, left = box[0]
            # image = image[top:bottom, left:right]

            detect_name = recognizer.recognition(image)
            print(name_map[name], detect_name[0][0], (name_map[name] == detect_name[0][0]))
            result[angle_folder][name].append((name_map[name] == detect_name[0][0]))

            with shelve.open('angle_result_result') as db:
                db["result"] = result

            print(result)

            counter += 1
            angle_counter_success += 1
            cv2.imshow("win", cv2.putText(image, angle_folder[1]+angle_folder[0]+" "+str(counter),
                                          (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2))
            cv2.waitKey(1)
        print(f"{angle_counter_success}/{angle_counter_all} ({-1 if angle_counter_all == 0 else angle_counter_success/angle_counter_all*100})\n")

    print(f"result {counter}/{counter_all}")
    cv2.destroyAllWindows()
