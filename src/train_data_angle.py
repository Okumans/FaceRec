from FaceTrainer_new import FileFaceTrainer
import glob
import time
from general import scan_files
from uuid import uuid4
from datetime import timedelta
import os.path as path
from os import mkdir
from recognition import Recognition

if __name__ == "__main__":
    output_path: str = "angle_result_trained_new"
    path_to_faces: str = "angle_result_trained_new"
    people_to_id: str = ""
    path_to_data: str = "angle_result_trained"
    recognizer: Recognition = Recognition(path_to_data, True)

    peoples = glob.glob(f"{path_to_faces}\*")
    if not path.exists(output_path):
        mkdir(output_path)

    print(f"Total: {len(peoples)} peoples.")
    print("\n".join(sorted(peoples, key=lambda a: int(path.basename(a).split("_")[0]))))

    time_per_one_person = 0

    for index, people in enumerate(sorted(peoples, key=lambda a: int(path.basename(a).split("_")[0]))):
        estimate_time = 0 if index == 0 else time_per_one_person * (len(peoples) - (index + 1))
        start = time.time()
        people_name = path.basename(people)
        ID: str = str(uuid4().hex)
        fft: FileFaceTrainer = FileFaceTrainer(ID=ID, output_path=output_path, core=8, num_jitters=20)
        images = scan_files(people, extension=".png")
        people_to_id += f"{ID}: {people_name}\n"

        print(f'TRAINING: "{people}" as {ID}')
        print(f"AMOUNT: {len(images)}")
        print(f"PLEASE WAIT!!!\n\t estimate time: {timedelta(seconds=estimate_time)}s.")

        fft.adds(images)
        fft.train_normal()

        time_per_one_person = time.time() - start
    with open(f"{output_path}/name_mapping.txt", "w", encoding="utf-8") as file:
        file.write(people_to_id)
