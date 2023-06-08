import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import json
import multiprocessing

# Initial settings
default_setting = {
    "project_path": "",
    "face_reg_path": "",
    "db_path": "",
    "db_cred_path": "",
    "name_map_path": "",
    "cache_path": "",
    "font": "Kanit",
    "autoBrightnessContrast": False,
    "sharpness_filter": False,
    "gray_mode": False,
    "debug": False,
    "liveness_detection": False,
    "fps_show": True,
    "average_fps": True,
    "remember_unknown_face": True,
    "save_as_video": False,
    "face_alignment": False,
    "video_source": 0,
    "min_detection_confidence": 0.7,
    "min_recognition_confidence": 0.55,
    "min_liveness_confidence": 0.7,
    "min_faceBlur_detection": 24,
    "autoBrightnessValue": 80,
    "autoContrastValue": 30,
    "face_check_amount": 1,
    "face_max_disappeared": 10,
    "night_mode_brightness": 40,
    "cpu_amount": multiprocessing.cpu_count(),
    "resolution": 1,
    "rotate_frame": 0,
    "platform": "win",
    "shared": False
}

try:
    with open("settings.json", "r") as f:
        setting = json.load(f)
except FileNotFoundError:
    with open("settings.json", "w") as f:
        json.dump(default_setting, f)
        setting = default_setting


def update_setting(key, value):
    print(key)
    setting[key] = value

    with open("settings.json", "w") as f:
        json.dump(setting, f)


def browse_directory(k, e):
    path = filedialog.askdirectory()
    if path:
        update_setting(k, path)
        e.delete(0, tk.END)
        e.insert(0, path)


def browse_file(k, e):
    path = filedialog.askopenfilename(filetypes=[('All files', "*")])
    if path:
        update_setting(k, path)
        e.delete(0, tk.END)
        e.insert(0, path)


def create_setting_frame():
    root = tk.Tk()
    root.title("Settings")
    root.geometry("500x500")

    # Create a scrollable area
    canvas = tk.Canvas(root)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    header_label = ttk.Label(frame, text="Settings", font=("Helvetica", 16, "bold"))
    header_label.pack(pady=10)

    for key, value in setting.items():
        if key in ["project_path", "face_reg_path", "db_path", "db_cred_path", "name_map_path", "cache_path"]:
            subframe = ttk.Frame(frame)
            subframe.pack(fill=tk.X, padx=10, pady=5)

            label = ttk.Label(subframe, text=key)
            label.pack(side=tk.LEFT)

            entry = ttk.Entry(subframe)
            entry.insert(0, value)
            entry.pack(side=tk.LEFT, expand=True)

            if key in ["db_cred_path", "name_map_path"]:
                button = ttk.Button(subframe, text=f"Browse", command=lambda k=key, e=entry: browse_file(k, e))
            else:
                button = ttk.Button(subframe, text=f"Browse", command=lambda k=key, e=entry: browse_directory(k, e))
            button.pack(side=tk.LEFT, padx=(10, 0))
        else:
            subframe = ttk.Frame(frame)
            subframe.pack(fill=tk.X, padx=10, pady=5)

            label = ttk.Label(subframe, text=key)
            label.pack(side=tk.LEFT)

            if type(value) == str:
                entry = ttk.Entry(subframe)
                entry.insert(0, value)
                entry.pack(side=tk.LEFT, expand=True)
            elif type(value) == int:
                entry = ttk.Spinbox(subframe, from_=0, to=1000)
                entry.set(value)
                entry.pack(side=tk.LEFT, expand=True)
            elif type(value) == float:
                entry = ttk.Spinbox(subframe, from_=0, to=1000, increment=0.01)
                entry.set(value)
                entry.pack(side=tk.LEFT, expand=True)
            elif type(value) == bool:
                entry = ttk.Combobox(subframe)
                entry["values"] = [False, True]
                entry.set(str(value))
                entry.pack(side=tk.LEFT, expand=True)
            else:
                entry = ttk.Entry(subframe)
                entry.insert(0, value)
                entry.pack(side=tk.LEFT, expand=True)

            button = ttk.Button(subframe, text="Update", command=lambda k=key: update_setting(k, entry.get()))
            button.pack(side=tk.LEFT, padx=(10, 0))

            # Save settings and exit

    def save_settings():
        with open("settings.json", "w") as f:
            json.dump(setting, f)
        root.destroy()

    finish_button = ttk.Button(frame, text="Finished", command=save_settings)
    finish_button.pack(pady=10)

    canvas.update_idletasks()  # Update the canvas to calculate scroll region
    root.mainloop()


if __name__ == "__main__":
    create_setting_frame()

