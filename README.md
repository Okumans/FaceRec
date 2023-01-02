🤖Face Recognition System
=======================

This is a face recognition system implemented using Python and PyQt5. It uses the mediapipe library to perform face detection and recognition.

Requirements📝
------------

To run this code, you will need to have the following libraries installed:

-   PyQt5
-   NumPy
-   OpenCV
-   PyAutoGUI
-   mediapipe
-   firebase_admin
-   and any other libraries used in the code (see the list of `import` statements at the top of the script for details)

Usage💕
-----

To use this code, run the script using the following command:


`python GoodFaceRecognition.py`

This will launch the GUI, which will automatically start the face recognition process using the default settings. You can adjust the settings using the controls provided in the GUI.

Settings⚙️
--------

The following settings are available for this face recognition system:

-   `video_source`: This specifies the source for the video feed that will be used for face detection and recognition. The default value is `0`, which uses the default camera on your computer.
-   `resolution`: This specifies the resolution scale of the input. the range of resolution is between `0.001` - `4`. The default value is `1`.
-   `min_detection_confidence`: This specifies the minimum confidence level required for a face to be detected. The default value is `0.75`.
-   `min_recognition_confidence`: This specifies the minimum confidence level required for a face to be recognized. The default value is `0.6`.
-   `min_faceBlur_detection`: This specifies the minimum amount of blur required for a face to be detected. The default value is `24`.
-   `autoBrightnessContrast`: This specifies whether automatic brightness and contrast adjustment should be applied to the video feed. The default value is `False`.
-   `autoBrightnessValue`: This specifies the amount of brightness to be applied when automatic brightness adjustment is enabled. The default value is `80`.
-   `autoContrastValue`: This specifies the amount of contrast to be applied when automatic contrast adjustment is enabled. The default value is `30`.
-   `face_check_amount`: This specifies the number of consecutive frames that a face must be detected in before it is recognized. The default value is `3`.
-   `face_max_disappeared`: This specifies the maximum number of consecutive frames that a face can be absent from the video feed before it is considered to have disappeared. The default value is `10`.
-   `night_mode_brightness`: This specifies the amount of brightness to be applied when night mode is enabled. The default value is `40`.
-   `sharpness_filter`: This specifies whether a sharpness filter should be applied to the video feed. The default value is `False`.
-   `gray_mode`: This specifies whether the video feed should be displayed in grayscale. The default value is `False`.
-   `debug`: This specifies whether debug information should be displayed in the GUI. The default value is `False`.
-   `fps_show`: This specifies whether the current frame rate should be displayed in the GUI. The default value is `False`.
-   `average_fps`: This specifies how frame rate should be display as average (easy to look) or normal (harder to look). The default value is `True`.
-   `cpu_amount`: This specifies the number of CPU cores that should be used for face detection and recognition. The default value is `8`.
-   `remember_unknown_face`: This specifies whether the system should remember faces that are not recognized. The default value is `True`.
-   `face_reg_path`: This specifies the path to the directory where face recognition data should be stored. The default value is `r"C:\general\Science_project
