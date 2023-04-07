import src.FaceTrainer_new as ft

print(ft.Direction((-10, -30, -10), error_rate=(1000, 50, 1000), name="-left").maximum_error())
