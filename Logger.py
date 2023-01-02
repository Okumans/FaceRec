from math import ceil


class Logger:
    def __init__(self):
        self.show = True

    class loading:
        def __init__(self, max_process=100, loading_bar_size=20, show_progress_number=True):
            self.max_process = max_process
            self.loading_bar_size = loading_bar_size
            self.show_progress_number = show_progress_number
            self.progress = 0
            self.fg = "#"
            self.bg = "."

        def display(self):
            percent = self.progress / self.max_process
            size = percent * self.loading_bar_size
            print(
                "\r|"
                + self.fg * ceil(size)
                + self.bg * (self.loading_bar_size - ceil(size))
                + "| ["
                + str(round(percent * 100, 2))
                + "%]",
                end="",
            )

        def update(self, current_progress):
            self.progress = current_progress
            self.display()

    def log(self, message):
        if self.show:
            print(message)

    def error(self, error_type, error_message):
        self.log(f"ERROR [{error_type}]: {error_message}")
