class App(QWidget):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent=parent)
        self.spacer = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.last_progress_ = {}
        self.info_boxes = {}
        self.info_boxes_ID = []
        self.db = DataBase("Students", sync_with_offline_db=True)
        self.db.offline_db_folder_path = r"C:\general\Science_project\Science_project_cp39\resources_test_2"
        parent.setWindowTitle("Qt live label demo")
        parent.resize(1336, 553)
        parent.setStyleSheet("background-color: #0b1615;")

        self.centralwidget = QWidget(parent)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)

        self.image_label = QLabel(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMinimumSize(QSize(640, 480))
        self.image_label.setMaximumSize(QSize(640 * 3, 480 * 3))
        self.image_label.setStyleSheet(
            "color: rgb(240, 240, 240);\n"
            "padding-top: 15px;\n"
            "background: qlineargradient( x1:0 y1:0, x2:0 y2:1, stop:0 rgb(32, 45, 47), stop:.5 #3d5c57, stop:1 rgb("
            "32, 45, 47)); "
            "border-radius: 10px;"
        )
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.setting_button = general.PushButton(self.centralwidget)
        self.setting_button.setIcon(QIcon("setting.png"))
        self.setting_button.setIconSize(QSize(30, 30))
        self.setting_button.setMaximumSize(QSize(60, 99999999))
        # self.setting_button.setFixedSize(QSize(100, 100))

        self.setting_button.clicked.connect(self.handle_dialog)
        self.setting_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.verticalLayout_1 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_1.addWidget(self.image_label)
        self.verticalLayout_1.addWidget(self.setting_button)
        self.verticalLayout_1.addItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.horizontalLayout.addLayout(self.verticalLayout_1)

        self.scrollArea = QScrollArea(self.centralwidget)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setAcceptDrops(False)
        self.scrollArea.setAutoFillBackground(False)
        self.scrollArea.setStyleSheet("background-color: #142523;\n" "border-radius: 10px;")
        self.scrollArea.verticalScrollBar().setStyleSheet(scrollbar_style)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)

        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 563, 539))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.scrollArea.raise_()
        self.image_label.raise_()
        parent.setCentralWidget(self.centralwidget)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_infobox_message_signal.connect(self.update_infobox)
        self.thread.start()