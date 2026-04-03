import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import subprocess

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "test_image.png")


class ImageGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image GUI")
        self.resize(600, 400)

        # --- Main layout ---
        main_layout = QVBoxLayout()

        # --- Buttons ---
        button_layout = QHBoxLayout()

        self.btn1 = QPushButton("Scan")
        self.btn2 = QPushButton("Select goals")
        self.btn3 = QPushButton("Start Pushing")

        # Connect buttons
        self.btn1.clicked.connect(self.on_scan)
        self.btn2.clicked.connect(self.on_select_goals)
        self.btn3.clicked.connect(self.on_start_pushing)

        button_layout.addWidget(self.btn1)
        button_layout.addWidget(self.btn2)
        button_layout.addWidget(self.btn3)


        # --- Image label ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        print("Loading:", image_path)
        print("Exists:", os.path.exists(image_path))

        self.pixmap = QPixmap(image_path)

        if self.pixmap.isNull():
            print("❌ Failed to load image")

        # Add to layout
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)

        # Initial display
        self.update_image()

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)

    def update_image(self):
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    # --- Button actions ---
    def on_scan(self):
        print("Scan button pressed")

        # Example action: change window title
        self.setWindowTitle("Button 1 Clicked")
        # Run another Python script (replace 'other_file.py' with your file)
        script_path = "push_anything_create_masks.py"  # can be relative or absolute
        subprocess.Popen(["python3", os.path.join(base_dir, script_path)])

    def on_select_goals(self):
        print("Now selecting goals")
        self.points = []
        self.update_image()
        self.setWindowTitle("Interactive Image GUI")

    def on_start_pushing(self):
        # close the window and exit the app
        print("Start Pushing button pressed")
        self.close()
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageGUI()
    window.show()
    sys.exit(app.exec_())