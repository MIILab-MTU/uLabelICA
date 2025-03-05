import os
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QImage, QColor
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider, QRadioButton,QButtonGroup,QAction,QMenu
from PyQt5 import  QtCore
from labelme.widgets.subsample_dialog import EraserSettingsDialog
class EraserWindow(QMainWindow):
    def __init__(self,parent_window,image_path):
        super().__init__()
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, False)

        self.parent_window = parent_window
        self.setWindowTitle("Image Eraser")
        self.setGeometry(0, 0, 512, 512)

        save_menu = self.menuBar().addMenu("Save Image")
        eraser_menu = self.menuBar().addMenu("Eraser")

        save_action = QAction("Save Image", self)
        save_action.triggered.connect(self.save_image)
        save_menu.addAction(save_action)
        self.menuBar().addMenu(save_menu)

        eraser_settings  = QAction("Eraser Settings",self)
        eraser_settings.triggered.connect(self.eraser_settings) #TODO DEFINE ERASER SETTINGS
        eraser_menu.addAction(eraser_settings)
        self.menuBar().addMenu(eraser_menu)

        self.image_label = QLabel()
        self.image_label.setGeometry(400,400,512,512)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.image_path = image_path
        self.image = QImage(self.image_path)

        self.eraser_tool_enabled = True
        if self.parent_window._config["eraser_color"] == 'black':
            self.erase_color = QColor(255,0,0,250)
        elif self.parent_window._config["eraser_color"] == 'white':
            self.erase_color = QColor.blue
        self.erase_size = int(self.parent_window._config["eraser_size"])
        self.last_pos = QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.eraser_tool_enabled:
            self.last_pos = event.pos()
            self.draw(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.eraser_tool_enabled:
            self.draw(event.pos())

    def draw(self, pos):
        painter = QPainter(self.image)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setPen(self.erase_color)
        painter.setBrush(self.erase_color)
        painter.drawEllipse(pos, self.erase_size, self.erase_size) #TODO ADD BRUSH CONTROL
        painter.end()
        self.update()

    def save_image(self):
        save_path = os.getcwd()+"labelme/data/label/image.png" #TODO REPLACE IMAGE WITH FILENAME
        predefined_path = self.image_path
        if self.image.save(predefined_path):
            print(f"Image saved to {predefined_path}")
        else:
            print("Failed to save the image.")
        self.parent_window.setEnabled(True)
        self.parent_window.loadFile(predefined_path)
        self.close()

    def eraser_settings(self):
        dialog = EraserSettingsDialog(self)
        dialog.slider.setValue(self.erase_size)
        if self.erase_color is Qt.black:
                dialog.black_checked.setChecked(True)
        elif self.erase_color is Qt.white:
            dialog.white_checked.setChecked(True)
        dialog.exec_()
        self.parent_window._config["eraser_size"] = dialog.slider.value()
        self.erase_size = self.parent_window._config["eraser_size"]
        print(dialog.black_checked.isChecked())
        if dialog.black_checked.isChecked():
            self.parent_window._config["eraser_color"] = 'black'
            self.erase_color = QColor(255,0,0,250)
        elif dialog.white_checked.isChecked():
            self.parent_window._config["eraser_color"] = 'white'
            self.erase_color = QColor(0,0,255,250)

