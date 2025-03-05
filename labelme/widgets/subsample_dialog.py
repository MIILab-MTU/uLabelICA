from qtpy import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider, QRadioButton, QButtonGroup, QVBoxLayout
from qtpy import QtWidgets


class PatientNameDialog(QtWidgets.QDialog):
    def __init__(self, value, parent=None):
        super(PatientNameDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Patient ID Value")
        self.line_edit = QtWidgets.QLineEdit(self)
        #self.line_edit.setText("{}".format(value))
        # Create a QPushButton object
        self.save_button = QtWidgets.QPushButton("Save")
        # Add the widgets to the layout
        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.line_edit)
        formLayout.addRow(self.save_button)
        self.setLayout(formLayout)

        # Connect the save button's clicked signal to a function that saves the line edit's text to the config file
        self.save_button.clicked.connect(self.on_save_button_clicked)

    def on_save_button_clicked(self):
        self.close()
class GraphGenerationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GraphGenerationDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Graph Generation Values")

        self.centerlineLengthLabel = QtWidgets.QLabel("Centerline Length Threshold:")
        self.centerlineLengthThreshold = QtWidgets.QSpinBox()
        self.centerlineLengthThreshold.setRange(1, 30)
        self.centerlineLengthThreshold.setSingleStep(1)
        self.centerlineLengthThreshold.setValue(8)

        self.radiusLabel = QtWidgets.QLabel("Radius Threshold:")
        self.radiusThreshold = QtWidgets.QDoubleSpinBox()
        self.radiusThreshold.setRange(0.1, 1.0)
        self.radiusThreshold.setSingleStep(0.1)
        self.radiusThreshold.setValue(0.5)

        self.save_button = QtWidgets.QPushButton("Done")

        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.centerlineLengthLabel, self.centerlineLengthThreshold)
        formLayout.addRow(self.radiusLabel, self.radiusThreshold)
        formLayout.addRow(self.save_button)
        self.setLayout(formLayout)

        self.save_button.clicked.connect(self.save_button_clicked)
    def save_button_clicked(self):
        self.close()




class EraserSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(EraserSettingsDialog, self).__init__(parent)
        self.setWindowTitle("Eraser Settings")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        # self.slider.setValue(10)
        #print(eraser_size, type(eraser_size))


        self.black_checked = QRadioButton("Paint")
        self.black_checked.setChecked(True)
        self.white_checked = QRadioButton("Erase")

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.black_checked)
        self.button_group.addButton(self.white_checked)
        self.button_group.setId(self.black_checked, 0)
        self.button_group.setId(self.white_checked, 1)

        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.slider)
        formLayout.addRow(self.black_checked)
        formLayout.addRow(self.white_checked)
        self.setLayout(formLayout)

