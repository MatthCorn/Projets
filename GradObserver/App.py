from PyQt5.QtWidgets import (QApplication, QWidget, QSplitter, QHBoxLayout, QScrollArea, QGridLayout,
                             QVBoxLayout, QLabel, QPushButton, QGroupBox, QFileDialog, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
import qdarkstyle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from GradObserver.GradObserverUI import GradObserverWidget
from GradObserver.VisualGradWindow import Histogram3DWidget
from GradObserver.FusionWindow import Fusion3DWidget
from GradObserver.UneditableLineEdit import UneditableLineEdit
from GradObserver.ErrorWindow import ErrorWindowWidget
from Tools.XMLTools import loadXmlAsObj
import pyqtgraph as pg
import pickle




# onglet scenario, composé de 4 sous-fenêtres
class ScenarioWindow(QWidget):
    resized = pyqtSignal()

    def __init__(self, dico):
        super().__init__()

        self.resize(1400, 800)

        self.stylesheet = qdarkstyle.load_stylesheet_pyqt5()
        self.setStyleSheet(self.stylesheet)
        self.openGradWindow = []

        self.Window = QSplitter(self)

        self.FusionWindow = Fusion3DWidget()
        self.TopLeftWindow = QSplitter()
        self.TopLeftWindow.setOrientation(Qt.Vertical)
        self.dico = dico
        self.GradObserverUI = GradObserverWidget(self.dico)
        self.GradObserverUI.CommandAll(self.MakeVisionWindow)
        self.ScrollGradObserver = QScrollArea()
        SBWidget = QWidget()
        layout = QHBoxLayout(SBWidget)
        layout.addWidget(self.GradObserverUI, alignment=Qt.AlignCenter)
        self.ScrollGradObserver.setWidget(SBWidget)
        self.ScrollGradObserver.setWidgetResizable(True)

        openWidget = QGroupBox()
        openLayout = QHBoxLayout()
        self.NetName = UneditableLineEdit('Network name')
        self.OpenButton = QPushButton('Open')
        self.OpenButton.clicked.connect(lambda: loadGradObserver(self))
        openLayout.addWidget(QLabel('Network name :'))
        openLayout.addWidget(self.NetName)
        openLayout.addWidget(self.OpenButton)
        openWidget.setLayout(openLayout)
        openWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.plotWindow = ErrorWindowWidget(self)

        self.TopLeftWindow.addWidget(openWidget)
        self.TopLeftWindow.addWidget(self.ScrollGradObserver)
        self.TopLeftWindow.addWidget(self.plotWindow)

        self.TopLeftWindow.setStretchFactor(0, 0)
        self.TopLeftWindow.setStretchFactor(1, 1)

        self.Window.addWidget(self.TopLeftWindow)
        self.Window.addWidget(self.FusionWindow)

        self.resized.connect(self.resizeWindow)

    def MakeVisionWindow(self, GradObUI):
        if GradObUI.full_name in self.openGradWindow:
            return
        subwin = QGroupBox(GradObUI.full_name)
        widget = Histogram3DWidget(GradObUI, self.FusionWindow)
        sublayout = QVBoxLayout()
        sublayout.addWidget(widget)
        widget.CloseButton.clicked.connect(lambda: self.openGradWindow.remove(widget.full_name))
        widget.CloseButton.clicked.connect(lambda: subwin.hide())
        subwin.setLayout(sublayout)
        self.openGradWindow.append(GradObUI.full_name)
        self.Window.insertWidget(1, subwin)

#######################################################################################
##  remise à l'échelle de la fenêtre                                                 ##
                                                                                     ##
    def resizeEvent(self, event):                                                    ##
        self.resized.emit()                                                          ##
        super(ScenarioWindow, self).resizeEvent(event)                               ##
                                                                                     ##
    def resizeWindow(self):                                                          ##
        self.Window.resize(self.geometry().width(), self.geometry().height())        ##
                                                                                     ##
#######################################################################################

def loadGradObserver(parent):
    file , check = QFileDialog.getOpenFileName(None, "Load gradient observer","", "pickel Files (*.pkl)")
    if check:
        parts = file.split('/')
        text = parts[-3] + '.' + parts[-2] + '.' + parts[-1]
        parent.NetName.setText(text)

        # Specify the path from where you want to load the object
        path_to_load = file

        # Deserialize the object from the specified file
        with open(path_to_load, 'rb') as file:
            loaded_object = pickle.load(file)

        parent.GradObserverUI.hide()
        parent.dico = loaded_object
        parent.GradObserverUI = GradObserverWidget(parent.dico)
        parent.GradObserverUI.CommandAll(parent.MakeVisionWindow)

        parent.ScrollGradObserver.widget().layout().addWidget(parent.GradObserverUI, alignment=Qt.AlignCenter)
        local = os.path.split(file.name)[0]
        errorData = loadXmlAsObj(os.path.join(local, 'error'))
        parent.plotWindow.redraw(errorData, name=os.path.split(local)[-1])


def RunAnalyser(dico):
    a = QApplication([])
    w = ScenarioWindow(dico)
    w.show()
    a.exec()

if __name__ == '__main__':
    RunAnalyser(None)
