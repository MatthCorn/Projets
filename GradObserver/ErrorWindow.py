from PyQt5.QtWidgets import QGroupBox, QGridLayout, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from Tools.XMLTools import loadXmlAsObj
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

class ErrorWindowWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__()

        self.setTitle('Error display')
        self.graphWidget = pg.PlotWidget(parent=parent)
        self.legend = self.graphWidget.addLegend()
        self.legend.setParentItem(self.graphWidget.graphicsItem())
        self.legend.anchor((1, 1), (1, 1))
        self.graphWidget.setBackground((69, 83, 100))
        self.graphWidget.showGrid(x=False, y=True, alpha=0.5)
        self.TrainingButton = QPushButton('Training')
        self.TrainingButton.setCheckable(True)
        self.TrainingButton.setChecked(True)
        self.TrainingButton.clicked.connect(self.TrainButAct)
        self.ValidationButton = QPushButton('Validation')
        self.ValidationButton.setCheckable(True)
        self.ValidationButton.setChecked(True)
        self.ValidationButton.clicked.connect(self.ValButAct)
        self.VSButton = QPushButton('Comparaison')
        self.VSButton.clicked.connect(self.CompButAct)
        self.reloadButton = QPushButton(icon=QIcon(os.path.join(local, 'GradObserver', 'updateButton.png')))
        self.reloadButton.clicked.connect(self.redraw)
        self.reloadButton.setMaximumWidth(30)
        layout = QGridLayout()
        layout.addWidget(self.graphWidget, 0, 0, 6, 6)
        layout.addWidget(self.TrainingButton, 6, 0, 1, 2)
        layout.addWidget(self.ValidationButton, 6, 2, 1, 2)
        layout.addWidget(self.VSButton, 6, 4, 1, 2)
        layout.addWidget(self.reloadButton, 0, 5, 1, 1)
        self.setLayout(layout)
        self.Plot = []
        self.color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.TrainShown = True
        self.ValShown = True
        self.main_data = None
        self.main_name = ''


    def update(self, data, name=''):
        R, G, B = self.color[len(self.Plot)]
        Training = data['Training']
        Validation = data['Validation']
        T = pg.PlotCurveItem()
        T.setData(Training, pen={'color': (R, G, B, 255), 'width': 2})
        V = pg.PlotCurveItem()
        V.setData(Validation, pen={'color': (R, G, B, 127), 'width': 2, 'style': Qt.DotLine})
        self.Plot.append({'train': T, 'val': V, 'name': name})
        if self.TrainShown:
            self.graphWidget.addItem(T)
            self.legend.addItem(T, 'training ' + name)
        if self.ValShown:
            self.graphWidget.addItem(V)
            self.legend.addItem(V, 'validation ' + name)

    def redraw(self, data=None, name=''):
        if type(data) is dict:
            self.main_data = data
            self.main_name = name
        self.graphWidget.clear()
        self.Plot = []
        if type(self.main_data) is dict:
            self.update(self.main_data, name=self.main_name)

    def TrainButAct(self):
        if self.TrainShown:
            self.TrainShown = False

            for P in self.Plot:
                T = P['train']
                self.graphWidget.removeItem(T)

        else:
            self.TrainShown = True
            for P in self.Plot:
                T = P['train']
                self.graphWidget.addItem(T)
                self.legend.addItem(T, 'training ' + P['name'])

    def ValButAct(self):
        if self.ValShown:
            self.ValShown = False

            for P in self.Plot:
                V = P['val']
                self.graphWidget.removeItem(V)

        else:
            self.ValShown = True
            for P in self.Plot:
                V = P['val']
                self.graphWidget.addItem(V)
                self.legend.addItem(V, 'validation ' + P['name'])

    def CompButAct(self):
        file, check = QFileDialog.getOpenFileName(None, "Load error", "")

        if check:
            data = loadXmlAsObj(file)

            name = os.path.split(os.path.split(file)[0])[-1]
            self.update(data, name=name)

