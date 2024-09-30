from PyQt5.QtWidgets import QToolButton, QWidget, QApplication, QHBoxLayout, QGridLayout, QPushButton, QGroupBox, QSizePolicy
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from GradObserver.QtRangeSlider import SliderEdit
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as mpatches
import os

local = os.path.join(os.path.abspath(__file__)[:(os.path.abspath(__file__).index('Projets'))], 'Projets')

class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.set_facecolor('gray')  # Set axes background color
        self.axes.xaxis.pane.fill = False
        self.axes.yaxis.pane.fill = False
        self.axes.zaxis.pane.fill = False
        self.axes.view_init(elev=10, azim=50)
        self.elev_ini = 10
        self.azim_ini = 50
        self.custom_legend = []
        fig.patch.set_facecolor('gray')  # Set figure background color
        super(MplCanvas, self).__init__(fig)

    def add_view(self, elev=None, azim=None):
        if elev is None:
            elev = 0
        if azim is None:
            azim = 0
        self.axes.azim += azim
        self.axes.elev += elev
        self.draw()

    def set_view(self, elev=None, azim=None):
        if elev is None:
            elev = self.axes.elev
        if azim is None:
            azim = self.axes.azim
        self.axes.azim = azim
        self.axes.elev = elev
        self.draw()

    def reset_view(self):
        self.axes.azim = self.azim_ini
        self.axes.elev = self.elev_ini
        self.draw()

    def clear_draw(self):
        self.axes.clear()
        self.draw()

    def set_legend(self, color_list, label_list):
        for i, color in enumerate(color_list):
            label = label_list[i]
            self.custom_legend.append(mpatches.Patch(color=color, label=label))

        self.axes.legend(handles=self.custom_legend, loc='upper right')

    def delete_legend(self, label):
        for legend in self.custom_legend:
            if legend._label == label:
                self.custom_legend.remove(legend)
        self.axes.legend(handles=self.custom_legend, loc='upper right')

    def add_legend(self, color, label):
        self.custom_legend.append(mpatches.Patch(color=color, label=label))
        self.axes.legend(handles=self.custom_legend, loc='upper right')

    def plot_hists(self, hist_list, bin_centers_list, pos_list=None, time_step=1, time_name='none', color='red'):
        for i, hist in enumerate(hist_list):
            bin_centers = bin_centers_list[i]
            if pos_list is not None:
                pos = pos_list[i]
            else:
                pos = i * time_step
            self.plot_hist(hist, bin_centers, pos, color=color, time_name=time_name)
        self.draw()


    def plot_hist(self, hist, bin_centers, pos, color='red', time_name='none'):
        bin_step = bin_centers[1] - bin_centers[0]
        self.axes.bar(bin_centers, hist, zs=pos, zdir='y', alpha=0.6, color=color,
                             label=time_name, width=bin_step, edgecolor='black')
        self.draw()


class DirectionalButton(QGroupBox):
    def __init__(self, canvas):
        super().__init__()
        self.elev_step = 5
        self.azim_step = 5
        self.leftarrow = QToolButton()
        self.leftarrow.setArrowType(Qt.LeftArrow)
        self.leftarrow.clicked.connect(lambda: canvas.add_view(azim=-self.azim_step))
        self.rightarrow = QToolButton()
        self.rightarrow.setArrowType(Qt.RightArrow)
        self.rightarrow.clicked.connect(lambda: canvas.add_view(azim=self.azim_step))
        self.uparrow = QToolButton()
        self.uparrow.setArrowType(Qt.UpArrow)
        self.uparrow.clicked.connect(lambda: canvas.add_view(elev=self.elev_step))
        self.downarrow = QToolButton()
        self.downarrow.setArrowType(Qt.DownArrow)
        self.downarrow.clicked.connect(lambda: canvas.add_view(elev=-self.elev_step))
        self.reloadButton = QPushButton(icon=QIcon(os.path.join(local, 'GradObserver', 'updateButton.png')))
        self.reloadButton.clicked.connect(canvas.reset_view)

        view_layout = QGridLayout()
        view_layout.addWidget(self.uparrow, 0, 1, 1, 1)
        view_layout.addWidget(self.leftarrow, 1, 0, 1, 1)
        view_layout.addWidget(self.reloadButton, 1, 1, 1, 1)
        view_layout.addWidget(self.rightarrow, 1, 2, 1, 1)
        view_layout.addWidget(self.downarrow, 2, 1, 1, 1)
        self.setLayout(view_layout)


class Histogram3DWidget(QWidget):
    def __init__(self, GradObUI, FusionWindow):
        super(Histogram3DWidget, self).__init__()
        self.GradObserver = GradObUI.GradObserver
        self.full_name = GradObUI.full_name

        # Set up the layout
        layout = QGridLayout()
        self.setLayout(layout)
        self.FusionWindow = FusionWindow

        # Create the Matplotlib canvas
        self.canvas = MplCanvas(width=8, height=6)
        self.nb_hist = 10

        self.RangeSlider = SliderEdit(min=0, max=self.GradObserver.time[-1])
        self.FusionButton = QToolButton()
        self.FusionButton.setText("Projeter")
        self.FusionButton.setFont(QFont('Times', 15))
        self.FusionButton.setCheckable(True)
        self.FusionButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.FusionButton.toggled.connect(self.FusionAction)
        self.CloseButton = QPushButton(icon=QIcon(os.path.join(local, 'GradObserver', 'C3.webp')))

        self.view_box = DirectionalButton(self.canvas)
        layout.addWidget(self.canvas, 0, 0, 1, 3)
        layout.addWidget(self.CloseButton, 0, 2, 1, 1, alignment=Qt.AlignRight | Qt.AlignTop)
        widget = QWidget()
        layout2 = QGridLayout()
        layout2.addWidget(self.RangeSlider, 0, 0, 1, 3)
        layout2.addWidget(self.FusionButton, 1, 0, 1, 3)
        layout2.addWidget(self.view_box, 0, 3, 2, 2, alignment=Qt.AlignRight)
        widget.setLayout(layout2)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(widget, 1, 0, 1, 3)
        self.view_box.reloadButton.clicked.connect(lambda: self.Replot_data(100))
        self.plot_data()
        self.RangeSlider.updated.connect(lambda : self.Replot_data(self.RangeSlider.rangeslider.first_position, self.RangeSlider.rangeslider.second_position))

    def plot_data(self):
        if self.nb_hist > len(self.GradObserver.time):
            self.hist_list = np.array(self.GradObserver.freq_save)
            bins_list = np.array(self.GradObserver.bins_save)
            self.bin_centers_list = (bins_list[:, 1:] + bins_list[:, :-1]) / 2
            self.time_list = self.GradObserver.time
            self.canvas.plot_hists(self.hist_list, self.bin_centers_list, self.time_list)
        else:
            id = np.arange(self.nb_hist)
            id = id * len(self.GradObserver.time) / self.nb_hist
            id = id.round().astype(np.int)
            self.hist_list = np.array(self.GradObserver.freq_save)[id]
            bins_list = np.array(self.GradObserver.bins_save)
            self.bin_centers_list = ((bins_list[:, 1:] + bins_list[:, :-1]) / 2)[id]
            self.time_list = list(np.array(self.GradObserver.time)[id])
            self.canvas.plot_hists(self.hist_list, self.bin_centers_list, self.time_list)

    def FusionAction(self, checked):
        if checked:
            self.FusionWindow.add_hist(self)
        else:
            self.FusionWindow.del_hist(self)

    def Replot_data(self, min, max):
        id = np.arange(self.nb_hist) / self.nb_hist
        id = (id * (max-min) + min) * (len(self.GradObserver.time) / self.GradObserver.time[-1])
        id = id.round().astype(np.int)
        self.hist_list = np.array(self.GradObserver.freq_save)[id]
        bins_list = np.array(self.GradObserver.bins_save)
        self.bin_centers_list = ((bins_list[:, 1:] + bins_list[:, :-1]) / 2)[id]
        self.time_list = list(np.array(self.GradObserver.time)[id])
        self.canvas.clear_draw()
        self.canvas.plot_hists(self.hist_list, self.bin_centers_list, self.time_list)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = Histogram3DWidget()
    mainWin.show()
    sys.exit(app.exec_())
