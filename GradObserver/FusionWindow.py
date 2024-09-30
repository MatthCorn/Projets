from PyQt5.QtWidgets import QToolButton, QWidget, QApplication, QGridLayout, QPushButton, QGroupBox, QSizePolicy, QHBoxLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
import sys
from GradObserver.VisualGradWindow import MplCanvas, DirectionalButton
import os

class Fusion3DWidget(QGroupBox):
    def __init__(self):
        super(Fusion3DWidget, self).__init__()
        # Set up the layout
        layout = QGridLayout()
        self.setLayout(layout)
        self.connectedWindows = []
        self.color_connect = [['red', None], ['blue', None], ['green', None], ['yellow', None], ['orange', None], ['purple', None]]

        # Create the Matplotlib canvas
        self.canvas = MplCanvas(width=8, height=6)

        self.view_box = DirectionalButton(self.canvas)

        self.ResetButton = QToolButton()
        self.ResetButton.setText("réinitialiser")
        self.ResetButton.setFont(QFont('Times', 15))
        self.ResetButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ResetButton.clicked.connect(self.resetButtonAction)
        layout.addWidget(self.canvas, 0, 0, 1, 3)
        widget = QWidget()
        layout2 = QHBoxLayout()
        layout2.addWidget(self.ResetButton)
        layout2.addWidget(self.view_box, alignment=Qt.AlignRight)
        widget.setLayout(layout2)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(widget, 1, 0, 1, 3)

        self.canvas.axes.grid(color='black')
        self.canvas.axes.set_xlabel('amplitude')
        self.canvas.axes.set_ylabel('iteration')
        self.canvas.axes.set_zlabel('fréquence')
        self.canvas.axes.set_title("distribution de l'amplitude du gradient \n au cours de l'apprentissage")
        self.canvas.draw()

    def add_hist(self, window, color=None):
        hist_list = window.hist_list
        bin_centers_list = window.bin_centers_list
        time_list = window.time_list
        if color is None:
            for duo in self.color_connect:
                color = duo[0]
                if duo[1] is None:
                    duo[1] = window.full_name
                    break
        self.connectedWindows.append(window)
        self.canvas.add_legend(color, window.full_name)
        self.canvas.plot_hists(hist_list, bin_centers_list, pos_list=time_list, color=color)

    def del_hist(self, window):
        self.connectedWindows.remove(window)
        self.canvas.delete_legend(window.full_name)
        for duo in self.color_connect:
            color = duo[0]
            if duo[1] == window.full_name:
                duo[1] = None
                break
        self.canvas.clear_draw()
        for window in self.connectedWindows:
            hist_list = window.hist_list
            bin_centers_list = window.bin_centers_list
            time_list = window.GradObserver.time
            for color, name in self.color_connect:
                if name == window.full_name:
                    break
            self.canvas.axes.legend(handles=self.canvas.custom_legend, loc='upper right')
            self.canvas.plot_hists(hist_list, bin_centers_list, pos_list=time_list, color=color)

    def resetButtonAction(self):
        while self.connectedWindows != []:
            window = self.connectedWindows[0]
            window.FusionButton.setChecked(False)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = Fusion3DWidget()
    mainWin.show()
    sys.exit(app.exec_())
