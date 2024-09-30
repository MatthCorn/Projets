from PyQt5.QtWidgets import (QApplication, QSizePolicy, QWidget, QSplitter, QTextEdit, QScrollArea, QVBoxLayout, QLabel, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
import qdarkstyle

class GradObserverWidget(QWidget):

    def __init__(self, GradObserver, ini='folded', name='origin', parent_full_name=''):
        super().__init__()
        self.GradObserver = GradObserver
        self.full_name = parent_full_name + '.' + name
        self.parent_full_name = parent_full_name

        if GetType(GradObserver) == 'GradObserver':
            self.Final = True
            self.PrintButton = QPushButton('print ' + name)
            self.PrintButton.clicked.connect(lambda: self.OwnCommand(self))
            layout = QVBoxLayout()
            layout.addWidget(self.PrintButton, alignment=Qt.AlignCenter)
            self.setLayout(layout)

        elif GetType(GradObserver) in ['dict', 'DictGradObserver']:
            self.Final = False
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

            self.stylesheet = qdarkstyle.load_stylesheet_pyqt5()
            self.setStyleSheet(self.stylesheet)

            self.form = ini
            self.setLayout(QVBoxLayout())

            self.foldedGBox = QGroupBox(name)
            self.foldedLayout = QVBoxLayout()
            self.foldedLayout.setAlignment(Qt.AlignCenter)
            self.unfoldButton = QPushButton('unfold')
            self.unfoldButton.clicked.connect(self.switch)
            self.foldedLayout.addWidget(self.unfoldButton)
            self.foldedGBox.setLayout(self.foldedLayout)

            self.unfoldedGBox = QGroupBox(name)
            self.unfoldedLayout = QVBoxLayout()
            self.unfoldedLayout.setAlignment(Qt.AlignCenter)
            self.foldButton = QPushButton('fold')
            self.foldButton.clicked.connect(self.switch)
            self.unfoldedLayout.addWidget(self.foldButton)
            self.MakeUnfoldedLayout()
            self.unfoldedGBox.setLayout(self.unfoldedLayout)

            self.layout().addWidget(self.unfoldedGBox)
            self.layout().addWidget(self.foldedGBox)
            self.layout().setAlignment(Qt.AlignCenter)

            if ini == 'unfolded':
                self.foldedGBox.hide()
            elif ini == 'folded':
                self.unfoldedGBox.hide()

        else:
            self.Final = True
            self.Label = QLabel(name)
            self.Label.setStyleSheet("border: 0.5px solid dimgray; border-radius: 3px;")
            self.Label.setAlignment(Qt.AlignCenter)
            layout = QVBoxLayout()
            layout.addWidget(self.Label)
            self.setLayout(layout)
            return



    # On donne une fonction à chaque GradObserver terminal quelque soit sa position
    def CommandAll(self, command):
        if self.Final:
            self.OwnCommand = command
        else:
            for i in range(1, self.unfoldedLayout.count()):
                widget = self.unfoldedLayout.itemAt(i).widget()
                widget.CommandAll(command)

    def MakeUnfoldedLayout(self):
        for key in self.GradObserver.keys():
            item = self.GradObserver[key]

            self.unfoldedLayout.addWidget(GradObserverWidget(item, name=key, parent_full_name=self.full_name), alignment=Qt.AlignCenter)


    def switch(self):
        if self.form == 'folded':
            self.form = 'unfolded'
            self.foldedGBox.hide()
            self.unfoldedGBox.show()

        elif self.form == 'unfolded':
            self.form = 'folded'
            self.unfoldedGBox.hide()
            self.foldedGBox.show()


def GetType(module):
    return str(type(module)).split("'")[1].split(".")[-1]

if __name__ == '__main__':
    a = QApplication([])
    dico = {'r': {'t': {'v': 4, 'w': (1, 2)}, 'x': [2]}, 'u': {8}}
    w = GradObserverWidget(dico, ini='unfolded')
    w.show()
    a.exec()