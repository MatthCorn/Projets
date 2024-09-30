from PyQt5.QtWidgets import QLineEdit

'''Classe héritant de QLineEdit dont toutes les propriétés de base restent 
excepté le fait qu'on ne peut pas modifier la chaine de caractère depuis le clavier'''

class UneditableLineEdit(QLineEdit):
    def __init__(self, text='', parent=None):
        super().__init__(text, parent=parent)
        self.__Editable = False
        self.textChanged.connect(lambda: self.__Editable or self.undo())
    
    def setText(self, txt):
        self.__Editable = True
        super().setText(txt)
        self.__Editable = False

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout
    import random
    
    a = QApplication([])
    w = QWidget()
    layout = QHBoxLayout()
    l = UneditableLineEdit(text='text initial', parent=w)
    p = QPushButton(w)
    p.clicked.connect(lambda: l.setText(str(random.randint(0, 10))))
    layout.addWidget(l)
    layout.addWidget(p)
    w.setLayout(layout)
    w.show()
    a.exec()