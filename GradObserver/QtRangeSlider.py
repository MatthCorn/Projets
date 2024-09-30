from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

class SliderEdit(QWidget):
    updated = pyqtSignal()

    def __init__(self, min=0, max=10, parent=None):
        super().__init__(parent)
        self.first_position_edit = QLineEdit(str(min))
        self.first_position_edit.setAlignment(Qt.AlignCenter)
        self.first_position_edit.setMaximumWidth(40)
        self.second_position_edit = QLineEdit(str(max))
        self.second_position_edit.setAlignment(Qt.AlignCenter)
        self.second_position_edit.setMaximumWidth(40)
        self.rangeslider = RangeSlider(min, max, parent=self)
        self.rangeslider.setMinimumWidth(200)
        layout = QHBoxLayout(self)
        layout.addWidget(self.first_position_edit)
        layout.addWidget(self.rangeslider)
        layout.addWidget(self.second_position_edit)

        self.first_position_edit.editingFinished.connect(self.first_edit)
        self.second_position_edit.editingFinished.connect(self.second_edit)
        self.rangeslider.sign.connect(self.slider_edit)

    def first_edit(self):
        if self.first_position_edit.text() == '':
            self.rangeslider.first_position = self.rangeslider.opt.minimum
            self.rangeslider.update()
            return

        nb2 = int(self.second_position_edit.text())
        try:
            nb1 = int(self.first_position_edit.text())
            if nb1 > nb2:
                self.first_position_edit.undo()
        except:
            self.first_position_edit.undo()
            return

        self.rangeslider.first_position = nb1
        self.rangeslider.update()
        self.updated.emit()

    def second_edit(self):
        if self.second_position_edit.text() == '':
            self.rangeslider.second_position = self.rangeslider.opt.maximum
            self.rangeslider.update()
            return

        nb1 = int(self.first_position_edit.text())
        try:
            nb2 = int(self.second_position_edit.text())
            if nb2 < nb1:
                self.second_position_edit.undo()
        except:
            self.second_position_edit.undo()
            return

        self.rangeslider.second_position = nb2
        self.rangeslider.update()
        self.updated.emit()

    def slider_edit(self):
        self.first_position_edit.setText(str(self.rangeslider.first_position))
        self.second_position_edit.setText(str(self.rangeslider.second_position))
        self.updated.emit()

class RangeSlider(QWidget):
    sign = pyqtSignal()

    def __init__(self, min=0, max=10, parent=None):
        super().__init__(parent)
        self.first_position = min
        self.second_position = max

        self.opt = QStyleOptionSlider()
        self.opt.minimum = min
        self.opt.maximum = max

        self.setTickPosition(QSlider.TicksAbove)
        self.setTickInterval(int((max-min)/10))

        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider)
        )

    def setRangeLimit(self, minimum: int, maximum: int):
        self.opt.minimum = minimum
        self.opt.maximum = maximum

    def setRange(self, start: int, end: int):
        self.first_position = start
        self.second_position = end

    def getRange(self):
        return (self.first_position, self.second_position)

    def setTickPosition(self, position: QSlider.TickPosition):
        self.opt.tickPosition = position

    def setTickInterval(self, ti: int):
        self.opt.tickInterval = ti

    def paintEvent(self, event: QPaintEvent):

        painter = QPainter(self)

        # Draw rule
        self.opt.initFrom(self)
        self.opt.rect = self.rect()
        self.opt.sliderPosition = 0
        self.opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderTickmarks

        #   Draw GROOVE
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        self.opt.sliderPosition = self.first_position
        self.opt.sliderPosition = self.second_position

        # Draw first handle

        self.opt.subControls = QStyle.SC_SliderHandle
        self.opt.sliderPosition = self.first_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

        # Draw second handle
        self.opt.sliderPosition = self.second_position
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)


    def mousePressEvent(self, event: QMouseEvent):

        self.opt.sliderPosition = self.first_position
        self._first_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, event.pos(), self
        )

        self.opt.sliderPosition = self.second_position
        self._second_sc = self.style().hitTestComplexControl(
            QStyle.CC_Slider, self.opt, event.pos(), self
        )

    def mouseMoveEvent(self, event: QMouseEvent):

        distance = self.opt.maximum - self.opt.minimum

        pos = self.style().sliderValueFromPosition(
            0, distance, event.pos().x(), self.rect().width()
        )

        if self._first_sc == QStyle.SC_SliderHandle:
            if pos <= self.second_position:
                self.first_position = pos
                self.update()
                self.sign.emit()
                return

        if self._second_sc == QStyle.SC_SliderHandle:
            if pos >= self.first_position:
                self.second_position = pos
                self.update()
                self.sign.emit()

    def sizeHint(self):
        """ override """
        SliderLength = 84
        TickSpace = 5

        w = SliderLength
        h = self.style().pixelMetric(QStyle.PM_SliderThickness, self.opt, self)

        if (
            self.opt.tickPosition & QSlider.TicksAbove
            or self.opt.tickPosition & QSlider.TicksBelow
        ):
            h += TickSpace

        return (
            self.style()
            .sizeFromContents(QStyle.CT_Slider, self.opt, QSize(w, h), self)
            .expandedTo(QApplication.globalStrut())
        )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    w = SliderEdit(min=0, max=800)
    w.show()

    # q = QSlider()
    # q.show()

    app.exec_()