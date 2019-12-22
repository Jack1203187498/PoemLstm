from EmbeddingLSTM import *
import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sip
import PyQt5.sip
import random

class Ui_PassageMade(QWidget):
    def __init__(self):
        # # noinspection PyArgumentList
        super(Ui_PassageMade, self).__init__()
        self.originalNumber()

    def originalNumber(self):
        self.col = 16
        self.row = 16
        self.labels = list()
        self.start0 = 0
        self.mines = 40
        self.alreadyShow = 0 #已经开启的方格数量

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setFixedSize(500, 80)  # 固定窗体大小
        Form.move(400, 200)
        Form.setWindowIcon(QIcon(''))  # 设置窗体图标

        self.gridLayoutWidget = QtWidgets.QWidget(Form)  # 设置布局
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 80, 700, 700))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setSpacing(2)

        self.inputRow = QtWidgets.QSpinBox(Form)
        self.inputRow.setObjectName("inputRow")
        self.inputRow.setGeometry(20, 30, 40, 20)

        self.inputCol = QtWidgets.QSpinBox(Form)
        self.inputCol.setObjectName("inputCol")
        self.inputCol.setGeometry(80, 30, 40, 20)

        self.inputMine = QtWidgets.QSpinBox(Form)
        self.inputMine.setObjectName("inputMine")
        self.inputMine.setGeometry(140, 30, 40, 20)

        self.startButton = QtWidgets.QPushButton(Form)
        self.startButton.setObjectName("startButton")
        self.startButton.setGeometry(200, 30, 40, 20)

        self.info = QtWidgets.QLabel(Form)
        self.info.setObjectName("info")
        self.info.setGeometry(250, 15, 200, 40)

        # 一些文字信息的显示
        self.retranslateUi(Form)

        Form.show()

        self.startButton.clicked.connect(lambda: self.loadGame())

    def retranslateUi(self, Form):#游戏开始前的提示文字
        _translate = QtCore.QCoreApplication.translate
        #设置窗体标题
        Form.setWindowTitle(_translate("Form", "你想要怎样的雷区呢？"))
        self.startButton.setText(_translate("Form", "开始"))
        self.info.setText(_translate("Form", "请设置雷区大小\n默认为16*16, 40个雷"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_PassageMade()
    ui.setupUi(Form)
    sys.exit(app.exec_())