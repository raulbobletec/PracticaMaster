# This Python file uses the following encoding: utf-8
import sys

from PyQt5.QtGui import QImage, QPixmap

import retea as r
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1152, 648)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # self.buttonAfisare = QtWidgets.QPushButton(self.centralwidget)
        # self.buttonAfisare.setGeometry(QtCore.QRect(640, 380, 140, 25))
        # self.buttonAfisare.setMinimumSize(QtCore.QSize(140, 0))
        # self.buttonAfisare.setObjectName("buttonAfisare")

        self.buttonLearnHOG = QtWidgets.QPushButton(self.centralwidget)
        self.buttonLearnHOG.setGeometry(QtCore.QRect(710, 480, 140, 25))
        self.buttonLearnHOG.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonLearnHOG.setObjectName("buttonLearnHOG")

        self.buttonTestHOG = QtWidgets.QPushButton(self.centralwidget)
        self.buttonTestHOG.setGeometry(QtCore.QRect(990, 480, 140, 25))
        self.buttonTestHOG.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonTestHOG.setObjectName("buttonTestHOG")

        self.buttonClasificareHOG = QtWidgets.QPushButton(self.centralwidget)
        self.buttonClasificareHOG.setGeometry(QtCore.QRect(850, 480, 140, 25))
        self.buttonClasificareHOG.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonClasificareHOG.setObjectName("buttonClasificareHOG")

        self.buttonLearnFAST = QtWidgets.QPushButton(self.centralwidget)
        self.buttonLearnFAST.setGeometry(QtCore.QRect(710, 510, 140, 25))
        self.buttonLearnFAST.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonLearnFAST.setObjectName("buttonLearnFAST")

        self.buttonTestFAST = QtWidgets.QPushButton(self.centralwidget)
        self.buttonTestFAST.setGeometry(QtCore.QRect(990, 510, 140, 25))
        self.buttonTestFAST.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonTestFAST.setObjectName("buttonTestFAST")

        self.buttonClasificareFAST = QtWidgets.QPushButton(self.centralwidget)
        self.buttonClasificareFAST.setGeometry(QtCore.QRect(850, 510, 140, 25))
        self.buttonClasificareFAST.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonClasificareFAST.setObjectName("buttonClasificareFAST")

        self.buttonLearnXception = QtWidgets.QPushButton(self.centralwidget)
        self.buttonLearnXception.setGeometry(QtCore.QRect(710, 540, 140, 25))
        self.buttonLearnXception.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonLearnXception.setObjectName("buttonLearnXception")

        self.buttonTestXception = QtWidgets.QPushButton(self.centralwidget)
        self.buttonTestXception.setGeometry(QtCore.QRect(990, 540, 140, 25))
        self.buttonTestXception.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonTestXception.setObjectName("buttonTestXception")

        self.buttonClasificareXception = QtWidgets.QPushButton(self.centralwidget)
        self.buttonClasificareXception.setGeometry(QtCore.QRect(850, 540, 140, 25))
        self.buttonClasificareXception.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonClasificareXception.setObjectName("buttonClasificareXception")

        self.buttonincarcareImagine = QtWidgets.QPushButton(self.centralwidget)
        self.buttonincarcareImagine.setGeometry(QtCore.QRect(850, 450, 140, 25))
        self.buttonincarcareImagine.setMinimumSize(QtCore.QSize(140, 0))
        self.buttonincarcareImagine.setObjectName("buttonincarcareImagine")

        self.textEditEvenimente = QtWidgets.QTextEdit(self.centralwidget)
        self.textEditEvenimente.setGeometry(QtCore.QRect(70, 430, 561, 161))
        self.textEditEvenimente.setObjectName("textEditEvenimente")
        self.textEditEvenimente.setReadOnly(True)

        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(290, 400, 81, 17))
        self.label_1.setObjectName("label_1")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1152, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.server_process = None
        self.buttonLearnHOG.clicked.connect(r.invatareHog)
        self.buttonLearnFAST.clicked.connect(r.invatareFast)
        self.buttonTestHOG.clicked.connect(r.testareHog)
        self.buttonClasificareHOG.clicked.connect(r.clasificareHog)
        self.buttonClasificareFAST.clicked.connect(r.clasificareFast)
        self.buttonTestFAST.clicked.connect(r.testareFast)
        self.buttonincarcareImagine.clicked.connect(r.fileDialog)
        self.buttonLearnXception.clicked.connect(r.invatareXception)
        self.buttonTestXception.clicked.connect(r.testareXception)
        self.buttonClasificareXception.clicked.connect(r.clasificareXception)
        #self.buttonAfisare.clicked.connect(self.afisareImg)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # def afisareImg(self):
    #     filename = QFileDialog.getOpenFileName()
    #     path = filename[0]
    #     my_image = QImage(path)
    #     self.label_1.setPixmap(QPixmap.fromImage(my_image))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AI"))
        self.buttonLearnHOG.setText(_translate("MainWindow", "Invatare HOG"))
        self.buttonTestHOG.setText(_translate("MainWindow", "Testare HOG"))
        self.buttonClasificareHOG.setText(_translate("MainWindow", "Clasificare HOG"))
        self.buttonLearnFAST.setText(_translate("MainWindow", "Invatare FAST"))
        self.buttonTestFAST.setText(_translate("MainWindow", "Testare FAST"))
        self.buttonClasificareFAST.setText(_translate("MainWindow", "Clasificare FAST"))
        self.buttonLearnXception.setText(_translate("MainWindow", "Invatare Xception"))
        self.buttonTestXception.setText(_translate("MainWindow", "Testare Xception"))
        self.buttonClasificareXception.setText(_translate("MainWindow", "Clasificare Xception"))
        self.buttonincarcareImagine.setText(_translate("MainWindow", "Incarcare imagine"))
        #self.buttonAfisare.setText(_translate("MainWindow", "Afisare imagine"))

        self.label_1.setText(_translate("MainWindow", "Evenimente"))

class runThreads(object):


    def invatareHog(self):
        self.thread1 = QtCore.QThread()
        self.rulare = r.invatareHog()
        self.rulare.moveToThread(self.thread1)
        self.thread1.started.connect(self.rulare.run)
        self.thread1.start()
        self.thread1.quit()
        self.thread1.wait()


if __name__ == "__main__":
    app = QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
