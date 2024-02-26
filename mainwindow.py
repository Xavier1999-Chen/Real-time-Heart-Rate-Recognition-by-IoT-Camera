from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(1308, 850)
        MainWindow.resize(1650, 1150)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: #F8F8FF;") # 设置背景颜色
        self.centralwidget.setObjectName("centralwidget")
        
        # camera
        self.face = QtWidgets.QLabel(self.centralwidget)
        self.face.setGeometry(QtCore.QRect(750, 10, 881, 591))
        self.face.setText("")
        self.face.setObjectName("face")
        # BVP signal
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 660, 731, 431))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.Layout_BVP = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.Layout_BVP.setContentsMargins(0, 0, 0, 0)
        self.Layout_BVP.setObjectName("Layout_BVP")
        # BPM Data
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 701, 560))# BPM text window size
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.Layout_button = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.Layout_button.setContentsMargins(0, 0, 0, 0)
        self.Layout_button.setObjectName("Layout_button")
        
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.comboBox.setMinimumSize(QtCore.QSize(0, 28))
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 28))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        self.Layout_button.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Button_RawTrue = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.Button_RawTrue.setObjectName("Button_RawTrue")
        self.horizontalLayout_2.addWidget(self.Button_RawTrue)
        self.Button_RawFalse = QtWidgets.QPushButton()
        self.Button_RawFalse.setObjectName("Button_RawFalse")
        self.horizontalLayout_2.addWidget(self.Button_RawFalse)
#
        self.Button_MqttFalse = QtWidgets.QPushButton()
        self.Button_MqttFalse.setObjectName("Button_MqttFalse")
        self.horizontalLayout_2.addWidget(self.Button_MqttFalse)
        self.Button_MqttTrue = QtWidgets.QPushButton()
        self.Button_MqttTrue.setObjectName("Button_MqttTrue")
        self.horizontalLayout_2.addWidget(self.Button_MqttTrue)
#
        self.Button_Shutdown = QtWidgets.QPushButton()
        self.Button_Shutdown.setObjectName("Exit")
        self.horizontalLayout_2.addWidget(self.Button_Shutdown)
#
        self.Layout_button.addLayout(self.horizontalLayout_2)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label.setMinimumSize(QtCore.QSize(0, 300))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(16) # BPM text font size
        self.label.setFont(font)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.Layout_button.addWidget(self.label)


        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(1140, 660, 461, 431))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.Layout_Signal = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.Layout_Signal.setContentsMargins(0, 0, 0, 0)
        self.Layout_Signal.setObjectName("Layout_Signal")
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(750, 660, 381, 431))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.Layout_Spec = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.Layout_Spec.setContentsMargins(0, 0, 0, 0)
        self.Layout_Spec.setObjectName("Layout_Spec")


        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "GREEN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "GREEN-RED"))
        self.comboBox.setItemText(2, _translate("MainWindow", "CHROM"))
        self.comboBox.setItemText(3, _translate("MainWindow", "PBV"))
        self.comboBox.setItemText(4, _translate("MainWindow", "LGI"))
        self.comboBox.setItemText(5, _translate("MainWindow", "POS"))
        self.comboBox.setItemText(6, _translate("MainWindow", "ICA-POH"))
        self.comboBox.setItemText(7, _translate("MainWindow", "PhysFormer"))
        self.comboBox.setItemText(8, _translate("MainWindow", "TSCAN"))
        self.Button_RawTrue.setText(_translate("MainWindow", "原始信号"))
        self.Button_RawFalse.setText(_translate("MainWindow", "滤波信号"))
        self.Button_MqttTrue.setText(_translate("MainWindow", "远程相机"))
        self.Button_MqttFalse.setText(_translate("MainWindow", "本地相机"))
        self.Button_Shutdown.setText(_translate("MainWindow", "退出程序"))
