import sys
from mainwindow import Ui_MainWindow
import paho.mqtt.client as mqtt
import time

from PyQt5.QtWidgets import QMainWindow, QApplication

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import pyqtgraph as pg


from scipy import signal
import numpy as np
import cv2 as cv
from series2rPPG import Series2rPPG

MIN_HZ = 0.83       # 50 BPM - minimum allowed heart rate
MAX_HZ = 2.5       # 150 BPM - maximum allowed heart rate


class mainwin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainwin, self).__init__(parent)
        self.setupUi(self)

        # 直方图
        self.Hist_fore = pg.PlotWidget(self)
        self.Hist_left = pg.PlotWidget(self)
        self.Hist_right = pg.PlotWidget(self)

        self.Hist_fore.setBackground(None)  # 设置背景色为透明
        self.Hist_left.setBackground(None)
        self.Hist_right.setBackground(None)

        self.Hist_fore.setYRange(0, 0.2)
        self.Hist_left.setYRange(0, 0.2)
        self.Hist_right.setYRange(0, 0.2)

        self.label_fore = QLabel(self.verticalLayoutWidget)
        self.label_left = QLabel(self.verticalLayoutWidget)
        self.label_right = QLabel(self.verticalLayoutWidget)
        self.Hist_fore_r = self.Hist_fore.plot()
        self.Hist_fore_g = self.Hist_fore.plot()
        self.Hist_fore_b = self.Hist_fore.plot()
        self.Hist_left_r = self.Hist_left.plot()
        self.Hist_left_g = self.Hist_left.plot()
        self.Hist_left_b = self.Hist_left.plot()
        self.Hist_right_r = self.Hist_right.plot()
        self.Hist_right_g = self.Hist_right.plot()
        self.Hist_right_b = self.Hist_right.plot()
        self.Layout_Signal.addWidget(self.Hist_fore)
        self.Layout_Signal.addWidget(self.Hist_left)
        self.Layout_Signal.addWidget(self.Hist_right)

        # 波形图
        self.Signal_fore = pg.PlotWidget(self)
        self.Signal_left = pg.PlotWidget(self)
        self.Signal_right = pg.PlotWidget(self)

        self.Signal_fore.setBackground(None)  # 设置背景色为透明
        self.Signal_left.setBackground(None)
        self.Signal_right.setBackground(None)  

        self.Sig_f = self.Signal_fore.plot()
        self.Sig_l = self.Signal_left.plot()
        self.Sig_r = self.Signal_right.plot()

        # 频谱图
        self.Spectrum_fore = pg.PlotWidget(self)
        self.Spectrum_left = pg.PlotWidget(self)
        self.Spectrum_right = pg.PlotWidget(self)

        self.Spectrum_fore.setBackground(None) 
        self.Spectrum_left.setBackground(None)
        self.Spectrum_right.setBackground(None)

        self.Spec_f = self.Spectrum_fore.plot()
        self.Spec_l = self.Spectrum_left.plot()
        self.Spec_r = self.Spectrum_right.plot()
        

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        # palette = QPalette()
        # palette.setColor(QPalette.WindowText, Qt.black)  # 设置字体颜色为红色


        font.setWeight(75)

        self.label_fore.setFont(font)
        self.label_fore.setText("Forehead Signal")
        # self.label_fore.setPalette(palette)
        self.Layout_BVP.addWidget(self.label_fore)

        self.Layout_BVP.addWidget(self.Signal_fore)

        self.label_left.setFont(font)
        self.label_left.setText("Left Cheek Signal")
        self.Layout_BVP.addWidget(self.label_left)

        self.Layout_BVP.addWidget(self.Signal_left)

        self.label_right.setFont(font)
        self.label_right.setText("Right Cheek Signal")
        self.Layout_BVP.addWidget(self.label_right)

        self.Layout_BVP.addWidget(self.Signal_right)

        self.Layout_Spec.addWidget(self.Spectrum_fore)
        self.Layout_Spec.addWidget(self.Spectrum_left)
        self.Layout_Spec.addWidget(self.Spectrum_right)

        self.face.setScaledContents(True)
        self.processor = Series2rPPG()
        self.processor.PROCESS_start()

        self.TIMER_Frame = QTimer()
        self.TIMER_Frame.setInterval(100)
        self.TIMER_Frame.start()

        self.TIMER_Hist = QTimer()
        self.TIMER_Hist.setInterval(100)
        self.TIMER_Hist.start()

        self.TIMER_SIGNAL = QTimer()
        self.TIMER_SIGNAL.setInterval(100)
        self.TIMER_SIGNAL.start()

        self.bpm_fore = 60
        self.bpm_left = 60
        self.bpm_right = 60
        self.bpm_avg = 60

        ##################
        self.pre_mode='GREEN'
        self.Mode_str ='GREEN'
        self.unsupervised_model=['GREEN','GREEN-RED','CHROM','PBV','LGI','POS','ICA-POH']

        self.PhysFormer_frames=None
        self.update_physFormer=0
        self.showed_physFormer=0
        self.PhysFormer=QTimer()
        self.PhysFormer.setInterval(25000)
        self.PhysFormer.timeout.connect(self.cal_physformer)

        self.TSCAN_frames=None
        self.update_TSCAN=0
        self.showed_TSCAN=0
        self.TSCAN=QTimer()
        self.TSCAN.setInterval(30000)
        self.TSCAN.timeout.connect(self.cal_TSCAN)
        ###################



        self.ModeDict = {'GREEN': self.processor.GREEN,
                         'GREEN-RED': self.processor.GREEN_RED, 'CHROM': self.processor.CHROM,
                         'PBV': self.processor.PBV, 'LGI': self.processor.LGI, 'POS': self.processor.POS,
                         'ICA-POH': self.processor.ICA_POH,'PhysFormer':True,'TSCAN':True}
        self.Mode = self.ModeDict['GREEN']
        self.Data_ShowRaw = True
        self.url = None
        self.message_received = False
        self.slot_init()

    def slot_init(self):
        self.TIMER_Frame.timeout.connect(self.DisplayImage)
        self.TIMER_Hist.timeout.connect(self.DisplayHist)
        self.TIMER_SIGNAL.timeout.connect(self.DisplaySignal)
        self.comboBox.activated[str].connect(self.Button_ChangeMode)
        self.Button_RawTrue.clicked.connect(self.Button_Data_RawTrue)
        self.Button_RawFalse.clicked.connect(self.Button_Data_RawFalse)
        self.Button_Shutdown.clicked.connect(self.shut_down)

        self.Button_MqttTrue.clicked.connect(self.Button_Mqtt_True)
        self.Button_MqttFalse.clicked.connect(self.Button_Mqtt_False)

#######################
    def changeWidget(self):
        # op = QGraphicsOpacityEffect()
        # op.setOpacity(0)
        self.Hist_fore.setVisible(False)
        self.Hist_left.setVisible(False)
        self.Hist_right.setVisible(False)
        self.label_fore.setText("Face Signal")
        self.label_left.setVisible(False)
        # self.Signal_left.setVisible(False)  #
        self.Signal_left.hideAxis('left')
        self.Signal_left.hideAxis('bottom')
        self.label_right.setVisible(False)
        # self.Signal_right.setVisible(False) #
        self.Signal_right.hideAxis('left')
        self.Signal_right.hideAxis('bottom')
        # self.Spectrum_left.setVisible(False)#
        self.Spectrum_left.hideAxis('left')
        self.Spectrum_left.hideAxis('bottom')        
        # self.Spectrum_right.setVisible(False)#
        self.Spectrum_right.hideAxis('left')
        self.Spectrum_right.hideAxis('bottom')


    def returnWidget(self):
        self.Hist_fore.setVisible(True)
        self.Hist_left.setVisible(True)
        self.Hist_right.setVisible(True)
        self.label_fore.setText("Forehead Signal")
        self.label_left.setVisible(True)
        # self.Signal_left.setVisible(True)   #
        self.Signal_left.showAxis('left')
        self.Signal_left.showAxis('bottom')
        self.label_right.setVisible(True)
        # self.Signal_right.setVisible(True)  #
        self.Signal_right.showAxis('left')
        self.Signal_right.showAxis('bottom')
        # self.Spectrum_left.setVisible(True) #
        self.Spectrum_left.showAxis('left')
        self.Spectrum_left.showAxis('bottom')
        # self.Spectrum_right.setVisible(True)#
        self.Spectrum_right.showAxis('left')
        self.Spectrum_right.showAxis('bottom')
        
########################
    def Button_ChangeMode(self, str):
        self.Mode = self.ModeDict[str]
        self.Mode_str=str
        if self.pre_mode in self.unsupervised_model:
            if self.Mode_str=='PhysFormer':
                self.update_physFormer=0
                self.showed_physFormer=0
                self.PhysFormer.start()
                self.changeWidget()
            elif self.Mode_str=='TSCAN':
                self.update_TSCAN=0
                self.showed_TSCAN=0
                self.TSCAN.start()
                self.changeWidget()
            else:
                pass
        elif self.pre_mode=='PhysFormer':            
            if self.Mode_str in self.unsupervised_model:
                self.PhysFormer.stop()
                self.returnWidget()
            elif self.Mode_str=='TSCAN':
                self.PhysFormer.stop()
                self.update_TSCAN=0
                self.showed_TSCAN=0
                self.TSCAN.start()
            else:
                pass
        else:
            if self.Mode_str in self.unsupervised_model:
                self.TSCAN.stop()
                self.returnWidget()
            elif self.Mode_str=='PhysFormer':
                self.TSCAN.stop()
                self.update_physFormer=0
                self.showed_physFormer=0
                self.PhysFormer.start()
            else:
                pass
        self.pre_mode=str


    def Button_Data_RawTrue(self):
        self.Data_ShowRaw = True

    def Button_Data_RawFalse(self):
        self.Data_ShowRaw = False

    # Define callback function to handle incoming messages
    def on_message(self, client, userdata, message):
        self.url = 'http://'+message.payload.decode()+':81/stream'
        print(f"Received message: {self.url}")
        self.message_received = True
    
    def Button_Mqtt_True(self):
        # Define MQTT broker details
        broker_address = "p224517a.emqx.cloud"
        broker_port = 1883
        topic = "aiot/esp32/test"
        mqtt_username = "aiot20"
        mqtt_password = "12345610"
      
        # Create MQTT client and set callback function
        client = mqtt.Client(userdata={"message_received": False})
        client.username_pw_set(mqtt_username, mqtt_password)

        client.on_message = self.on_message

        # Connect to MQTT broker
        client.connect(broker_address, broker_port)

        # Subscribe to the topic
        client.subscribe(topic)

        # Start the MQTT loop to receive messages
        client.loop_start()

        # Wait until a message is received or timeout occurs
        timeout = 10  # seconds
        start_time = time.time()
        while not self.message_received:
            if time.time() - start_time > timeout:
                break

        if self.url == None:
            print("Connection timeout.")
            return

        # self.url = 'http://192.168.28.85:81/stream' 
        self.processor.series_class.cam = cv.VideoCapture(self.url)

    def Button_Mqtt_False(self):
        # self.Mqtt_Mode = False
        self.message_received = False
        self.processor.series_class.cam = cv.VideoCapture(0)
        
    def shut_down(self):
        self.processor.__del__()
        sys.exit()

    def DisplayImage(self):
        # frame = self.processor.series_class.frame_display
        Mask = self.processor.series_class.face_mask
        # Mask = cv.ellipse(Mask, [320, 240], [80, 120], 0, 0, 360,
        #                   [0, 255, 0], 1, cv.LINE_AA)
        # Mask = cv.circle(Mask, [320, 240], 2, [255, 0, 0], 2, cv.LINE_AA)

        if Mask is not None:
            # Mask = cv.resize(Mask, (331, 321))
            img = cv.cvtColor(Mask, cv.COLOR_BGR2RGB)
            qimg = QImage(
                img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)

            self.face.setPixmap(QPixmap.fromImage(qimg))

    def DisplayHist(self):
        Hist_fore = np.array(self.processor.series_class.hist_fore)
        Hist_left = np.array(self.processor.series_class.hist_left)
        Hist_right = np.array(self.processor.series_class.hist_right)
        if Hist_fore.size != 1:
            self.Hist_fore_r.setData(Hist_fore[0, :], pen=(174, 0, 0))
            self.Hist_fore_g.setData(Hist_fore[1, :], pen=(0, 174, 0))
            self.Hist_fore_b.setData(Hist_fore[2, :], pen=(0, 0, 174))
        else:
            self.Hist_fore_r.clear()
            self.Hist_fore_g.clear()
            self.Hist_fore_b.clear()
        if Hist_left.size != 1:
            self.Hist_left_r.setData(Hist_left[0, :], pen=(174, 0, 0))
            self.Hist_left_g.setData(Hist_left[1, :], pen=(0, 174, 0))
            self.Hist_left_b.setData(Hist_left[2, :], pen=(0, 0, 174))
        else:
            self.Hist_left_r.clear()
            self.Hist_left_g.clear()
            self.Hist_left_b.clear()
        if Hist_fore.size != 1:
            self.Hist_right_r.setData(Hist_right[0, :], pen=(174, 0, 0))
            self.Hist_right_g.setData(Hist_right[1, :], pen=(0, 174, 0))
            self.Hist_right_b.setData(Hist_right[2, :], pen=(0, 0, 174))
        else:
            self.Hist_right_r.clear()
            self.Hist_right_g.clear()
            self.Hist_right_b.clear()

    # Creates the specified Butterworth filter and applies it.
    def butterworth_filter(self, data, low, high, sample_rate, order=11):
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)
    

    def DisplaySignal(self):

        if self.Mode_str in self.unsupervised_model:

            Sig_fore = np.array(self.processor.series_class.Sig_fore)
            Sig_left = np.array(self.processor.series_class.Sig_left)
            Sig_right = np.array(self.processor.series_class.Sig_right)
            if self.processor.series_class.Flag_Queue:
                if Sig_fore.size != 1:
                    # self.bvp_fore_raw = self.processor.PBV(Sig_fore)
                    self.bvp_fore_raw = self.Mode(Sig_fore)
                    self.quality_fore = 1 / \
                        (max(self.bvp_fore_raw)-min(self.bvp_fore_raw))
                    self.bvp_fore = self.butterworth_filter(
                        self.processor.Signal_Preprocessing_single(self.bvp_fore_raw), MIN_HZ, MAX_HZ, self.processor.series_class.fps, order=5)
                    self.spc_fore = np.abs(np.fft.fft(self.bvp_fore))
                    self.bpm_fore = self.processor.cal_bpm(
                        self.bpm_fore, self.spc_fore, self.processor.series_class.fps)
                    if self.Data_ShowRaw:
                        # self.Sig_f.setData(self.bvp_fore_raw, pen=(0, 255, 255))
                        self.Sig_f.setData(self.bvp_fore_raw, pen=(174, 0, 0))
                    else:
                        # self.Sig_f.setData(self.bvp_fore, pen=(0, 255, 255))
                        self.Sig_f.setData(self.bvp_fore, pen=(174, 0, 0))
                    # self.Spec_f.setData(np.linspace(0,self.processor.series_class.fps/2*60,int((len(self.spc_fore)+1)/2)),
                    #     self.spc_fore[:int((len(self.spc_fore)+1)/2)], pen=(0, 255, 255))
                    self.Spec_f.setData(np.linspace(0,self.processor.series_class.fps/2*60,int((len(self.spc_fore)+1)/2)),
                        self.spc_fore[:int((len(self.spc_fore)+1)/2)], pen=(174, 0, 0))
                else:
                    self.Sig_f.setData([0], [0])
                    self.Spec_f.setData([0], [0])
                if Sig_left.size != 1:
                    # self.bvp_left_raw = self.processor.GREEN(Sig_left)
                    self.bvp_left_raw = self.Mode(Sig_left)
                    self.quality_left = 1 / \
                        (max(self.bvp_left_raw)-min(self.bvp_left_raw))
                    self.bvp_left = self.butterworth_filter(
                        self.processor.Signal_Preprocessing_single(self.bvp_left_raw), MIN_HZ, MAX_HZ, self.processor.series_class.fps, order=5)
                    self.spc_left = np.abs(np.fft.fft(self.bvp_left))
                    self.bpm_left = self.processor.cal_bpm(
                        self.bpm_left, self.spc_left, self.processor.series_class.fps)
                    if self.Data_ShowRaw:
                        # self.Sig_l.setData(self.bvp_left_raw, pen=(255, 0, 255))
                        self.Sig_l.setData(self.bvp_left_raw, pen=(0, 174, 0))
                    else:
                        # self.Sig_l.setData(self.bvp_left, pen=(255, 0, 255))
                        self.Sig_l.setData(self.bvp_left, pen=(0, 174, 0))
                    # self.Spec_l.setData(np.linspace(0,self.processor.series_class.fps/2*60,int((len(self.spc_left)+1)/2)),
                    #     self.spc_left[:int((len(self.spc_left)+1)/2)], pen=(255, 0, 255))
                    self.Spec_l.setData(np.linspace(0,self.processor.series_class.fps/2*60,int((len(self.spc_left)+1)/2)),
                        self.spc_left[:int((len(self.spc_left)+1)/2)], pen=(0, 174, 0))
                else:
                    self.Sig_l.setData([0], [0])
                    self.Spec_l.clear([0], [0])
                if Sig_right.size != 1:
                    # self.bvp_right_raw = self.processor.CHROM(Sig_right)
                    self.bvp_right_raw = self.Mode(Sig_right)
                    self.quality_right = 1 / \
                        (max(self.bvp_right_raw)-min(self.bvp_right_raw))
                    self.bvp_right = self.butterworth_filter(
                        self.processor.Signal_Preprocessing_single(self.bvp_right_raw), MIN_HZ, MAX_HZ, self.processor.series_class.fps, order=5)
                    self.spc_right = np.abs(np.fft.fft(self.bvp_right))
                    self.bpm_right = self.processor.cal_bpm(
                        self.bpm_right, self.spc_right, self.processor.series_class.fps)
                    if self.Data_ShowRaw:
                        # self.Sig_r.setData(self.bvp_right_raw, pen=(255, 255, 0))
                        self.Sig_r.setData(self.bvp_right_raw, pen=(0, 0, 174))
                    else:
                        # self.Sig_r.setData(self.bvp_right, pen=(255, 255, 0))
                        self.Sig_r.setData(self.bvp_right, pen=(0, 0, 174))
                    # self.Spec_r.setData(np.linspace(0,self.processor.series_class.fps/2*60,int((len(self.spc_right)+1)/2)),
                    #     self.spc_right[:int((len(self.spc_right)+1)/2)], pen=(255, 255, 0))
                    self.Spec_r.setData(np.linspace(0,self.processor.series_class.fps/2*60,int((len(self.spc_right)+1)/2)),
                        self.spc_right[:int((len(self.spc_right)+1)/2)], pen=(0, 0, 174))
                else:
                    self.Sig_r.setData([0], [0])
                    self.Spec_r.setData([0], [0])
                self.quality_all = self.quality_fore+self.quality_left+self.quality_right
                self.confidence_fore = self.quality_fore/self.quality_all
                self.confidence_left = self.quality_left/self.quality_all
                self.confidence_right = self.quality_right/self.quality_all
                self.bpm_avg = self.bpm_fore*self.confidence_fore+self.bpm_left * \
                    self.confidence_left+self.bpm_right*self.confidence_right
                Label_Text = "fps: \t\t"+str(round(self.processor.series_class.fps,2)) \
                                +"\nFore BPM: \t"+str(round(self.bpm_fore,2))+"\nFore Confidence: "\
                                +str(round(self.confidence_fore*100,2))+"%\nLeft BPM: \t"\
                                +str(round(self.bpm_left,2))+"\nLeft Confidence: "\
                                +str(round(self.confidence_left*100,2))+"%\nRight BPM:\t"\
                                +str(round(self.bpm_right,2))+"\nRight Confidence:"\
                                +str(round(self.confidence_right*100,2))+"%\n\nBPM Overall: \t"\
                                +str(round(self.bpm_avg,2))
                self.label.setText(Label_Text)
            else:
                self.Sig_f.setData([0], [0])
                self.Spec_f.setData([0], [0])
                self.Sig_l.setData([0], [0])
                self.Spec_l.setData([0], [0])
                self.Sig_r.setData([0], [0])
                self.Spec_r.setData([0], [0])
                self.label.setText(
                    "fps:\t\t"+str(round(self.processor.series_class.fps,2))\
                        +"\nData Collecting...")
        elif self.Mode_str=='PhysFormer':
            self.Sig_l.setData([0], [0])
            self.Spec_l.setData([0], [0])
            self.Sig_r.setData([0], [0])
            self.Spec_r.setData([0], [0])

            if self.processor.series_class.Flag_P_Queue:
                if self.update_physFormer!=self.showed_physFormer:
                    self.bvp_fore_raw = self.PhysFormer_frames
                    self.spc_fore = np.abs(np.fft.fft(self.bvp_fore_raw))
                    self.bpm_fore = self.processor.cal_bpm(
                        self.bpm_fore, self.spc_fore, self.processor.series_class.fs)

                    self.Sig_f.setData(self.bvp_fore_raw, pen=(174, 174, 0))
                    self.Spec_f.setData(np.linspace(0,self.processor.series_class.fs/2*60,int((len(self.spc_fore)+1)/2)),
                        self.spc_fore[:int((len(self.spc_fore)+1)/2)], pen=(174, 174, 0))
                    self.showed_physFormer+=1
                Label_Text = "Fs: \t\t"+str(self.processor.series_class.fs)+"\nBPM: \t"+str(self.bpm_fore)
                self.label.setText(Label_Text)
            else:
                self.Sig_f.setData([0], [0])
                self.Spec_f.setData([0], [0]) 
                self.label.setText("Fs:\t\t"+str(self.processor.series_class.fs)+"\nData Collecting...")

        elif self.Mode_str=='TSCAN':
            self.Sig_l.setData([0], [0])
            self.Spec_l.setData([0], [0])
            self.Sig_r.setData([0], [0])
            self.Spec_r.setData([0], [0])

            if self.processor.series_class.Flag_T_Queue:
                if self.update_TSCAN!=self.showed_TSCAN:
                    self.bvp_fore_raw = self.TSCAN_frames
                    self.spc_fore = np.abs(np.fft.fft(self.bvp_fore_raw))
                    self.bpm_fore = self.processor.cal_bpm(
                        self.bpm_fore, self.spc_fore, self.processor.series_class.fs)

                    self.Sig_f.setData(self.bvp_fore_raw, pen=(174, 174, 0))
                    self.Spec_f.setData(np.linspace(0,self.processor.series_class.fs/2*60,int((len(self.spc_fore)+1)/2)),
                        self.spc_fore[:int((len(self.spc_fore)+1)/2)], pen=(174, 174, 0))
                    self.showed_TSCAN+=1
                Label_Text = "Fs: \t\t"+str(self.processor.series_class.fs)+"\nBPM: \t"+str(self.bpm_fore)
                self.label.setText(Label_Text)
            else:
                self.Sig_f.setData([0], [0])
                self.Spec_f.setData([0], [0]) 
                self.label.setText("Fs:\t\t"+str(self.processor.series_class.fs)+"\nData Collecting...")
        else:
            print('Wrong Mode!')


    def cal_physformer(self):
        self.update_physFormer=False
        P_frames= np.array(list(self.processor.series_class.video.queue)[-640:])
        #print('P_frames',P_frames.shape)  640, 720, 1280, 3
        P_frames = (np.round(P_frames * 255)).astype(np.uint8)
        self.PhysFormer_frames= self.processor.PhysFormer_predict(P_frames)
        self.update_physFormer+=1

    def cal_TSCAN(self):
        self.update_TSCAN=False
        T_frames= np.array(list(self.processor.series_class.video.queue)[-720:])
        T_frames = (np.round(T_frames * 255)).astype(np.uint8)
        self.TSCAN_frames = self.processor.TSCAN_predict(T_frames)
        self.update_TSCAN+=1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = mainwin()
    ui.show()
    if app.exec_()==0:
        ui.processor.__del__()
        sys.exit()
    
    