import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QRegExp, Qt, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QRegExpValidator, QIntValidator, QDoubleValidator, QCursor, QPainter, QPen, \
    QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QHeaderView, QTableWidgetItem, QLabel, QColorDialog
from Tools import *
from MyUI import Ui_Form
import torch.backends.cudnn as cudnn
# NST相关
from models import FSRCNN
from utils_SR import convert_ycbcr_to_rgb, preprocess
import torch
import utils_NST
from torch.autograd import Variable
from transformer_net import TransformerNet
import datetime

def getGrayDiffPromote2(img, avg, tmpPoint):
    return abs(avg - img[tmpPoint.y, tmpPoint.x, 2])

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.y, currentPoint.x]) - int(img[tmpPoint.y, tmpPoint.x]))

def comp_max(tp1, tp2):
    if tp1[0] < tp2[0]:
        tp1[0] = tp2[0]
    if tp1[1] < tp2[1]:
        tp1[1] = tp2[1]
    if tp1[2] < tp2[2]:
        tp1[2] = tp2[2]
    return tp1

class Point:
    def __init__(self, x, y, type=0):
        self.x = x
        self.y = y
        self.type = type

class MyLabel_inteSeg(QLabel):

    def __init__(self, parent=None):
        super(MyLabel_inteSeg, self).__init__((parent))
        self.flag = False
        self.isShow = False
        self.point_type = 0  # 0-左键前景点,1-右键背景点
        self.clk_pos = None
        self.x = None
        self.y = None
        self.pointList = []
        self.signDrawRect = False
        self.flagDrawRect = False
        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0





    def clearPoints(self):
        self.pointList = []

    def getPoints(self):
        return self.pointList

    def mousePressEvent(self, event):
        QLabel.mousePressEvent(self, event)
        if event.buttons() == QtCore.Qt.LeftButton:
            self.point_type = 0
        elif event.buttons() == QtCore.Qt.RightButton:
            self.point_type = 1
        if self.signDrawRect:
            self.flagDrawRect = True
            self.x0 = event.x()
            self.y0 = event.y()

        if self.isShow == True:
            self.update()
            self.clk_pos = event.globalPos()
            self.x = event.x()
            self.y = event.y()
            print('add')
            self.pointList.append(Point(self.x, self.y, self.point_type))

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter()
        if len(self.pointList) != 0:
            for point in self.pointList:
                painter.begin(self)
                if point.type == 1:
                    painter.setPen(QPen(QColor(0, 255, 0), 7))
                else:
                    painter.setPen(QPen(QColor(255, 0, 0), 7))
                painter.drawPoint(point.x, point.y)
                painter.end()

        if self.isShow == True:
            # while self.isShow == True:
            if self.point_type == 0:
                painter.begin(self)
                painter.setPen(QPen(QColor(255, 0, 0), 7))
                painter.drawPoint(self.x, self.y)
                painter.end()

            elif self.point_type == 1:
                painter.begin(self)
                painter.setPen(QPen(QColor(0, 255, 0), 7))
                painter.drawPoint(self.x, self.y)
                painter.end()
        if self.signDrawRect:
            rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect)

    def mouseReleaseEvent(self, event):
        QLabel.mouseReleaseEvent(self, event)
        self.flagDrawRect = False
        # self.isShow = False

    def getPointGlobalPos(self):
        return self.clk_pos

    def mouseMoveEvent(self, event):
        if self.flagDrawRect:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def getRect(self):
        # return {'x0':self.x0,'x1':self.x1,'y0':self.y0,'y1':self.y1}
        return (self.x0, self.x1, self.y0, self.y1)

def oilpaint(img, basicSize=4, grayLevelSize=8, gap=2, Mean_sta=True):
    imgHeight, imgWidth, _ = img.shape
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dstImg = np.zeros(img.shape, np.uint8)
    for i in range(basicSize, imgHeight - basicSize, gap):
        for j in range(basicSize, imgWidth - basicSize, gap):
            # 灰度等级统计
            grayLevel = np.zeros(grayLevelSize, np.uint8)  # 存放各个灰度等级的个数
            graySum = [0, 0, 0]  # 用于最后高频灰度等级均值计算
            # 对小区域进行遍历统计
            for m in range(-basicSize, basicSize):
                for n in range(-basicSize, basicSize):
                    pixlv = int(grayImg[i + m, j + n] / (256 / grayLevelSize))  # 判断像素等级
                    grayLevel[pixlv] += 1  # 计算对应灰度等级个数
            # 找出最高频灰度等级及其索引
            mostLevel = np.max(grayLevel)
            mostLevelIndex = np.argmax(grayLevel)
            level_pixNum = 0
            # 计算最高频等级内的所有灰度值的均值
            for m in range(-basicSize, basicSize):
                for n in range(-basicSize, basicSize):
                    if int(grayImg[i + m, j + n] / (256 / grayLevelSize)) == mostLevelIndex:
                        level_pixNum += 1
                        if Mean_sta:
                            graySum += img[i + m, j + n]
                        else:
                            graySum = comp_max(graySum, img[i + m, j + n])
            if Mean_sta:
                graySum = graySum / level_pixNum
            # 写入目标像素
            for m in range(gap):
                for n in range(gap):
                    dstImg[i + m, j + n] = graySum

    return dstImg

class MyApp(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)
        self.setupUIAdded()  # 对UI的另外的改动
        self.disableAll()
        self.disableAll2()
        self.m_flag = False
        self.img1 = None
        self.img2 = None
        self.imgTmp = None
        self.tmpHue = None
        # self.imgTmp2 = None
        self.sign = False
        self.weight = float(self.edWeight.text())
        self.fPro4Zero = 0.25
        self.fPro4One = 0.75
        self.signSeedChoose = False
        self.thresh = 12
        self.signSegMode = 'Add'
        self.signOffset = False
        # 计时器
        self.timer_camera = QTimer()
        self.slotsInit()
        self.timerInit()
        #修复
        self.signInpaint = 1
        self.colorOutline = np.array([0,0,255])

    def setupUIAdded(self):
        rx = QRegExp("^-?(255|([1,2]?[0-4]?\d|[1,2]?5[0-4]?)(\.\d)?)$")
        pReg = QRegExpValidator(rx)
        self.edMax.setValidator(QIntValidator(0, 255))
        self.edMin.setValidator(QIntValidator(0, 255))
        self.edLogC.setValidator(QDoubleValidator())
        self.edS1.setValidator(pReg)
        self.edS2.setValidator(pReg)
        self.edT1.setValidator(pReg)
        self.edT2.setValidator(pReg)
        self.ed_theta.setValidator(QIntValidator(-180, 180))
        self.edZoomInRateX.setValidator(QDoubleValidator())
        self.edZoomInRateY.setValidator(QDoubleValidator())
        self.edSNR.setValidator(QDoubleValidator())
        self.edMean.setValidator(QDoubleValidator())
        self.edVar.setValidator(QDoubleValidator())

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        # self.setFixedSize(1196, 816)
        self.lbAbove.hide()


        self.cascade_path = "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(self.cascade_path)

        self.twHuffman.setColumnCount(4)
        self.twHuffman.setRowCount(256)
        self.twHuffman.setHorizontalHeaderLabels(['灰度值', '概率值', '编码', '长度'])
        self.twHuffman.horizontalHeader().setVisible(True)
        self.twHuffman.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.twHuffman.resizeColumnsToContents()
        self.twHuffman.resizeRowsToContents()

        self.cbEdgeDetect.addItems(['Robert', 'Sobel', 'Prewitt', 'Lapplacian'])
        self.cbScale.addItems(['x2','x3','x4'])
        self.cbNSTModel.addItems(['candy','mosaic','starry-night','udnie'])

    def slotsInit(self):
        self.pbCamera.clicked.connect(self.openCamera)
        self.pbOpen.clicked.connect(self.readImg)
        self.pbSave.clicked.connect(self.saveImg)
        self.pbGrayTransformLinearTransform.clicked.connect(self.LinearTransform)
        self.pbGrayTransformSegLinearTransform.clicked.connect(self.SegLinearTransform)
        self.pbGrayTransformLogLinearTransform.clicked.connect(self.LogLinearTransform)
        self.pbGrayTransformHistorgramEqual.clicked.connect(self.HistorgramEqual)
        self.pbGrayTransformMiddleSmooth.clicked.connect(self.MiddleSmooth)
        self.pbGrayTransformAverageSmooth.clicked.connect(self.AverageSmooth)

        self.pbTransformZoomIn.clicked.connect(self.ZoomIn)
        self.pbSR.clicked.connect(self.SR)
        self.pbTransformRotateBetter.clicked.connect(self.RotateBetter)
        self.pbTransformVerInverse.clicked.connect(self.VerInverse)
        self.pbTransformHorInverse.clicked.connect(self.HorInverse)
        self.pbSpNoise.clicked.connect(self.SpNoise)
        self.pbGasNoise.clicked.connect(self.GasNoise)
        self.pbRandomNoise.clicked.connect(self.RandomNoise)
        self.pbOffset.clicked.connect(self.Offset)
        self.pbOffsetSave.clicked.connect(self.OffsetSave)
        self.pbClip.clicked.connect(self.Clip)
        self.pbClipSave.clicked.connect(self.ClipSave)

        self.pbAddFrame.clicked.connect(self.AddFrame)
        self.pbOpenPic1.clicked.connect(self.OpenPic1)
        self.pbOpenPic2.clicked.connect(self.OpenPic2)
        self.pbAddedShow.clicked.connect(self.AddedShow)
        self.pbAdd.clicked.connect(self.Add)
        self.pbMinus.clicked.connect(self.Minus)
        self.pbCatHor.clicked.connect(self.CatHor)
        self.pbCatVer.clicked.connect(self.CatVer)
        self.sliderWeight.valueChanged.connect(self.changeWeightFromSd)
        self.edWeight.editingFinished.connect(self.changeWeightFromEd)

        self.pbBeautifyBuffing.clicked.connect(self.BeautifyBuffing)
        self.sliderBeautifyBuffing.valueChanged.connect(self.BeautifyBuffingFromSilder)
        self.pbBeautifyRecover.clicked.connect(lambda:self.display(None,None))
        self.pbBeautifySaturability.clicked.connect(self.BeautifySaturability)
        self.sliderBeautifySaturability.valueChanged.connect(self.BeautifySaturabilityFromSlider)
        self.pbBeautifyGray.clicked.connect(self.BeautifyGray)
        self.sliderBeautifyGray.valueChanged.connect(self.BeautifyGrayFromSlider)
        self.pbAutoBeautify.clicked.connect(self.AutoBeautify)
        self.pbRBeautifyGray.clicked.connect(self.BeautifyGrayR)
        self.pbRBeautifyHue.clicked.connect(self.BeautifyHueR)
        self.pbRBeautifySaturability.clicked.connect(self.BeautifySaturabilityR)

        self.pbClose.clicked.connect(self.close)
        self.pbMini.clicked.connect(self.showMinimized)
        self.pbMax.clicked.connect(self.showChange)

        # Codeing
        self.pbEncodeHuffman.clicked.connect(self.EncodeHuffman)
        self.pbEncodeShannon.clicked.connect(self.EncodeShanno)
        # EdgeDetection
        self.pbEncodeEdgeDetect.clicked.connect(self.EdgeDetect)
        self.sliderWeightEdge.valueChanged.connect(self.EdgechangeWeightFromSd)
        self.edWeightEdge.editingFinished.connect(self.EdgechangeWeightFromEd)
        self.pbEmbossment.clicked.connect(self.Embossment)
        self.pbEdgeSaveChange.clicked.connect(self.SegSaveChange)
        self.pbEdgeRecover.clicked.connect(self.SegRecover)
        self.pbSharpen.clicked.connect(self.Sharpen)
        # self.sliderOutline.valueChanged.connect(self.OutlineFromSd)
        self.pbOutline.clicked.connect(self.Outline)
        self.pbOutlineSave.clicked.connect(self.OutlineSave)
        self.pbOutlineColor.clicked.connect(self.OutlineColor)
        self.pbOutlineRecover.clicked.connect(self.display)
        # Seg
        self.pbSegRegionGrow.clicked.connect(self.SegRegionGrowing)
        self.pbSegRegionGrowChooseSeed.clicked.connect(self.ChooseSeed)
        self.pbSegRegionGrowRealTime.clicked.connect(self.ChooseSeedRealTime)
        self.sliderThresh.valueChanged.connect(self.changeThreshFromSd)
        self.edThresh.editingFinished.connect(self.changeThreshFromEd)
        self.pbSegSaveChange.clicked.connect(self.SegSaveChange)
        self.pbSegRecover.clicked.connect(self.SegRecover)
        self.rbSegAdd.toggled.connect(lambda: self.changeSegMode(self.rbSegAdd))
        self.rbSegMin.toggled.connect(lambda: self.changeSegMode(self.rbSegMin))
        self.pbSegThresh.clicked.connect(self.Segthresh)
        self.pbSegRegionGrowAuto.clicked.connect(self.SegAuto)
        # 色相调整
        self.sliderHue.valueChanged.connect(self.changeHue)
        # 滤镜
        self.pbFilterCyberpunk.clicked.connect(self.filterCyberpunk)
        self.pbFilterRetro.clicked.connect(self.filterRetro)
        self.pbFilterOilPainting.clicked.connect(self.filterOilPainting)
        # 风格迁移
        self.pbNST.clicked.connect(self.NST)
        # Inpaint
        self.pbInpaint.clicked.connect(self.inPaint)
        self.sliderInpaint.valueChanged.connect(self.inPintThreshChangeFromSd)

    def updateInfo(self,img):
        # Size
        self.edInfoSize.setText("{}×{}".format(img.shape[1],img.shape[0]))
        # entropy
        entropy = self.getEntropy(img)
        self.edInfoEntropy.setText('%.5f' % entropy)
        # time
        now_time = datetime.datetime.now()
        time1_str = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S')
        self.edInfoEditTime.setText(time1_str)



    def showChange(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeHue(self):
        self.Hue = self.sliderHue.value()
        tmpHue = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        tmpHue[:, :, 0] += self.Hue
        self.tmpHue = cv2.cvtColor(tmpHue, cv2.COLOR_HSV2BGR)
        self.display(self.tmpHue)

    def saveHue(self):
        self.img = self.tmpHue
        self.display()

    def timerInit(self):
        self.timer_camera.timeout.connect(self.cameraCollect)

    def changeSegMode(self, rb):
        if rb.text() == '添选':
            self.signSegMode = 'Add'
        elif rb.text() == '删选':
            self.signSegMode = 'Min'

    def changeThreshFromSd(self):
        self.thresh = self.sliderThresh.value()
        self.edThresh.setText(str(self.thresh))

    def changeThreshFromEd(self):
        self.thresh = int(self.edThresh.text())
        self.sliderThresh.setValue(int(self.thresh))

    def EdgechangeWeightFromEd(self):
        self.weight = float(self.edWeightEdge.text())
        self.sliderWeightEdge.setValue(int(self.weight * 100))

    def EdgechangeWeightFromSd(self):
        self.weight = self.sliderWeightEdge.value() / 100
        self.edWeightEdge.setText(str(self.weight))

    def changeWeightFromSd(self):
        self.weight = self.sliderWeight.value() / 100
        self.edWeight.setText(str(self.weight))

    def changeWeightFromEd(self):
        self.weight = float(self.edWeight.text())
        self.sliderWeight.setValue(int(self.weight * 100))

    def disableAll(self):
        self.pbTransformRotateBetter.setEnabled(False)
        self.pbTransformHorInverse.setEnabled(False)
        self.pbTransformVerInverse.setEnabled(False)
        self.pbClip.setEnabled(False)
        self.pbClipSave.setEnabled(False)
        self.pbTransformZoomIn.setEnabled(False)
        self.pbSR.setEnabled(False)

        self.pbRBeautifyGray.setEnabled(False)
        self.sliderBeautifyGray.setEnabled(False)
        self.pbRBeautifyHue.setEnabled(False)
        self.sliderHue.setEnabled(False)
        self.pbRBeautifySaturability.setEnabled(False)
        self.sliderBeautifySaturability.setEnabled(False)
        self.pbBeautifyGray.setEnabled(False)
        self.pbRBeautifyHue.setEnabled(False)
        self.pbRBeautifySaturability.setEnabled(False)
        self.pbGrayTransformSegLinearTransform.setEnabled(False)
        self.pbGrayTransformLogLinearTransform.setEnabled(False)
        self.pbGrayTransformLinearTransform.setEnabled(False)
        self.pbGrayTransformHistorgramEqual.setEnabled(False)

        self.sliderBeautifyBuffing.setEnabled(False)
        self.pbBeautifyBuffing.setEnabled(False)
        self.pbBeautifyRecover.setEnabled(False)
        self.pbGrayTransformAverageSmooth.setEnabled(False)
        self.pbGrayTransformMiddleSmooth.setEnabled(False)
        self.pbFilterOilPainting.setEnabled(False)
        self.pbFilterRetro.setEnabled(False)
        self.pbFilterCyberpunk.setEnabled(False)
        self.pbEmbossment.setEnabled(False)
        self.pbRandomNoise.setEnabled(False)
        self.pbSpNoise.setEnabled(False)
        self.pbGasNoise.setEnabled(False)
        self.pbAutoBeautify.setEnabled(False)
        self.pbNST.setEnabled(False)

        self.pbEncodeEdgeDetect.setEnabled(False)
        self.pbEdgeRecover.setEnabled(False)
        self.pbEdgeSaveChange.setEnabled(False)
        self.pbInpaint.setEnabled(False)
        self.pbSharpen.setEnabled(False)
        self.pbOutlineColor.setEnabled(False)
        self.pbOutlineRecover.setEnabled(False)
        self.pbOutlineSave.setEnabled(False)
        self.pbOutline.setEnabled(False)

        self.pbSegRegionGrow.setEnabled(False)
        self.pbSegRegionGrowChooseSeed.setEnabled(False)
        self.pbSegRegionGrowRealTime.setEnabled(False)
        self.pbSegRecover.setEnabled(False)
        self.pbSegSaveChange.setEnabled(False)
        self.pbSegRegionGrowAuto.setEnabled(False)
        self.pbSegThresh.setEnabled(False)

        self.pbEncodeShannon.setEnabled(False)
        self.pbEncodeHuffman.setEnabled(False)

    def enableAll(self):
        self.pbTransformRotateBetter.setEnabled(True)
        self.pbTransformHorInverse.setEnabled(True)
        self.pbTransformVerInverse.setEnabled(True)
        self.pbClip.setEnabled(True)
        self.pbClipSave.setEnabled(True)
        self.pbTransformZoomIn.setEnabled(True)
        self.pbSR.setEnabled(True)

        self.pbRBeautifyGray.setEnabled(True)
        self.sliderBeautifyGray.setEnabled(True)
        self.pbRBeautifyHue.setEnabled(True)
        self.sliderHue.setEnabled(True)
        self.pbRBeautifySaturability.setEnabled(True)
        self.sliderBeautifySaturability.setEnabled(True)
        self.pbBeautifyGray.setEnabled(True)
        self.pbRBeautifyHue.setEnabled(True)
        self.pbRBeautifySaturability.setEnabled(True)
        self.pbGrayTransformSegLinearTransform.setEnabled(True)
        self.pbGrayTransformLogLinearTransform.setEnabled(True)
        self.pbGrayTransformLinearTransform.setEnabled(True)
        self.pbGrayTransformHistorgramEqual.setEnabled(True)

        self.sliderBeautifyBuffing.setEnabled(True)
        self.pbBeautifyBuffing.setEnabled(True)
        self.pbBeautifyRecover.setEnabled(True)
        self.pbGrayTransformAverageSmooth.setEnabled(True)
        self.pbGrayTransformMiddleSmooth.setEnabled(True)
        self.pbFilterOilPainting.setEnabled(True)
        self.pbFilterRetro.setEnabled(True)
        self.pbFilterCyberpunk.setEnabled(True)
        self.pbEmbossment.setEnabled(True)
        self.pbRandomNoise.setEnabled(True)
        self.pbSpNoise.setEnabled(True)
        self.pbGasNoise.setEnabled(True)
        self.pbAutoBeautify.setEnabled(True)
        self.pbNST.setEnabled(True)

        self.pbEncodeEdgeDetect.setEnabled(True)
        self.pbEdgeRecover.setEnabled(True)
        self.pbEdgeSaveChange.setEnabled(True)
        self.pbInpaint.setEnabled(True)
        self.pbSharpen.setEnabled(True)
        self.pbOutlineColor.setEnabled(True)
        self.pbOutlineRecover.setEnabled(True)
        self.pbOutlineSave.setEnabled(True)
        self.pbOutline.setEnabled(True)

        self.pbSegRegionGrow.setEnabled(True)
        self.pbSegRegionGrowChooseSeed.setEnabled(True)
        self.pbSegRegionGrowRealTime.setEnabled(True)
        self.pbSegRecover.setEnabled(True)
        self.pbSegSaveChange.setEnabled(True)
        self.pbSegRegionGrowAuto.setEnabled(True)
        self.pbSegThresh.setEnabled(True)

        self.pbEncodeShannon.setEnabled(True)
        self.pbEncodeHuffman.setEnabled(True)

    def disableAll2(self):
        self.pbAddedShow.setEnabled(False)
        self.pbAdd.setEnabled(False)
        self.pbAddFrame.setEnabled(False)
        self.pbCatHor.setEnabled(False)
        self.pbCatVer.setEnabled(False)
        self.pbMinus.setEnabled(False)

    def enableAll2(self):
        self.pbAddedShow.setEnabled(True)
        self.pbAdd.setEnabled(True)
        self.pbAddFrame.setEnabled(True)
        self.pbCatHor.setEnabled(True)
        self.pbCatVer.setEnabled(True)
        self.pbMinus.setEnabled(True)


    def display(self, img=None, sign=None):
        if sign == None:
            if img is not None:
                show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                                   QImage.Format_RGB888)
                self.updateInfo(img)
            else:
                show = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                                   QImage.Format_RGB888)
                self.updateInfo(self.img)
            self.lbDisplay.setPixmap(
                QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())
        elif sign == 1:
            bk = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            front = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i][j].all() == 0:
                        bk[i][j][2] = 80
                    else:
                        bk[i][j] = front[i][j]
            show = cv2.cvtColor(bk, cv2.COLOR_HSV2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                               QImage.Format_RGB888)
            self.updateInfo(self.img)
            self.lbDisplay.setPixmap(QPixmap.fromImage(showImage))


    def openCamera(self):
        if self.timer_camera.isActive():
            self.img = self.capimg
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.lbDisplay.clear()  # 清空视频显示区域
            self.display()
            self.pbCamera.setText('摄像头采集')
        else:
            self.cap = cv2.VideoCapture()  # 视频流
            flag = self.cap.open(0)
            if not flag:
                QMessageBox.warning(None, 'Error', '打开摄像头失败')
                return
            self.timer_camera.start(20)
            self.pbCamera.setText('截取')

    def cameraCollect(self):
        flag, image = self.cap.read()
        if flag:
            self.capimg = image
            # show = cv2.resize(image, (640, 480))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            show = image.copy()
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.lbDisplay.setPixmap(QtGui.QPixmap.fromImage(showImage))
            # 框住人脸的矩形边框颜色
            color = (0, 255, 0)
            if flag:
                frame_gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)  # 图像灰化，降低计算复杂度
                faceRects = self.cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            else:
                return
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    cv2.rectangle(show, (x, y), (x + w, y + h), color, 2)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.lbDisplay.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def readImg(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", ".", "ImageFile(*.jpg *.jpeg *.bmp *.png)")
        if not ok:
            return
        self.img = cv2.imdecode(np.fromfile(get_filename_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.display()
        self.enableAll()

    def saveImg(self):
        get_directory_path, ok = QFileDialog.getSaveFileName(self, "选择保存目录", ".", "ImageFile(*.jpg *.jpeg *.bmp *.png)")
        if not '.' in get_directory_path:
            get_directory_path += '.jpg'
        str = get_directory_path.split('.')[-1]
        if ok:
            cv2.imencode('.' + str, self.img)[1].tofile(get_directory_path)

    def LinearTransform(self):
        graytf = GrayTransform(self.img)
        # get Params
        if self.edMax.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        max = int(self.edMax.text())
        min = int(self.edMin.text())
        self.img = graytf.LinearTransform(max=max, min=min)
        self.display()


    def SegLinearTransform(self):
        graytf = GrayTransform(self.img)
        # get Params
        if self.edS1.text() == '' or self.edS2.text() == '' or self.edT1.text() == '' or self.edT2.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        S1 = int(self.edS1.text())
        S2 = int(self.edS2.text())
        T1 = int(self.edT1.text())
        T2 = int(self.edT2.text())
        # 处理
        self.img = graytf.SegLinearTransform(s1=S1, s2=S2, t1=T1, t2=T2)
        self.display()

    def LogLinearTransform(self):
        graytf = GrayTransform(self.img)
        # get Params
        if self.edLogC.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        c = float(self.edLogC.text())
        # 处理
        self.img = graytf.LogTransform(c=c)
        self.display()

    def HistorgramEqual(self):
        # 处理
        graytf = GrayTransform(self.img)
        self.img = graytf.HistorgramEqual()
        self.display()

    def MiddleSmooth(self):
        graytf = GrayTransform(self.img)
        size = self.spinSize.value()
        self.img = graytf.MiddleSmooth(size=size)
        self.display()

    def AverageSmooth(self):
        graytf = GrayTransform(self.img)
        size = self.spinSize.value()
        self.img = graytf.AverageSmooth(size=size)
        self.display()

    def ZoomIn(self):
        tf = Transform(self.img)
        if self.edZoomInRateX.text() == '' or self.edZoomInRateY.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        rateX = float(self.edZoomInRateX.text())
        rateY = float(self.edZoomInRateY.text())
        self.img = tf.ZoomIn(rate_w=rateX, rate_h=rateY)
        self.display()

    def ZoomOut(self):
        tf = Transform(self.img)
        if self.edZoomOutRateX.text() == '' or self.edZoomOutRateY.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        rateX = float(self.edZoomOutRateX.text())
        rateY = float(self.edZoomOutRateY.text())
        self.img = tf.ZoomOut(rate_w=rateX, rate_h=rateY)
        self.display()

    def SR(self):
        scale = self.cbScale.currentText()
        if scale == 'x2':
            scale = 2
        elif scale == 'x3':
            scale = 3
        elif scale == 'x4':
            scale = 4
        weights_file = 'SR/fsrcnn_x{}.pth'.format(scale)
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = FSRCNN(scale_factor=scale).to(device)
        state_dict = model.state_dict()
        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        model.eval()

        image_width = self.img.shape[1]
        image_height = self.img.shape[0]
        self.img = cv2.resize(self.img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        new_width = image_width * scale
        new_height = image_height * scale
        hr = cv2.resize(self.img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        lr, _ = preprocess(self.img, device)
        _, ycbcr = preprocess(hr, device)
        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        self.img = output
        self.display()

    def Rotate(self):
        tf = Transform(self.img)
        if self.ed_theta.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        angle = float(self.ed_theta.text())
        self.img = tf.Rotate(angle)
        self.display()

    def RotateBetter(self):
        tf = Transform(self.img)
        if self.ed_theta.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        angle = float(self.ed_theta.text())
        self.img = tf.RotateBetter(angle)
        self.display()

    def VerInverse(self):
        tf = Transform(self.img)
        self.img = tf.VerInverse()
        self.display()

    def HorInverse(self):
        tf = Transform(self.img)
        self.img = tf.HorInverse()
        self.display()

    def SpNoise(self):
        tf = Transform(self.img)
        if self.edSNR.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        SNR = float(self.edSNR.text())
        self.img = tf.spNoise(SNR)
        self.display()

    def GasNoise(self):
        tf = Transform(self.img)
        if self.edMean.text() == '' or self.edVar.text() == '':
            QMessageBox.warning(None, 'Warning', '参数有误！')
            return
        mean = float(self.edMean.text())
        var = float(self.edVar.text())
        self.img = tf.gasussNoise(mean, var)
        self.display()

    def RandomNoise(self):
        noiseRate = float(self.edNoiseRate.text())
        num = int(noiseRate * self.img.shape[0] * self.img.shape[1])
        for t in range(num):
            x = np.random.randint(0, self.img.shape[0])
            y = np.random.randint(0, self.img.shape[1])
            self.img[x, y, :] = 255
        self.display()

    def OffsetSave(self):
        shape = self.img2.shape
        x = self.lbAbove.pos().x()
        y = self.lbAbove.pos().y()
        # newPos = self.lbDisplay.mapFromGlobal(QCursor.pos())
        # x = newPos.x()
        # y = newPos.y()
        if x < shape[1]:
            xStart = shape[1]
            xStop = x + shape[1]
        elif x >= shape[1]:
            xStart = x
            xStop = 2 * shape[1]
        if y < shape[0]:
            yStart = shape[0]
            yStop = y + shape[0]
        elif y >= shape[0]:
            yStart = y
            yStop = 2 * shape[0]
        img = np.zeros(shape, dtype=np.uint8)
        for i in range(yStart - shape[0], yStop - shape[0]):
            for j in range(xStart - shape[1], xStop - shape[1]):
                img[i][j] = self.img2[i + shape[0] - y][j + shape[1] - x]
        self.img = img
        self.display()
        self.lbAbove.hide()
        self.sign = False

    def Offset(self):
        # self.signOffset = True
        self.pbOffsetSave.setEnabled(True)
        self.img1 = np.zeros((3 * self.img.shape[0], 3 * self.img.shape[1], self.img.shape[2]), dtype=np.uint8)
        for i in range(self.img.shape[0], 2 * self.img.shape[0]):
            for j in range(self.img.shape[1], 2 * self.img.shape[1]):
                self.img1[i][j] = 255
        self.img2 = self.img
        self.img = self.img1
        self.display()
        self.lbAbove.setFixedSize(self.img2.shape[1], self.img2.shape[0])
        show = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                           QImage.Format_RGB888)
        self.lbAbove.setPixmap(QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())
        self.lbAbove.show()
        self.sign = True

    def Clip(self):
        self.lbDisplay.signDrawRect = True

    def ClipSave(self):
        self.lbDisplay.signDrawRect = False
        (x0, x1, y0, y1) = self.lbDisplay.getRect()
        if x0 >= x1:
            tmp=x0;x0=x1;x1=tmp
        if y0 >= y1:
            tmp=y0;y0=y1;y1=tmp
        if x1>=self.img.shape[1]:
            x1 = self.img.shape[1]
        if y1>=self.img.shape[0]:
            y1 = self.img.shape[0]
        newWid = abs(x0 - x1)
        newHei = abs(y0 - y1)
        img = np.zeros(shape=(newHei, newWid, self.img.shape[2]), dtype=np.uint8)
        for i in range(y0,y1):
            for j in range(x0,x1):
                img[i-y0][j-x0] = self.img[i][j]
        self.img = img
        self.display()

    def OpenPic1(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", ".", "ImageFile(*.jpg *.jpeg *.bmp *.png)")
        if not ok:
            return
        self.img1 = cv2.imdecode(np.fromfile(get_filename_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        show = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0],show.shape[1] * show.shape[-1], QImage.Format_RGB888)
        self.lbPic1.setPixmap(
            QPixmap.fromImage(showImage).scaled(self.lbPic1.width(),self.lbPic1.height()))
        if self.img1 is not None and self.img2 is not None:
            self.enableAll2()

    def OpenPic2(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", ".", "ImageFile(*.jpg *.jpeg *.bmp *.png)")
        if not ok:
            return
        self.img2 = cv2.imdecode(np.fromfile(get_filename_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        show = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0],show.shape[1] * show.shape[-1], QImage.Format_RGB888)
        self.lbPic2.setPixmap(
            QPixmap.fromImage(showImage).scaled(self.lbPic2.width(),self.lbPic2.height()))
        if self.img1 is not None and self.img2 is not None:
            self.enableAll2()

    def AddedShow(self):
        if self.sign == False:
            self.enableAll()
            if self.img1.shape[0] >= self.img2.shape[0] and self.img1.shape[1] >= self.img2.shape[1]:
                self.img = self.img1

            elif self.img1.shape[0] < self.img2.shape[0] and self.img1.shape[1] < self.img2.shape[1]:
                self.img = self.img2
                img = self.img2
                self.img2 = self.img1
                self.img1 = img

            else:
                tf = Transform(self.img1)
                if self.img1.shape[0] < self.img2.shape[0]:
                    rate_h = self.img2.shape[0] / self.img1.shape[0]
                else:
                    rate_h = 1
                if self.img1.shape[1] < self.img2.shape[1]:
                    rate_w = self.img2.shape[1] / self.img1.shape[1]
                else:
                    rate_w = 1
                self.img1 = tf.ZoomIn(rate_h=rate_h, rate_w=rate_w)
                self.img = self.img1
            self.display()
            self.lbAbove.setFixedSize(self.img2.shape[1], self.img2.shape[0])
            show = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                               QImage.Format_RGB888)
            self.lbAbove.setPixmap(
                QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())
            self.lbAbove.show()
            self.sign = True
            self.pbAddedShow.setText("显示原图")
        elif self.sign==True:
            self.display()
            self.sign = False
            self.pbAddedShow.setText("显示在窗口")
            self.lbAbove.hide()

    def Add(self):
        self.enableAll()
        Ar = Arithmetic(self.img1, self.img2)
        self.img = Ar.Add(self.lbAbove.pos().x(), self.lbAbove.pos().y(), self.weight)
        self.display()
        self.lbAbove.hide()
        self.sign = False

    def AddFrame(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", ".", "ImageFile(*.jpg *.jpeg *.bmp *.png)")
        if not ok:
            return
        frame = cv2.imdecode(np.fromfile(get_filename_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.img1 = frame
        self.img2 = self.img
        tf = Transform(self.img1)
        rate_h = self.img2.shape[0] / self.img1.shape[0]
        rate_w = self.img2.shape[1] / self.img1.shape[1]
        self.img1 = tf.ZoomIn(rate_h=rate_h, rate_w=rate_w)
        shape = self.img1.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.img1[i][j][0] == 0 and self.img1[i][j][1] == 0 and self.img1[i][j][2] == 0:
                    self.img[i][j] = self.img2[i][j]
                else:
                    self.img[i][j] = self.img1[i][j]
        self.display()
        # self.lbAbove.setFixedSize(self.img2.shape[1],self.img2.shape[0])
        # show = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
        # showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1]*show.shape[-1],QImage.Format_RGB888)
        # self.lbAbove.setPixmap(
        #     QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())
        # self.lbAbove.show()
        # self.sign = True

    def Minus(self):
        self.enableAll()
        Ar = Arithmetic(self.img1, self.img2)
        self.img = Ar.Minus(self.lbAbove.pos().x(), self.lbAbove.pos().y(), self.weight)
        self.display()
        self.enableAll()
        self.lbAbove.hide()
        self.sign = False

    def CatHor(self):
        self.enableAll()
        Ar = Arithmetic(self.img1, self.img2)
        self.img = Ar.CatHor()
        self.display()
        self.enableAll()
        self.lbAbove.hide()
        self.sign = False

    def CatVer(self):
        self.enableAll()
        Ar = Arithmetic(self.img1, self.img2)
        self.img = Ar.CatVer()
        self.display()
        self.enableAll()
        self.lbAbove.hide()
        self.sign = False

    def BeautifyGrayR(self):
        self.sliderBeautifyGray.setValue(0)
        self.display()

    def BeautifyHueR(self):
        self.sliderHue.setValue(0)
        self.display()

    def BeautifySaturabilityR(self):
        self.sliderBeautifySaturability.setValue(0)
        self.display()

    def BeautifyBuffing(self):
        p = self.sliderBeautifyBuffing.value()
        p = 1 - p / 100
        self.img = self.Buffing(p,self.img)
        self.display()

    def Buffing(self, Op,img):
        # int value1 = 3,value2 = 1;磨皮程度与细节程度
        value1 = 3
        value2 = 1
        dx = value1 * 12  # 双边滤波参数一
        fc = value1 * 8  # 双边滤波参数二
        p = Op  # 图片融合比例(透明度)
        temp4 = np.zeros_like(img)
        temp1 = cv2.bilateralFilter(img, dx, fc * 2, fc / 2)  # 双边滤波EPFFilter(Src)
        temp2 = cv2.subtract(temp1, img)  # EPFFilter(Src)-Src
        temp2 = cv2.add(temp2, (10, 10, 10, 128))  # EPFFilter(Src)-Src+128
        temp3 = cv2.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)  # 高斯模糊
        temp3 = 2 * temp3  # 2*GuassianBlur
        temp4 = cv2.add(img, temp3)  # Src+2*GuassianBlur
        dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
        dst = cv2.add(dst, (10, 10, 10, 255))
        return dst

    def BeautifyBuffingFromSilder(self):
        p = self.sliderBeautifyBuffing.value()
        p = 1 - p / 100
        img = self.Buffing(p,self.img)
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                           QImage.Format_RGB888)
        self.lbDisplay.setPixmap(
            QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())

    def BeautifySaturability(self):
        p = self.sliderBeautifySaturability.value()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                self.img[i][j][1] = p
        self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2RGB)
        self.display()

    def BeautifySaturabilityFromSlider(self):
        p = self.sliderBeautifySaturability.value()
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                img[i][j][1] = p
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                           QImage.Format_RGB888)
        self.lbDisplay.setPixmap(
            QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())

    def BeautifyGray(self):
        p = self.sliderBeautifyGray.value()/5
        blank = np.zeros(self.img.shape, np.uint8)
        self.img = cv2.addWeighted(self.img, p, blank, 1 - p, 0)
        self.display()

    def BeautifyGrayFromSlider(self):
        p = self.sliderBeautifyGray.value()/5
        blank = np.zeros(self.img.shape, np.uint8)
        img = cv2.addWeighted(self.img, p, blank, 1 - p,0)
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[-1],
                           QImage.Format_RGB888)
        self.lbDisplay.setPixmap(QPixmap.fromImage(showImage))  # .scaled(self.lbDisplay.width(),self.lbDisplay.height())

    def getEntropy(self, img):
        nColorNum = 256
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        lCountSum = height * width
        dProba = np.zeros(nColorNum, np.float)
        for i in range(height):
            for j in range(width):
                grayValue = img[i][j]
                dProba[grayValue] += 1
        for i in range(nColorNum):
            if dProba[i] == 0:
                continue
            dProba[i] /= lCountSum

        m_dEntropy = 0
        for i in range(nColorNum):
            if dProba[i] > 0:
                m_dEntropy -= dProba[i] * math.log(dProba[i]) / math.log(2.0)
        return m_dEntropy

    def EncodeHuffman(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        height, width, channel = img.shape
        nColorNum = 256
        lCountSum = height * width
        dProba = np.zeros(nColorNum, np.float)
        dTemp = np.zeros(nColorNum, np.float)
        for i in range(height):
            for j in range(width):
                grayValue = img[i][j][2]
                dProba[grayValue] += 1
        for i in range(nColorNum):
            dProba[i] /= lCountSum
            dTemp[i] = dProba[i]
        n4Turn = []
        for i in range(nColorNum):
            n4Turn.append(i)
        for j in range(nColorNum - 1):
            for i in range(nColorNum - j - 1):
                if dTemp[i] > dTemp[i + 1]:
                    dT = dTemp[i]
                    dTemp[i] = dTemp[i + 1]
                    dTemp[i + 1] = dT
                    for k in range(nColorNum):
                        if n4Turn[k] == i:
                            n4Turn[k] = i + 1
                        elif n4Turn[k] == i + 1:
                            n4Turn[k] = i
        nonzero = dTemp.nonzero()[0]
        min = nonzero.min()
        m_strCode = []
        for i in range(nColorNum):
            m_strCode.append('')
        for i in range(min, nColorNum - 1):
            for k in range(nColorNum):
                # 灰度值是否i
                if (n4Turn[k] == i):
                    # 灰度值较小的码字加1
                    m_strCode[k] = "1" + m_strCode[k]

                elif (n4Turn[k] == i + 1):
                    # 灰度值较小的码字加0
                    m_strCode[k] = "0" + m_strCode[k]
            dTemp[i + 1] += dTemp[i]
            for k in range(nColorNum):
                if n4Turn[k] == i:
                    n4Turn[k] = i + 1

            for j in range(i + 1, nColorNum - 1):
                if dTemp[j] > dTemp[j + 1]:
                    dT = dTemp[j]
                    dTemp[j] = dTemp[j + 1]
                    dTemp[j + 1] = dT

                    for k in range(nColorNum):
                        if n4Turn[k] == j:
                            n4Turn[k] = j + 1
                        elif n4Turn[k] == j + 1:
                            n4Turn[k] = j
                else:
                    break
        m_dEntropy = 0
        m_dCodLength = 0
        # 计算图像熵
        for i in range(nColorNum):
            if dProba[i] > 0:
                m_dEntropy -= dProba[i] * math.log(dProba[i]) / math.log(2.0)
        # 计算平均编码长度
        for i in range(nColorNum):
            if dProba[i] == 0:
                continue
            m_dCodLength += dProba[i] * len(m_strCode[i])
        # 计算编码效率
        m_dRatio = m_dEntropy / m_dCodLength
        ############### display ###############
        self.edEntropy.setText('%.5f' % m_dEntropy)
        self.edCodLength.setText('%.5f' % m_dCodLength)
        self.edRatio.setText('%.5f' % m_dRatio)
        for i in range(256):
            self.twHuffman.setItem(i, 0, QTableWidgetItem(str(i)))
            self.twHuffman.setItem(i, 1, QTableWidgetItem(str(dProba[i])))
            self.twHuffman.setItem(i, 2, QTableWidgetItem(m_strCode[i]))
            self.twHuffman.setItem(i, 3, QTableWidgetItem(str(len(m_strCode[i]))))

    def EncodeShanno(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        height, width, channel = img.shape
        nColorNum = 256
        lCountSum = height * width
        dProba = np.zeros(nColorNum, np.float)
        dTemp = np.zeros(nColorNum, np.float)
        for i in range(height):
            for j in range(width):
                grayValue = img[i][j][2]
                dProba[grayValue] += 1
        for i in range(nColorNum):
            dProba[i] /= lCountSum
            dTemp[i] = dProba[i]
        n4Turn = []
        for i in range(nColorNum):
            n4Turn.append(i)
        for j in range(nColorNum - 1):
            for i in range(nColorNum - j - 1):
                if dTemp[i] > dTemp[i + 1]:
                    dT = dTemp[i]
                    dTemp[i] = dTemp[i + 1]
                    dTemp[i + 1] = dT
                    for k in range(nColorNum):
                        if n4Turn[k] == i:
                            n4Turn[k] = i + 1
                        elif n4Turn[k] == i + 1:
                            n4Turn[k] = i
        nonzero = dTemp.nonzero()[0]
        min = nonzero.min()
        m_strCode = []
        for i in range(nColorNum):
            m_strCode.append('')
        dSum = 0
        for i in range(min, nColorNum - 1):
            if i != min:
                dSum += dTemp[i]
            if dTemp[i] == 0:
                continue
            codeLength = int(-math.log(dTemp[i]) + 0.5)
            p = 1
            t = 0.0
            while len(m_strCode[i]) < codeLength:
                t += pow(2, -p)
                if t < dSum:
                    m_strCode[i] += "1"
                else:
                    m_strCode[i] += "0"
                    t -= pow(2, -p)
                p += 1
            for k in range(nColorNum):
                # 灰度值是否i
                if (n4Turn[k] == i):
                    # 灰度值较小的码字加1
                    m_strCode[k] = "1" + m_strCode[k]

                elif (n4Turn[k] == i + 1):
                    # 灰度值较小的码字加0
                    m_strCode[k] = "0" + m_strCode[k]
        m_dEntropy = 0
        m_dCodLength = 0
        # 计算图像熵
        for i in range(nColorNum):
            if dProba[i] > 0:
                m_dEntropy -= dProba[i] * math.log(dProba[i]) / math.log(2.0)
        # 计算平均编码长度
        for i in range(nColorNum):
            if dProba[i] == 0:
                continue
            m_dCodLength += dProba[i] * len(m_strCode[i])
        # 计算编码效率
        m_dRatio = m_dEntropy / m_dCodLength
        ############### display ###############
        self.edEntropy.setText('%.5f' % m_dEntropy)
        self.edCodLength.setText('%.5f' % m_dCodLength)
        self.edRatio.setText('%.5f' % m_dRatio)
        self.twHuffman.clearContents()
        for i in range(256):
            self.twHuffman.setItem(i, 0, QTableWidgetItem(str(i)))
            self.twHuffman.setItem(i, 1, QTableWidgetItem(str(dProba[i])))
            self.twHuffman.setItem(i, 2, QTableWidgetItem(m_strCode[i]))
            self.twHuffman.setItem(i, 3, QTableWidgetItem(str(len(m_strCode[i]))))

    def EdgeDetect(self):
        method = self.cbEdgeDetect.currentText()
        height, width, channel = self.img.shape
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img = np.zeros(self.img.shape, np.uint8)
        if method == 'Robert':
            for i in range(height):
                for j in range(width):
                    if i == height - 1 or j == 0:
                        continue
                    dfx = self.img[i][j][2] - self.img[i + 1][j - 1][2].astype(np.int)
                    dfy = self.img[i][j - 1][2] - self.img[i + 1][j][2].astype(np.int)
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Sobel':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    dfx = self.img[i - 1][j - 1][2].astype(np.int) + 2 * self.img[i - 1][j][2].astype(np.int) + \
                          self.img[i - 1][j + 1][2].astype(np.int) \
                          - self.img[i + 1][j - 1][2].astype(np.int) - 2 * self.img[i + 1][j][2].astype(np.int) - \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfx /= 4
                    dfy = -self.img[i - 1][j - 1][2].astype(np.int) - 2 * self.img[i][j - 1][2].astype(np.int) - \
                          self.img[i + 1][j - 1][2].astype(np.int) \
                          + self.img[i - 1][j + 1][2].astype(np.int) + 2 * self.img[i][j + 1][2].astype(np.int) + \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfy /= 4
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Prewitt':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    dfx = self.img[i - 1][j - 1][2].astype(np.int) + self.img[i - 1][j][2].astype(np.int) + \
                          self.img[i - 1][j + 1][2].astype(np.int) \
                          - self.img[i + 1][j - 1][2].astype(np.int) - self.img[i + 1][j][2].astype(np.int) - \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfy = -self.img[i - 1][j - 1][2].astype(np.int) - self.img[i][j - 1][2].astype(np.int) - \
                          self.img[i + 1][j - 1][2].astype(np.int) \
                          + self.img[i - 1][j + 1][2].astype(np.int) + self.img[i][j + 1][2].astype(np.int) + \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfx /= 3
                    dfy /= 3
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Lapplacian':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    t = self.img[i][j + 1][2].astype(np.int) + self.img[i][j - 1][2].astype(np.int) + \
                        self.img[i + 1][j][2].astype(np.int) \
                        + self.img[i - 1][j][2].astype(np.int) - 4 * self.img[i][j][2].astype(np.int)
                    if t < 0:
                        t = 0
                    if t > 255:
                        t = 255
                    img[i][j] = t
                    # img[i][j] = -4*self.img[i][j].astype(np.int) + self.img[i-1][j-1].astype(np.int) + self.img[i-1][j+1].astype(np.int) + self.img[i+1][j-1].astype(np.int) + self.img[i+1][j+1].astype(np.int)
        # 计算原图信息熵
        self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
        Entropy = self.getEntropy(self.img)
        # self.edEntropyPre.setText('%.5f' % Entropy)
        Ar = Arithmetic(self.img, img)
        # self.img = Ar.Add(0,0,self.weight)
        self.imgTmp = Ar.Add(0, 0, self.weight)
        # 计算混合后信息熵
        Entropy = self.getEntropy(self.imgTmp)
        # self.edEntropyMix.setText('%.5f' % Entropy)


        graytf = GrayTransform(self.imgTmp)
        max = 255
        min = 0
        self.imgTmp = graytf.LinearTransform(max=max, min=min)
        self.display(self.imgTmp)

    def Outline(self):
        method = self.cbEdgeDetect.currentText()
        height, width, channel = self.img.shape
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img = np.zeros(self.img.shape, np.uint8)
        if method == 'Robert':
            for i in range(height):
                for j in range(width):
                    if i == height - 1 or j == 0:
                        continue
                    dfx = self.img[i][j][2] - self.img[i + 1][j - 1][2].astype(np.int)
                    dfy = self.img[i][j - 1][2] - self.img[i + 1][j][2].astype(np.int)
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Sobel':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    dfx = self.img[i - 1][j - 1][2].astype(np.int) + 2 * self.img[i - 1][j][2].astype(np.int) + \
                          self.img[i - 1][j + 1][2].astype(np.int) \
                          - self.img[i + 1][j - 1][2].astype(np.int) - 2 * self.img[i + 1][j][2].astype(np.int) - \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfx /= 4
                    dfy = -self.img[i - 1][j - 1][2].astype(np.int) - 2 * self.img[i][j - 1][2].astype(np.int) - \
                          self.img[i + 1][j - 1][2].astype(np.int) \
                          + self.img[i - 1][j + 1][2].astype(np.int) + 2 * self.img[i][j + 1][2].astype(np.int) + \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfy /= 4
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Prewitt':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    dfx = self.img[i - 1][j - 1][2].astype(np.int) + self.img[i - 1][j][2].astype(np.int) + \
                          self.img[i - 1][j + 1][2].astype(np.int) \
                          - self.img[i + 1][j - 1][2].astype(np.int) - self.img[i + 1][j][2].astype(np.int) - \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfy = -self.img[i - 1][j - 1][2].astype(np.int) - self.img[i][j - 1][2].astype(np.int) - \
                          self.img[i + 1][j - 1][2].astype(np.int) \
                          + self.img[i - 1][j + 1][2].astype(np.int) + self.img[i][j + 1][2].astype(np.int) + \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfx /= 3
                    dfy /= 3
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Lapplacian':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    t = self.img[i][j + 1][2].astype(np.int) + self.img[i][j - 1][2].astype(np.int) + \
                        self.img[i + 1][j][2].astype(np.int) \
                        + self.img[i - 1][j][2].astype(np.int) - 4 * self.img[i][j][2].astype(np.int)
                    if t < 0:
                        t = 0
                    if t > 255:
                        t = 255
                    img[i][j] = t
                    # img[i][j] = -4*self.img[i][j].astype(np.int) + self.img[i-1][j-1].astype(np.int) + self.img[i-1][j+1].astype(np.int) + self.img[i+1][j-1].astype(np.int) + self.img[i+1][j+1].astype(np.int)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
        graytf = GrayTransform(img)
        max = 255
        min = 0
        img = graytf.LinearTransform(max=max, min=min)
        self.tmpOutline = self.img.copy()
        thresh = self.sliderOutline.value()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j][0] >= thresh:
                    self.tmpOutline[i][j] = self.colorOutline
        self.display(self.tmpOutline)

    def OutlineFromSd(self):
        method = self.cbEdgeDetect.currentText()
        height, width, channel = self.img.shape
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img = np.zeros(self.img.shape, np.uint8)
        if method == 'Robert':
            for i in range(height):
                for j in range(width):
                    if i == height - 1 or j == 0:
                        continue
                    dfx = self.img[i][j][2] - self.img[i + 1][j - 1][2].astype(np.int)
                    dfy = self.img[i][j - 1][2] - self.img[i + 1][j][2].astype(np.int)
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Sobel':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    dfx = self.img[i - 1][j - 1][2].astype(np.int) + 2 * self.img[i - 1][j][2].astype(np.int) + \
                          self.img[i - 1][j + 1][2].astype(np.int) \
                          - self.img[i + 1][j - 1][2].astype(np.int) - 2 * self.img[i + 1][j][2].astype(np.int) - \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfx /= 4
                    dfy = -self.img[i - 1][j - 1][2].astype(np.int) - 2 * self.img[i][j - 1][2].astype(np.int) - \
                          self.img[i + 1][j - 1][2].astype(np.int) \
                          + self.img[i - 1][j + 1][2].astype(np.int) + 2 * self.img[i][j + 1][2].astype(np.int) + \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfy /= 4
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Prewitt':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    dfx = self.img[i - 1][j - 1][2].astype(np.int) + self.img[i - 1][j][2].astype(np.int) + \
                          self.img[i - 1][j + 1][2].astype(np.int) \
                          - self.img[i + 1][j - 1][2].astype(np.int) - self.img[i + 1][j][2].astype(np.int) - \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfy = -self.img[i - 1][j - 1][2].astype(np.int) - self.img[i][j - 1][2].astype(np.int) - \
                          self.img[i + 1][j - 1][2].astype(np.int) \
                          + self.img[i - 1][j + 1][2].astype(np.int) + self.img[i][j + 1][2].astype(np.int) + \
                          self.img[i + 1][j + 1][2].astype(np.int)
                    dfx /= 3
                    dfy /= 3
                    t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                    if t >= 255:
                        t = 255
                    img[i][j] = t
        elif method == 'Lapplacian':
            for i in range(height):
                for j in range(width):
                    if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                        continue
                    t = self.img[i][j + 1][2].astype(np.int) + self.img[i][j - 1][2].astype(np.int) + \
                        self.img[i + 1][j][2].astype(np.int) \
                        + self.img[i - 1][j][2].astype(np.int) - 4 * self.img[i][j][2].astype(np.int)
                    if t < 0:
                        t = 0
                    if t > 255:
                        t = 255
                    img[i][j] = t
                    # img[i][j] = -4*self.img[i][j].astype(np.int) + self.img[i-1][j-1].astype(np.int) + self.img[i-1][j+1].astype(np.int) + self.img[i+1][j-1].astype(np.int) + self.img[i+1][j+1].astype(np.int)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
        graytf = GrayTransform(img)
        max = 255
        min = 0
        img = graytf.LinearTransform(max=max, min=min)
        self.tmpOutline = self.img.copy()
        thresh = self.sliderOutline.value()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j][0] >= thresh:
                    self.tmpOutline[i][j] = self.colorOutline
        self.display(self.tmpOutline)

    def OutlineSave(self):
        self.img = self.tmpOutline
        self.display()
        # method = self.cbEdgeDetect.currentText()
        # height, width, channel = self.img.shape
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # img = np.zeros(self.img.shape, np.uint8)
        # if method == 'Robert':
        #     for i in range(height):
        #         for j in range(width):
        #             if i == height - 1 or j == 0:
        #                 continue
        #             dfx = self.img[i][j][2] - self.img[i + 1][j - 1][2].astype(np.int)
        #             dfy = self.img[i][j - 1][2] - self.img[i + 1][j][2].astype(np.int)
        #             t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
        #             if t >= 255:
        #                 t = 255
        #             img[i][j] = t
        # elif method == 'Sobel':
        #     for i in range(height):
        #         for j in range(width):
        #             if i == 0 or i == height - 1 or j == 0 or j == width - 1:
        #                 continue
        #             dfx = self.img[i - 1][j - 1][2].astype(np.int) + 2 * self.img[i - 1][j][2].astype(np.int) + \
        #                   self.img[i - 1][j + 1][2].astype(np.int) \
        #                   - self.img[i + 1][j - 1][2].astype(np.int) - 2 * self.img[i + 1][j][2].astype(np.int) - \
        #                   self.img[i + 1][j + 1][2].astype(np.int)
        #             dfx /= 4
        #             dfy = -self.img[i - 1][j - 1][2].astype(np.int) - 2 * self.img[i][j - 1][2].astype(np.int) - \
        #                   self.img[i + 1][j - 1][2].astype(np.int) \
        #                   + self.img[i - 1][j + 1][2].astype(np.int) + 2 * self.img[i][j + 1][2].astype(np.int) + \
        #                   self.img[i + 1][j + 1][2].astype(np.int)
        #             dfy /= 4
        #             t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
        #             if t >= 255:
        #                 t = 255
        #             img[i][j] = t
        # elif method == 'Prewitt':
        #     for i in range(height):
        #         for j in range(width):
        #             if i == 0 or i == height - 1 or j == 0 or j == width - 1:
        #                 continue
        #             dfx = self.img[i - 1][j - 1][2].astype(np.int) + self.img[i - 1][j][2].astype(np.int) + \
        #                   self.img[i - 1][j + 1][2].astype(np.int) \
        #                   - self.img[i + 1][j - 1][2].astype(np.int) - self.img[i + 1][j][2].astype(np.int) - \
        #                   self.img[i + 1][j + 1][2].astype(np.int)
        #             dfy = -self.img[i - 1][j - 1][2].astype(np.int) - self.img[i][j - 1][2].astype(np.int) - \
        #                   self.img[i + 1][j - 1][2].astype(np.int) \
        #                   + self.img[i - 1][j + 1][2].astype(np.int) + self.img[i][j + 1][2].astype(np.int) + \
        #                   self.img[i + 1][j + 1][2].astype(np.int)
        #             dfx /= 3
        #             dfy /= 3
        #             t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
        #             if t >= 255:
        #                 t = 255
        #             img[i][j] = t
        # elif method == 'Lapplacian':
        #     for i in range(height):
        #         for j in range(width):
        #             if i == 0 or i == height - 1 or j == 0 or j == width - 1:
        #                 continue
        #             t = self.img[i][j + 1][2].astype(np.int) + self.img[i][j - 1][2].astype(np.int) + \
        #                 self.img[i + 1][j][2].astype(np.int) \
        #                 + self.img[i - 1][j][2].astype(np.int) - 4 * self.img[i][j][2].astype(np.int)
        #             if t < 0:
        #                 t = 0
        #             if t > 255:
        #                 t = 255
        #             img[i][j] = t
        #             # img[i][j] = -4*self.img[i][j].astype(np.int) + self.img[i-1][j-1].astype(np.int) + self.img[i-1][j+1].astype(np.int) + self.img[i+1][j-1].astype(np.int) + self.img[i+1][j+1].astype(np.int)
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
        # graytf = GrayTransform(img)
        # max = 255
        # min = 0
        # img = graytf.LinearTransform(max=max, min=min)
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         if img[i][j][0]>=50:
        #             self.img[i][j] = self.colorOutline
        # self.display()

    def OutlineColor(self):
        c = QColorDialog.getColor()
        self.colorOutline = np.array([c.blue(),c.green(),c.red()])

    def ChooseSeed(self):
        self.lbDisplay.isShow = True
        self.pbSegRegionGrowChooseSeed.setEnabled(False)
        self.pbSegRegionGrowRealTime.setEnabled(False)

    def ChooseSeedRealTime(self):
        if self.signSegMode == 'Add':
            self.imgTmp = np.zeros(self.img.shape, np.uint8)
        elif self.signSegMode == 'Min':
            self.imgTmp = self.img.copy()
        self.signSeedChoose = True
        self.pbSegRegionGrowChooseSeed.setEnabled(False)

    def Sharpen(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img = self.img.copy()
        (height, width, channel) = self.img.shape
        for i in range(height):
            for j in range(width):
                if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                    continue
                dfx = self.img[i - 1][j - 1][2].astype(np.int) + self.img[i - 1][j][2].astype(np.int) + \
                      self.img[i - 1][j + 1][2].astype(np.int) \
                      - self.img[i + 1][j - 1][2].astype(np.int) - self.img[i + 1][j][2].astype(np.int) - \
                      self.img[i + 1][j + 1][2].astype(np.int)
                dfy = -self.img[i - 1][j - 1][2].astype(np.int) - self.img[i][j - 1][2].astype(np.int) - \
                      self.img[i + 1][j - 1][2].astype(np.int) \
                      + self.img[i - 1][j + 1][2].astype(np.int) + self.img[i][j + 1][2].astype(np.int) + \
                      self.img[i + 1][j + 1][2].astype(np.int)
                dfx /= 3
                dfy /= 3
                t = math.sqrt(dfx * dfx + dfy * dfy) + 0.5
                if t >= 255:
                    t = 255
                if t >= 128:
                    value = t + 100
                    if value >= 255:
                        value = 255
                    img[i][j][2] = value
        self.img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        self.display()

    def SegRecover(self):
        self.display()
        self.signSeedChoose = False
        self.pbSegRegionGrowChooseSeed.setEnabled(True)

    def SegSaveChange(self):
        self.img = self.imgTmp
        self.display()
        self.signSeedChoose = False
        self.pbSegRegionGrowChooseSeed.setEnabled(True)

    def SegRegionGrowing(self):
        self.lbDisplay.isShow = False
        Points = self.lbDisplay.getPoints()
        self.lbDisplay.clearPoints()
        # self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        mask = self.regionGrow(self.img, Points, self.thresh)
        if self.signSegMode == 'Add':
            pass
        elif self.signSegMode == 'Min':
            mask = np.uint8(mask == 0)
        self.img = self.img * mask
        self.display()
        self.pbSegRegionGrowChooseSeed.setEnabled(True)
        self.pbSegRegionGrowRealTime.setEnabled(True)

    def regionGrow(self, image, seeds, thresh):
        height, width, _ = image.shape
        seedMark = np.zeros(image.shape, np.uint8)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        posX = [-1, 0, 1, 1, 1, 0, -1, -1]
        posY = [-1, -1, -1, 0, 1, 1, 1, 0]
        while (len(seedList) > 0):
            point = seedList.pop(0)
            if point.x > width or point.x < 0 or point.y > height or point.y < 0:
                continue
            seedMark[point.y, point.x] = 1
            avg = int(img[point.y, point.x, 2])
            k = 1
            for i in range(8):
                tmpX = point.x + posX[i]
                tmpY = point.y + posY[i]
                if tmpX <= 0 or tmpY <= 0 or tmpX >= width - 1 or tmpY >= height - 1:
                    continue
                # grayDiff = getGrayDiff(img, point, Point(tmpX, tmpY))
                grayDiff = getGrayDiffPromote2(img, avg, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpY, tmpX].all() == 0:
                    seedMark[tmpY, tmpX] = 1
                    seedList.append(Point(tmpX, tmpY))
                    avg += int(img[tmpY, tmpX, 2])
                    k += 1
                if k != 0:
                    avg /= k
        return seedMark

    def Segthresh(self):
        # 选取T1
        T1 = 0
        T2 = 128
        imgHSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        while abs(T2 - T1) > 0.5:
            T1 = T2
            # 求G1 G2 平均灰度
            grayG1 = 0
            grayG2 = 0
            numG1 = 0
            numG2 = 0
            for i in range(imgHSV.shape[0]):
                for j in range(imgHSV.shape[1]):
                    if imgHSV[i][j][2] >= T1:
                        numG1 += 1
                        grayG1 += imgHSV[i][j][2]
                    else:
                        numG2 += 1
                        grayG2 += imgHSV[i][j][2]
            u1 = grayG1 / numG1
            u2 = grayG2 / numG2
            T2 = (u1 + u2) / 2
        self.img = 255 * np.uint8(imgHSV[:, :, 2] > T2)
        self.display()

    def SegAuto(self):
        frame_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 图像灰化，降低计算复杂度
        faceRects = self.cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        sx = []
        sy = []
        if len(faceRects > 0):
            for i in range(len(faceRects)):
                x, y, w, h = faceRects[i]
                sx.append(x + w / 2)
                sy.append(y + h / 2)
        else:
            return
        Points = []
        if len(sx) > 0:
            for i in range(len(sx)):
                Points.append(Point(int(sx[i]), int(sy[i])))
            img = self.regionGrow(self.img, Points, self.thresh)
            mask = self.regionGrow(self.img, Points, self.thresh)
            if self.signSegMode == 'Add':
                pass
            elif self.signSegMode == 'Min':
                mask = np.uint8(mask == 0)
            self.img = self.img * mask
            self.display()

    # 滤镜-赛博朋克、浮雕、复古
    def filterCyberpunk(self):
        image = self.img
        # 反转色相
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        image_hls = np.asarray(image_hls, np.float32)
        hue = image_hls[:, :, 0]
        hue[hue < 90] = 180 - hue[hue < 90]
        image_hls[:, :, 0] = hue

        image_hls = np.asarray(image_hls, np.uint8)
        image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2BGR)

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        image_lab = np.asarray(image_lab, np.float32)

        # 提高像素亮度，让亮的地方更亮
        light_gamma_high = np.power(image_lab[:, :, 0], 0.8)
        light_gamma_high = np.asarray(light_gamma_high / np.max(light_gamma_high) * 255, np.uint8)

        # 降低像素亮度，让暗的地方更暗
        light_gamma_low = np.power(image_lab[:, :, 0], 1.2)
        light_gamma_low = np.asarray(light_gamma_low / np.max(light_gamma_low) * 255, np.uint8)

        # 调色至偏紫
        dark_b = image_lab[:, :, 2] * (light_gamma_low / 255) * 0.1
        dark_a = image_lab[:, :, 2] * (1 - light_gamma_high / 255) * 0.3

        image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_b, 0, 255)
        image_lab[:, :, 2] = np.clip(image_lab[:, :, 2] - dark_a, 0, 255)

        image_lab = np.asarray(image_lab, np.uint8)
        self.img = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
        self.display()

    def Embossment(self):
        method = self.cbEdgeDetect.currentText()
        height, width, channel = self.img.shape
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img = np.zeros(self.img.shape, np.uint8)
        for i in range(height):
            for j in range(width):
                if i == height - 1 or j == width - 1:
                    continue
                t = self.img[i][j][2] - self.img[i + 1][j + 1][2].astype(np.int)
                t += 128
                if t > 255:
                    t = 255
                if t < 0:
                    t = 0
                img[i][j] = t
        self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
        edgeMask = np.uint8(img == img.max())
        for i in range(height):
            for j in range(width):
                if edgeMask[i, j].all() == 0:
                    pass
                else:
                    img[i, j] = self.img[i, j]
        self.img = img
        self.display()

    def filterRetro(self):
        rows, cols = self.img.shape[:2]  # 创建高斯滤波器
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        filter = 255 * kernel / np.linalg.norm(kernel)
        vintage_im = np.copy(self.img)  # 对于输入图像中的每个通道，我们将应用上述滤波器
        for i in range(3):
            vintage_im[:, :, i] = vintage_im[:, :, i] * filter
        self.img = vintage_im
        self.display()

    def filterOilPainting(self):
        self.img = oilpaint(self.img)
        self.display()

    # 风格迁移
    def NST(self):
        model = self.cbNSTModel.currentText()
        model = 'NST/{}.pth'.format(model)
        out = 'NSTtmp.jpg'
        cuda = 1  # GPU

        content_image = utils_NST.tensor_load_rgbimage(self.img, scale=None)
        content_image = content_image.unsqueeze(0)

        if cuda:
            content_image = content_image.cuda()
        with torch.no_grad():
            content_image = Variable(utils_NST.preprocess_batch(content_image))
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(model))

        if cuda:
            style_model.cuda()

        output = style_model(content_image)
        utils_NST.tensor_save_bgrimage(output.data[0], out, cuda)

        self.img = cv2.imread('NSTtmp.jpg')
        self.display()

    # 图像修复
    def inPaint(self):
        if self.signInpaint == 1:
            self.lbDisplay.signDrawRect = True
            self.pbInpaint.setText('下一步')
            self.signInpaint = 2
            self.lbTips.setText("Tips:鼠标框选待修复区域")
        elif self.signInpaint == 2:
            self.lbTips.setText("Tips:调整阈值，直到污渍被完美分割")
            self.lbDisplay.signDrawRect = False
            (x0,x1,y0,y1) = self.lbDisplay.getRect()
            if x0 >= x1:
                tmp = x0
                x0 = x1
                x1 = tmp
            if y0 >= y1:
                tmp = y0
                y0 = y1
                y1 = tmp
            self.inPaintRect = (x0,x1,y0,y1)
            imgRect = self.img[y0:y1,x0:x1]
            self.tmpInpaint = imgRect
            self.display(self.tmpInpaint)
            self.pbInpaint.setText('处理')
            self.signInpaint = 3
        elif self.signInpaint == 3:
            rect = self.inPintTool(self.tmpInpaint,self.threshInpaint)
            (x0, x1, y0, y1) = self.inPaintRect
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    if i>y0 and i<y1 and j>x0 and j<x1:
                        self.img[i][j] = rect[i-y0][j-x0]
                    else:
                        pass
            self.display()
            self.signInpaint = 1
            self.lbTips.setText("Tips:点击开始")

    def inPintThreshChangeFromSd(self):
        self.threshInpaint = self.sliderInpaint.value()
        self.edThreshInpaint.setText(str(self.threshInpaint))
        threshImg = cv2.inRange(self.tmpInpaint, np.array([self.threshInpaint, self.threshInpaint, self.threshInpaint]), np.array([255, 255, 255]))
        self.display(threshImg)

    def inPaintThreshChangeFromEd(self):
        self.threshInpaint = int(self.edThreshInpaint.text())
        self.sliderInpaint.setValue(int(self.threshInpaint))
        threshImg = cv2.inRange(self.tmpInpaint, np.array([self.threshInpaint, self.threshInpaint, self.threshInpaint]),
                                np.array([255, 255, 255]))
        self.display(threshImg)

    def inPintTool(self,img,thresh):
        hight, width, depth = img.shape[0:3]
        # 图片二值化处理
        threshImg = cv2.inRange(img, np.array([thresh, thresh, thresh]), np.array([255, 255, 255]))
        # 创建形状和尺寸的结构元素
        kernel = np.ones((3, 3), np.uint8)
        # 扩张待修复区域
        hi_mask = cv2.dilate(threshImg, kernel, iterations=1)
        specular = cv2.inpaint(img, hi_mask, 5, flags=cv2.INPAINT_TELEA)
        return specular





    def AutoBeautify(self):
        frame_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 图像灰化，降低计算复杂度
        faceRects = self.cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        sx = []
        sy = []
        if len(faceRects > 0):
            for i in range(len(faceRects)):
                x, y, w, h = faceRects[i]
                sx.append(x + w / 2)
                sy.append(y + h / 2)
        else:
            return
        Points = []
        if len(sx) > 0:
            for i in range(len(sx)):
                Points.append(Point(int(sx[i]), int(sy[i])))
            maskFaceRegion = self.regionGrow(self.img, Points, self.thresh)  # 人脸区域1，其余为0
            maskOtherRegion = np.uint8(maskFaceRegion == 0)  # 人脸区域0，其余为1

            tmpFaceRegion = self.img * maskFaceRegion  # 扣出人脸区域

            p = self.sliderAutoBeautiftyValue.value()  # 美化人脸区域
            p = 1 - p / 100
            tmpFaceRegion = self.Buffing(p,tmpFaceRegion)
            tmpFaceRegion *= maskFaceRegion  # 结果：人脸区域美白,其余区域为0

            tmpOtherRegion = self.img*maskOtherRegion  # 扣出其他区域

            self.img = tmpFaceRegion+tmpOtherRegion
            self.display()
    # def AutoBeautify(self):
    #     # self.HistorgramEqual()  # 直方图均衡化
    #     frame_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 图像灰化，降低计算复杂度
    #     faceRects = self.cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    #     x = 0
    #     y = 0
    #     w = 0
    #     h = 0
    #     if len(faceRects) > 0:
    #         x, y, w, h = faceRects[0]
    #         img = self.img[y: y + h, x: x + w]
    #         dst = np.zeros_like(img)
    #         # int value1 = 3,value2 = 1;磨皮程度与细节程度
    #         value1 = 3
    #         value2 = 1
    #         dx = value1 * 12  # 双边滤波参数一
    #         fc = value1 * 8  # 双边滤波参数二
    #         p = 0.3  # 图片融合比例(透明度)
    #         temp4 = np.zeros_like(img)
    #         temp1 = cv2.bilateralFilter(img, dx, fc * 2, fc / 2)  # 双边滤波EPFFilter(Src)
    #         temp2 = cv2.subtract(temp1, img)  # EPFFilter(Src)-Src
    #         temp2 = cv2.add(temp2, (10, 10, 10, 128))  # EPFFilter(Src)-Src+128
    #         temp3 = cv2.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0, 0)  # 高斯模糊
    #         temp3 = 2 * temp3  # 2*GuassianBlur
    #         temp4 = cv2.add(img, temp3)  # Src+2*GuassianBlur
    #         dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
    #         dst = cv2.add(dst, (10, 10, 10, 255))
    #         img_new = np.zeros(self.img.shape, np.uint8)
    #         for i in range(img_new.shape[0]):
    #             for j in range(img_new.shape[1]):
    #                 if i > y and i < y + h and j > x and j < x + w:
    #                     img_new[i][j] = dst[i - y][j - x]
    #                 else:
    #                     img_new[i][j] = self.img[i][j]
    #         self.img = img_new
    #     else:
    #         pass
    #     self.display()

    # recall
    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag and not self.sign:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()
        if self.sign:
            newPos = self.lbDisplay.mapFromGlobal(QCursor.pos())
            x = newPos.x()
            y = newPos.y()
            if newPos.y() > (self.img1.shape[0] - self.img2.shape[0]):
                y = (self.img1.shape[0] - self.img2.shape[0])
            if newPos.y() < 0:
                y = 0
            if newPos.x() < 0:
                x = 0
            if newPos.x() > (self.img1.shape[1] - self.img2.shape[1]):
                x = (self.img1.shape[1] - self.img2.shape[1])
            self.lbAbove.move(x, y)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.signSeedChoose:
                newPos = self.lbDisplay.mapFromGlobal(QCursor.pos())
                x = newPos.x()
                y = newPos.y()
                mask = self.regionGrow(self.img, [Point(x, y)], self.thresh)
                if self.signSegMode == 'Add':
                    for i in range(self.img.shape[0]):
                        for j in range(self.img.shape[1]):
                            if mask[i][j].all() != 0:
                                self.imgTmp[i][j] = self.img[i][j]
                    self.display(self.imgTmp, 1)
                elif self.signSegMode == 'Min':
                    # mask = np.uint8(mask == 0)
                    for i in range(self.img.shape[0]):
                        for j in range(self.img.shape[1]):
                            if mask[i][j].all() != 0:
                                self.imgTmp[i][j] = 0
                    self.display(self.imgTmp)
                # self.imgTmp = self.img * mask

            else:
                self.m_flag = True
                self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
                event.accept()
                self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyApp()
    my_pyqt_form.show()
    sys.exit(app.exec_())
