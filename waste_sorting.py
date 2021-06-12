import io
import logging
import sys
import threading
from time import sleep

import keras
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QCoreApplication, QThread, QSize, pyqtSignal, Qt, QBasicTimer
from PyQt5.QtWidgets import (QWidget, QToolTip,
                             QPushButton, QApplication, QMessageBox, QDesktopWidget, QHBoxLayout, QFileDialog,
                             QVBoxLayout, QLabel, QAction, QMenu, QScrollArea, QGridLayout, QSplashScreen)
from PyQt5.QtGui import QFont, QCursor, QPixmap

import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from scipy.spatial import transform

# 可能的结果
item_list = ["衣服", "玻璃", "口服液瓶", "口罩", "牙刷", "尿布", "鞋", "塑料瓶", '报纸', "易拉罐", "纸箱", "果蔬", "电池", "打火机", "废弃灯泡",
             "花盆", "卫生纸", "纱布", "棉签", "橡胶手套", "化妆刷", "创可贴", '一次性筷子', "熟食", "塑料袋", "手机", "笔", "笔记本", "背包", "键盘", "电子表",
             "电脑", "鼠标", "杯子", "磁带", "帽子", "电子闹钟", "皮带", "袜子", "刀"]

waste_sorting_model = keras.Model


# 模型加载
class Model:
    def __init__(self):
        # 1 为解析 h5 模型，2为解析pb模型,3为移动端模型
        self.Tag = 2

    def load_model(self):
        global waste_sorting_model

        try:
            # 加载 tensorflow pb 模型
            if self.Tag == 1:
                # 加载 keras h5模型
                waste_sorting_model = load_model('dataset/model/densenet-121-garbage.h5')
            elif self.Tag == 2:
                waste_sorting_model = load_model('dataset/model/Bagging-5-MobileNetV3')
            elif self.Tag == 3:
                waste_sorting_model = load_model('dataset/model/base-MobileNetV3-0.h5')

        except Exception as ex:
            logging.error(ex)

    def load_weight(self):
        global waste_sorting_model
        # 预处理
        image = cv2.imread("dataset/pic/pre_load_img.jpg")

        # resize图片大小 将原本的大小 ---> (224,224,3)
        image = cv2.resize(image, (224, 224))
        # 转换np数组格式
        image = np.array(image)
        # 转换成指定格式
        pre_load_img = (image.reshape(1, 224, 224, 3)).astype('int32') / 255
        waste_sorting_model.predict(pre_load_img)

        print("加载权重成功")


# 加载页面
class SplashPanel(QSplashScreen):
    def __init__(self):

        super(SplashPanel, self).__init__()
        message_font = QFont()
        message_font.setBold(True)
        message_font.setPointSize(14)
        self.setFont(message_font)
        pixmap = QPixmap("dataset/pic/load_logo.jpg")
        # pixmap = QPixmap("D:\\github\\bdmaster\\app\\resource\\images\\timg.png")
        self.setPixmap(pixmap)
        # self.showMessage('正在加载文件资源', alignment=Qt.AlignBottom, color=Qt.black)
        self.show()

        # ------------- 加载权重 -------------
        model = Model()
        model.load_model()
        model.load_weight()

        for i in range(1, 5):
            self.showMessage('正在加载文件资源{}'.format('.' * i), alignment=Qt.AlignBottom, color=Qt.black)
            sleep(1)

    def mousePressEvent(self, evt):
        pass
        # 重写鼠标点击事件，阻止点击后消失

    def mouseDoubleClickEvent(self, *args, **kwargs):
        pass
        # 重写鼠标移动事件，阻止出现卡顿现象

    def enterEvent(self, *args, **kwargs):
        pass
        # 重写鼠标移动事件，阻止出现卡顿现象

    def mouseMoveEvent(self, *args, **kwargs):
        pass
        # 重写鼠标移动事件，阻止出现卡顿现象


# ------------------------------ 软件主界面 ------------------------------
class Main(QWidget):
    imgNames = []
    predict_results = []

    def __init__(self):
        super().__init__()

        # 统计当前图片总数
        self.pic_count = 0

        # 矩阵行列数
        self.row = 0
        self.col = -1

        # 初始化ui
        self.initUI()

    def initUI(self):
        # 获取屏幕大小
        self.desktop = QApplication.desktop()
        self.height = self.desktop.height()
        self.width = self.desktop.width()

        # 图片大小
        self.display_image_size = 500

        # 使用10px滑体字体。
        QToolTip.setFont(QFont('SansSerif', 10))

        # 创建一个提示
        self.setToolTip('Welcome to use out <b>Waste Sorting</b> program')

        # 创建滚动条
        self.scroll_area_images = QScrollArea(self)
        self.scroll_area_images.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget(self)
        self.scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContends')
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents)
        self.scroll_area_images.setWidget(self.scrollAreaWidgetContents)
        self.scroll_area_images.setGeometry(50, 50, int(self.width * 0.7), int(self.height * 0.7))
        self.vertocall = QVBoxLayout()
        self.vertocall.addWidget(self.scroll_area_images)

        # btns
        upload_btn = QPushButton('Upload', self)
        upload_btn.setToolTip('Click this button to <b>upload pictures</b>')
        upload_btn.resize(upload_btn.sizeHint())
        upload_btn.clicked.connect(self.upload)

        predict_btn = QPushButton('Predict', self)
        predict_btn.setToolTip('Click this button to <b>predict pictures</b> and get result')
        predict_btn.resize(predict_btn.sizeHint())
        predict_btn.clicked.connect(self.predict_thread)

        quit_btn = QPushButton('Quit', self)
        quit_btn.clicked.connect(QCoreApplication.instance().quit)
        quit_btn.resize(quit_btn.sizeHint())

        btn_hbox = QHBoxLayout()
        btn_hbox.addStretch(1)
        btn_hbox.addWidget(upload_btn)
        btn_hbox.addWidget(predict_btn)
        btn_hbox.addWidget(quit_btn)

        self.vertocall.addLayout(btn_hbox)

        self.setLayout(self.vertocall)

        self.resize(int(self.width * 0.85), int(self.height * 0.8))
        self.center()
        self.setWindowTitle('Waste-Sorting')
        self.show()

    def center(self):
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def openImage(self):
        try:
            self.imgNames, imgType = QFileDialog.getOpenFileNames(self, "打开图片", "",
                                                                  "*.jpg;;*.png;;*.jpeg;;All Files(*)")
            images = []
            for imgName in self.imgNames:
                # 读取图片
                image = cv2.imread(imgName)
                # resize图片大小 将原本的大小 ---> (224,224,3)
                image = cv2.resize(image, (224, 224))
                # 转换np数组格式
                image = np.array(image)
                # 转换成指定格式
                img = (image.reshape(1, 224, 224, 3)).astype('int32') / 255
                # 加入到list中
                images.append(img)

            return images

        except Exception as ex:
            logging.error(ex)

    def upload(self):
        try:
            # 打开图片
            self.imgs = self.openImage()

            if self.imgs:
                self.start_img_viewer()

        except Exception as ex:
            logging.error(ex)

    def predict_thread(self):
        # 多线程，避免程序因预测图片过久导致程序长时间无响应
        t = threading.Thread(target=self.predict)
        t.setDaemon(True)
        t.start()

    def predict(self):
        global waste_sorting_model
        try:
            # 预测样本类别
            predictions = []

            for img in self.imgs:
                prediction = waste_sorting_model.predict(img)
                # 将所有结果存入数组
                self.predict_results.append(prediction)
                # 提取可能性最高的结果
                prediction = np.argmax(prediction, axis=1)
                predictions.append(prediction)

            count = 0
            for prediction in predictions:
                print(item_list[prediction[0]])
                # 更新标签
                self.set_predict_result(item_list[prediction[0]], count)
                count += 1

        except Exception as ex:
            logging.error(ex)

    # 初始化滚动栏
    def clear_layout(self):
        # 初始化
        self.row = 0
        self.col = -1

        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()

    # 刷新列表
    def start_img_viewer(self):
        # 清空layout
        self.clear_layout()

        # 获取图片数量
        photo_num = len(self.imgNames)
        if photo_num != 0:
            for i in range(photo_num):
                image_id = self.imgNames[i].replace('/', '\\')
                pixmap = QPixmap(image_id)
                self.addImage(pixmap, image_id)
                # 实时加载，可能图片加载数量比较多
                QApplication.processEvents()
        else:
            QMessageBox.information(self, '提示', '未选择图片')

    def get_nr_of_image_columns(self):
        # 展示图片的区域，计算每排显示图片数。返回的列数-1是因为我不想频率拖动左右滚动条，影响数据筛选效率
        scroll_area_images_width = int(0.68 * self.width)
        if scroll_area_images_width > self.display_image_size:
            # 计算出一行几列；
            pic_of_columns = scroll_area_images_width // self.display_image_size
        else:
            pic_of_columns = 1

        return pic_of_columns - 1

    def addImage(self, pixmap, image_id):
        # 获取图片列数
        nr_of_columns = self.get_nr_of_image_columns()
        nr_of_widgets = self.gridLayout.count()
        self.max_columns = nr_of_columns
        if self.col < self.max_columns:
            self.col += 1
        else:
            self.col = 0
            self.row += 1
        clickable_image = QClickableImage(self.display_image_size, self.display_image_size, pixmap, image_id)
        # clickable_image.clicked.connect(self.on_left_clicked)
        clickable_image.rightClicked.connect(self.on_right_clicked)
        clickable_image.setFixedSize(QSize(500, 500))

        self.gridLayout.addWidget(clickable_image, self.row, self.col)

    def set_predict_result(self, text, index):
        try:
            # 获取图片列数
            nr_of_columns = self.get_nr_of_image_columns() + 1

            x = index // nr_of_columns
            y = index % nr_of_columns

            self.gridLayout.itemAtPosition(x, y).widget().setText(text)

        except Exception as ex:
            logging.error(ex)

    def on_right_clicked(self):
        return


# ------------------------------ 可右键标签 ------------------------------
class PicLabel(QLabel):
    global PIC_value, PIC_dict

    def __init__(self, pixmap=None, image_id=None):
        QLabel.__init__(self)
        self.pixmap = pixmap
        self.image_id = image_id
        self.setPixmap(pixmap)

        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        # 开放右键策略
        self.customContextMenuRequested.connect(self.rightMenuShow)

    def rightMenuShow(self, point):
        # 添加右键菜单
        self.popMenu = QMenu()
        de = QAction(u'详细信息', self)
        sc = QAction(u'删除', self)
        xs = QAction(u'不知道啥功能，再说吧', self)
        self.popMenu.addAction(de)
        self.popMenu.addAction(sc)
        self.popMenu.addAction(xs)
        # 绑定事件
        de.triggered.connect(self.detail_info)
        sc.triggered.connect(self.delete)
        xs.triggered.connect(self.rshow)
        self.showContextMenu(QCursor.pos())

    def rshow(self):
        '''
        do something
        '''

    def delete(self):
        '''
        do something
        '''

    def detail_info(self):
        # 显示predict详细信息
        '''
        do something
        '''

    def showContextMenu(self, pos):
        # 调整位置
        '''''
        右键点击时调用的函数
        '''
        # 菜单显示前，将它移动到鼠标点击的位置

        self.popMenu.move(pos)
        self.popMenu.show()

    def menuSlot(self, act):
        print(act.text())


# ------------------------------ 可点击图片 ------------------------------
class QClickableImage(QWidget):
    # 图片路径地址
    image_id = ''
    info = []

    def __init__(self, width=0, height=0, pixmap=None, image_id=''):
        QWidget.__init__(self)

        self.width = width
        self.height = height
        self.pixmap = pixmap

        self.layout = QVBoxLayout(self)
        self.lable2 = QLabel()
        self.lable2.setObjectName('label2')

        if self.width and self.height:
            self.resize(self.width, self.height)
        if self.pixmap and image_id:
            pixmap = self.pixmap.scaled(QSize(self.width, self.height))

            self.label1 = PicLabel(pixmap, image_id)
            self.label1.setObjectName('label1')
            self.label1.setScaledContents(True)
            # self.label1.connect(self.mouseressevent())
            self.layout.addWidget(self.label1)

        if image_id:
            self.image_id = image_id
            self.lable2.setText(image_id.split('\\')[-1])
            self.lable2.setAlignment(Qt.AlignCenter)
            # 让文字自适应大小
            self.lable2.adjustSize()
            self.layout.addWidget(self.lable2)
        self.setLayout(self.layout)

    clicked = pyqtSignal(object)
    rightClicked = pyqtSignal(object)

    def imageId(self):
        return self.image_id

    def setText(self, text):
        self.lable2.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    splash = SplashPanel()
    app.processEvents()

    ex = Main()
    ex.show()

    splash.finish(ex)
    splash.deleteLater()

    sys.exit(app.exec_())
