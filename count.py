import argparse
import random
import sys
import torch
import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
# 显示图片
from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt
from qt_material import apply_stylesheet
from main_window_camera import Ui_MainWindow
from threading import Thread

import argparse
import os
import sys
import shutil
from pathlib import Path

import cv2
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOTS = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_sync

import camera_rc

# 已有功能：
# 权重加载，权重初始化
# 图像导入，图像检测，图像结果展示，图像导出，图像检测退出
# 视频导入，视频检测，视频结果展示，视频导出，视频检测退出
# 摄像头导入，摄像头检测，摄像头结果展示，摄像头导出，摄像头检测退出
# 检测时间，检测目标数，检测目标类别，检测目标位置信息

def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.input_width = self.input.width()
        self.input_height = self.input.height()
        self.output_width = self.output.width()
        self.output_height = self.output.height()
        self.imgsz = 640
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer_c = QTimer(self)
        self.timer_c.timeout.connect(self.detect_camera)
        self.video = None
        self.out = None
        self.device = "cuda:0"
        self.num_stop = 1
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.results = []
        self.camera = None
        self.running = False
        self.bind_slots()
        self.init_icons()

    def open_image(self):
        self.timer.stop()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "./", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)
        if self.file_path:
            dialog = QFileDialog(self, "Open File", self.file_path)
            dialog.resize(800, 600)
            dialog.close()
            self.input.setPixmap(QPixmap(self.file_path))
            self.lineEdit.setText('图片打开成功！！！')

    def open_video(self):
        self.timer.stop()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select vidos", dir='./', filter="Videos (*.mp4 *.avi *.gif *.MPEG)", options=options)
        if self.video_path:
            dialog = QFileDialog(self, "Open File", self.video_path)
            dialog.resize(800, 600)
            dialog.close()
            self.video_path = self.video_path
            self.video = cv2.VideoCapture(self.video_path)

            # 读取一帧用于展示
            ret, frame = self.video.read()
            if ret:
                self.lineEdit.setText("成功打开视频！！！")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dst_size = (self.input_width, self.input_height)
                resized_frame = cv2.resize(frame, dst_size, interpolation=cv2.INTER_AREA)
                self.input.setPixmap(QPixmap(convert2QImage(resized_frame)))
            else:
                self.lineEdit.setText("视频有误，请重新打开！！！")
            self.out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (int(self.video.get(3)), int(self.video.get(4))))

    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.button_weight_select, '选择权重文件',
                                                                  'weights/', "Weights (*.pt *.onnx *.engine)", options=options)
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"权重打开失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            dialog = QFileDialog(self, "Open File", self.openfile_name_model)
            dialog.resize(800, 600)
            dialog.close()
            result_str = '成功加载模型权重, 权重地址: ' + str(self.openfile_name_model)
            self.lineEdit.setText(result_str)

    def init_model(self):
        self.weights_path = str(self.openfile_name_model)
        self.device = select_device(self.device)
        # self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True
        self.model = DetectMultiBackend(self.weights_path, device=self.device)  # load FP32 model
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        print("model initial done")

        QtWidgets.QMessageBox.information(self, u"!", u"模型初始化成功", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        self.lineEdit.setText("成功初始化模型!!!")

    def detect_begin(self):
        img = cv2.imread(self.file_path)
        self.img_showimg = img
        name_list = []
        with torch.no_grad():
            t1 = time_sync()
            img = letterbox(img, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            self.pred = self.model(img)[0]
            # Apply NMS
            self.pred = non_max_suppression(self.pred, self.numcon, self.numiou, max_det=1000)
            t2 = time_sync()
            self.lineEdit_detect_time.setText(str(round(t2 - t1, 2)))
            self.lineEdit_detect_object_nums.setText(str(self.pred[0].shape[0]))

            self.results = self.pred[0].tolist()
            if self.pred[0].shape[0]:
                for i in range(self.pred[0].shape[0]):
                    self.comboBox.addItem('目标' + str(i + 1))
            # self.lineEdit.setText(str(self.pred))

            # Process detections
            for i, det in enumerate(self.pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], self.img_showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, self.img_showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)

            QtWidgets.QMessageBox.information(self, u"!", u"成功检测图像", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            self.lineEdit.setText("成功检测图像!!!")
            # cv2.imwrite('prediction.jpg', self.img_showimg)

    def convert2QImage(img):
        height, width, channel = img.shape
        return QImage(img, width, height, width * channel, QImage.Format_RGB888)

    def detect_show(self):
        results = self.pred[0].tolist()
        bbox = []
        conf = []
        cls = []
        for j in results:
            box_tmp = []
            conf_tmp = []
            cls_tmp = []
            for i in j[:4]:
                box_tmp.append(i)
            conf_tmp.append(j[4])
            cls_tmp.append(int(j[5]))
            bbox.append(box_tmp)
            conf.append(conf_tmp)
            cls.append(cls_tmp)

        self.result = cv2.cvtColor(self.img_showimg, cv2.COLOR_BGR2BGRA)
        self.QtImg = QtGui.QImage(
            self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
        self.output.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.output.setScaledContents(True)  # 自适应界面大小
        self.lineEdit.setText('图片检测成功！！！')

    # 视频检测
    def detect_video(self):
        self.timer.start()
        ret, frame = self.video.read()
        if not ret:
            self.timer.stop()
            self.video.release()
            self.out.release()
        else:
            name_list = []
            result_input = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            self.QtImg_input = QtGui.QImage(
                result_input.data, result_input.shape[1], result_input.shape[0], QtGui.QImage.Format_RGB32)
            self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg_input))
            self.input.setScaledContents(True)  # 自适应界面大小

            self.comboBox.clear()

            with torch.no_grad():
                t1 = time_sync()
                frame_showing = frame
                frame = letterbox(frame, new_shape=self.imgsz)[0]
                frame = frame[:, :, ::-1].transpose(2, 0, 1)
                frame = np.ascontiguousarray(frame)
                frame = torch.from_numpy(frame).to(self.device)
                frame = frame.float()  # uint8 to fp16/32
                frame /= 255.0  # 0 - 255 to 0.0 - 1.0
                if frame.ndimension() == 3:
                    frame = frame.unsqueeze(0)
                # Inference
                self.preds = self.model(frame)[0]
                # Apply NMS
                self.preds = non_max_suppression(self.preds, self.numcon, self.numiou, max_det=1000)
                t2 = time_sync()
                self.lineEdit_detect_time.setText(str(round(t2 - t1, 2)))
                self.lineEdit_detect_object_nums.setText(str(self.preds[0].shape[0]))
                self.results = self.preds[0].tolist()
                if self.preds[0].shape[0]:
                    for i in range(self.preds[0].shape[0]):
                        self.comboBox.addItem('目标' + str(i+1))

                # Process detections
                for i, det in enumerate(self.preds):
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(
                            frame.shape[2:], det[:, :4], frame_showing.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, frame_showing, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            self.out.write(frame_showing)
            result = cv2.cvtColor(frame_showing, cv2.COLOR_BGR2BGRA)
            self.QtImg = QtGui.QImage(
                result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB32)
            self.output.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.output.setScaledContents(True)  # 自适应界面大小

    def suspend_video(self):
        self.timer.blockSignals(False)
        if self.timer.isActive() == True and self.num_stop % 2 == 1:
            self.button_video_suspend.setText(u'继续视频检测')  # 当前状态为暂停状态
            self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
            self.timer.blockSignals(True)
        else:
            self.num_stop = self.num_stop + 1
            self.button_video_suspend.setText(u'暂停视频检测')

    def stop_video(self):
        if self.num_stop % 2 == 0:
            self.video.release()
            self.out.release()
            self.input.setPixmap(QPixmap("input.png"))
            self.input.setScaledContents(True)
            self.output.setPixmap(QPixmap("input.png"))
            self.output.setScaledContents(True)
            self.button_video_suspend.setText(u'暂停视频检测')
            self.num_stop = self.num_stop + 1
            self.timer.blockSignals(False)
            self.lineEdit_detect_time.clear()
            self.lineEdit_detect_object_nums.clear()
            self.lineEdit_xmin.clear()
            self.lineEdit_ymin.clear()
            self.lineEdit_xmax.clear()
            self.lineEdit_ymax.clear()
            self.lineEdit.clear()
        else:
            self.video.release()
            self.out.release()
            self.input.clear()
            self.output.clear()
            self.timer.blockSignals(False)
            self.lineEdit_detect_time.clear()
            self.lineEdit_detect_object_nums.clear()
            self.lineEdit_xmin.clear()
            self.lineEdit_ymin.clear()
            self.lineEdit_xmax.clear()
            self.lineEdit_ymax.clear()
            self.lineEdit.clear()

    def stop_image(self):
        self.input.setPixmap(QPixmap("input.png"))
        self.input.setScaledContents(True)
        self.output.setPixmap(QPixmap("input.png"))
        self.output.setScaledContents(True)
        self.lineEdit_detect_time.clear()
        self.lineEdit_detect_object_nums.clear()
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()
        self.comboBox.clear()
        self.lineEdit.clear()

    def export_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.OutputDir, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "导出图片",  # 标题
            r".",  # 起始目录
            "图片类型 (*.jpg *.jpeg *.png *.bmp)",  # 选择类型过滤项，过滤内容在括号中
            options=options
        )

        if self.output == "":
            QtWidgets.QMessageBox.warning(self, '提示', '请先选择图片保存的位置')
        else:
            try:
                dialog = QFileDialog(self, "Save image", self.OutputDir)
                dialog.resize(800, 600)
                dialog.close()
                cv2.imwrite(self.OutputDir, self.img_showimg)
                QtWidgets.QMessageBox.warning(self, '提示', '导出成功!')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, '提示', '请先完成识别工作')
                print(e)

    def export_videos(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.OutputDirs, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "导出视频",  # 标题
            r".",  # 起始目录
            "图片类型 (*.mp3 *.mp4 *.gif *.avi)",  # 选择类型过滤项，过滤内容在括号中
            options=options
        )
        if self.output == "":
            QtWidgets.QMessageBox.warning(self, '提示', '请先选择视频保存的位置')
        else:
            self.out.release()
            try:
                dialog = QFileDialog(self, "Save video", self.OutputDirs)
                dialog.resize(800, 600)
                dialog.close()
                shutil.copy(str(ROOT) + '/prediction.mp4', self.OutputDirs)
                QtWidgets.QMessageBox.warning(self, '提示', '导出成功!')
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, '提示', '请先完成识别工作')

    def ValueChange(self):
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.con_number.setValue(self.numcon)
        self.iou_number.setValue(self.numiou)

    def Value_change(self):
        num_conf = self.con_number.value()
        num_ious = self.iou_number.value()
        self.con_slider.setValue(int(num_conf * 100))
        self.iou_slider.setValue(int(num_ious * 100))
        self.numcon = num_conf
        self.numiou = num_ious

    def value_change_comboBox(self):
        self.lineEdit_xmin.clear()
        self.lineEdit_ymin.clear()
        self.lineEdit_xmax.clear()
        self.lineEdit_ymax.clear()
        object = self.comboBox.currentText()
        if object:
            object_number_str = object[-1]
            object_number_int = int(object_number_str)
            object_number_index = object_number_int - 1
            if self.results:
                self.lineEdit_xmin.setText(str(int(self.results[object_number_index][0])))
                self.lineEdit_ymin.setText(str(int(self.results[object_number_index][1])))
                self.lineEdit_xmax.setText(str(int(self.results[object_number_index][2])))
                self.lineEdit_ymax.setText(str(int(self.results[object_number_index][3])))

    def pred(self):
        pass

    def open_camera(self):
        self.lineEdit.setText("打开摄像头中...")
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            self.lineEdit.setText("成功打开摄像头！")
            self.timer_c.start(30)

    def detect_camera(self):
        ret, frame = self.camera.read()
        if ret:
            result_input = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            self.QtImg_input = QtGui.QImage(
                result_input.data, result_input.shape[1], result_input.shape[0], QtGui.QImage.Format_RGB32)
            self.input.setPixmap(QtGui.QPixmap.fromImage(self.QtImg_input))
            self.input.setScaledContents(True)

            if self.running:
                name_list = []
                self.comboBox.clear()
                with torch.no_grad():
                    t1 = time_sync()
                    frame_showing = frame
                    frame = letterbox(frame, new_shape=self.imgsz)[0]
                    frame = frame[:, :, ::-1].transpose(2, 0, 1)
                    frame = np.ascontiguousarray(frame)
                    frame = torch.from_numpy(frame).to(self.device)
                    frame = frame.float()  # uint8 to fp16/32
                    frame /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if frame.ndimension() == 3:
                        frame = frame.unsqueeze(0)
                    # Inference
                    self.preds = self.model(frame)[0]
                    # Apply NMS
                    self.preds = non_max_suppression(self.preds, self.numcon, self.numiou, max_det=1000)
                    t2 = time_sync()
                    self.lineEdit_detect_time.setText(str(round(t2 - t1, 2)))
                    self.lineEdit_detect_object_nums.setText(str(self.preds[0].shape[0]))
                    self.results = self.preds[0].tolist()
                    if self.preds[0].shape[0]:
                        for i in range(self.preds[0].shape[0]):
                            self.comboBox.addItem('目标' + str(i+1))

                # Process detections
                for i, det in enumerate(self.preds):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            frame.shape[2:], det[:, :4], frame_showing.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, frame_showing, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

                result = cv2.cvtColor(frame_showing, cv2.COLOR_BGR2BGRA)
                self.QtImg = QtGui.QImage(
                    result.data, result.shape[1], result.shape[0], QtGui.QImage.Format_RGB32)
                self.output.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.output.setScaledContents(True)  # 自适应界面大小

        else:
            self.timer_c.stop()
            self.camera.release()
            self.camera = None

    def close_camera(self):
        self.running = False
        self.camera = None
        self.timer_c.stop()
        self.QtImg_input = None
        self.input.setPixmap(QtGui.QPixmap())
        self.QtImg = None
        self.output.setPixmap(QtGui.QPixmap())
        self.lineEdit.setText("已关闭摄像头！")

    def detect_camera_running(self):
        self.running = True

    def bind_slots(self):
        self.buttton_image_select.clicked.connect(self.open_image)
        self.buttton_video_select.clicked.connect(self.open_video)
        self.button_weight_select.clicked.connect(self.load_model)
        self.button_weight_init.clicked.connect(self.init_model)
        self.button_image_detect.clicked.connect(self.detect_begin)
        self.button_image_show.clicked.connect(self.detect_show)
        self.button_video_detect.clicked.connect(self.detect_video)
        self.button_video_suspend.clicked.connect(self.suspend_video)
        self.button_video_stop.clicked.connect(self.stop_video)
        self.button_image_stop.clicked.connect(self.stop_image)
        self.button_image_export.clicked.connect(self.export_images)
        self.button_video_export.clicked.connect(self.export_videos)
        self.con_slider.valueChanged.connect(self.ValueChange)
        self.iou_slider.valueChanged.connect(self.ValueChange)
        self.con_number.valueChanged.connect(self.Value_change)
        self.iou_number.valueChanged.connect(self.Value_change)
        self.comboBox.currentTextChanged.connect(self.value_change_comboBox)
        self.timer.timeout.connect(self.detect_video)
        self.button_camera_start.clicked.connect(self.open_camera)
        self.button_camera_stop.clicked.connect(self.close_camera)
        self.button_camera_detect.clicked.connect(self.detect_camera_running)

    def init_icons(self):
        self.label_weight_select.setPixmap(QPixmap(':/image/icons/weight.png'))
        self.label_weight_select.setScaledContents(True)
        self.label_weight_init.setPixmap(QPixmap(':/image/icons/init.png'))
        self.label_weight_init.setScaledContents(True)
        self.label_image_select.setPixmap(QPixmap(':/image/icons/image.png'))
        self.label_image_select.setScaledContents(True)
        self.label_video_select.setPixmap(QPixmap(':/image/icons/video.png'))
        self.label_video_select.setScaledContents(True)
        self.label_image_detect.setPixmap(QPixmap(':/image/icons/recognition.png'))
        self.label_image_detect.setScaledContents(True)
        self.label_video_detect.setPixmap(QPixmap(':/image/icons/detect_video.png'))
        self.label_video_detect.setScaledContents(True)
        self.label_image_show.setPixmap(QPixmap(':/image/icons/image_result.png'))
        self.label_image_show.setScaledContents(True)
        self.label_video_suspend.setPixmap(QPixmap(':/image/icons/suspend_video.png'))
        self.label_video_suspend.setScaledContents(True)
        self.label_image_stop.setPixmap(QPixmap(':/image/icons/stop_image.png'))
        self.label_image_stop.setScaledContents(True)
        self.label_video_stop.setPixmap(QPixmap(':/image/icons/stop_video.png'))
        self.label_video_stop.setScaledContents(True)
        self.label_image_export.setPixmap(QPixmap(':/image/icons/export.png'))
        self.label_image_export.setScaledContents(True)
        self.label_video_export.setPixmap(QPixmap(':/image/icons/export.png'))
        self.label_video_export.setScaledContents(True)
        self.label_detect_time.setPixmap(QPixmap(':/image/icons/used_time.png'))
        self.label_detect_time.setScaledContents(True)
        self.label_detect_object_nums.setPixmap(QPixmap(':/image/icons/object_nums.png'))
        self.label_detect_object_nums.setScaledContents(True)
        self.label_detect_object_pos.setPixmap(QPixmap(':/image/icons/position.png'))
        self.label_detect_object_pos.setScaledContents(True)
        self.label_detect_object_all.setPixmap(QPixmap(':/image/icons/All_nums.png'))
        self.label_detect_object_all.setScaledContents(True)
        self.label_camera_select.setPixmap(QPixmap(':/image/icons/camera_start.png'))
        self.label_camera_select.setScaledContents(True)
        self.label_camera_detect.setPixmap(QPixmap(':/image/icons/camera_detect.png'))
        self.label_camera_detect.setScaledContents(True)
        self.label_camera_stop.setPixmap(QPixmap(':/image/icons/camera_stop.png'))
        self.label_camera_stop.setScaledContents(True)
        self.input.setPixmap(QPixmap("input.png"))
        self.input.setScaledContents(True)
        self.output.setPixmap(QPixmap("input.png"))
        self.output.setScaledContents(True)
        self.label_main.setPixmap(QPixmap("bg.png"))
        self.label_main.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('高精度检测识别系统')
    window.setFixedSize(window.size())
    window.show()
    app.exec()

# YOLOv5 馃殌 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        print(pred)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'trained/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/names.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_save)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == 'Darwin'  # macOS environment


def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


@try_export
def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLOv5 TorchScript model export
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    f = file.with_suffix('.torchscript')

    ts = torch.jit.trace(model, im, strict=False)
    d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLOv5 ONNX export
    # check_requirements('onnx')
    import onnx

    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')
    # output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output']
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
            import onnxsim

            LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    return f, model_onnx


@try_export
def export_openvino(file, metadata, half, prefix=colorstr('OpenVINO:')):
    # YOLOv5 OpenVINO export
    check_requirements('openvino-dev')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.inference_engine as ie

    LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')

    cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
    subprocess.run(cmd.split(), check=True, env=os.environ)  # export
    yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_paddle(model, im, file, metadata, prefix=colorstr('PaddlePaddle:')):
    # YOLOv5 Paddle export
    check_requirements(('paddlepaddle', 'x2paddle'))
    import x2paddle
    from x2paddle.convert import pytorch2paddle

    LOGGER.info(f'\n{prefix} starting export with X2Paddle {x2paddle.__version__}...')
    f = str(file).replace('.pt', f'_paddle_model{os.sep}')

    pytorch2paddle(module=model, save_dir=f, jit_type='trace', input_examples=[im])  # export
    yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
    return f, None


@try_export
def export_coreml(model, im, file, int8, half, prefix=colorstr('CoreML:')):
    # YOLOv5 CoreML export
    check_requirements('coremltools')
    import coremltools as ct

    LOGGER.info(f'\n{prefix} starting export with coremltools {ct.__version__}...')
    f = file.with_suffix('.mlmodel')

    ts = torch.jit.trace(model, im, strict=False)  # TorchScript model
    ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=im.shape, scale=1 / 255, bias=[0, 0, 0])])
    bits, mode = (8, 'kmeans_lut') if int8 else (16, 'linear') if half else (32, None)
    if bits < 32:
        if MACOS:  # quantization only supported on macOS
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        else:
            print(f'{prefix} quantization only supported on macOS, skipping...')
    ct_model.save(f)
    return f, ct_model


@try_export
def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == 'Linux':
            check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
        import tensorrt as trt

    if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, False, dynamic, simplify)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, False, dynamic, simplify)  # opset 12
    onnx = file.with_suffix('.onnx')

    LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix}WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    LOGGER.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return f, None


@try_export
def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix=colorstr('TensorFlow SavedModel:')):
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from models.tf import TFModel

    LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format='tf')
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(tfm,
                            f,
                            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False) if check_version(
                                tf.__version__, '2.6') else tf.saved_model.SaveOptions())
    return f, keras_model


@try_export
def export_pb(keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLOv5 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = file.with_suffix('.pb')

    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
    return f, None


@try_export
def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    import tensorflow as tf

    LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace('.pt', '-fp16.tflite')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from models.tf import representative_dataset_gen
        dataset = LoadImages(check_dataset(check_yaml(data))['train'], img_size=imgsz, auto=False)
        converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        f = str(file).replace('.pt', '-int8.tflite')
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    return f, None


@try_export
def export_edgetpu(file, prefix=colorstr('Edge TPU:')):
    # YOLOv5 Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    cmd = 'edgetpu_compiler --version'
    help_url = 'https://coral.ai/docs/edgetpu/compiler/'
    assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
    if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
        LOGGER.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
        sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
        for c in (
                'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
            subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    LOGGER.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
    f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
    f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model

    cmd = f"edgetpu_compiler -s -d -k 10 --out_dir {file.parent} {f_tfl}"
    subprocess.run(cmd.split(), check=True)
    return f, None


@try_export
def export_tfjs(file, prefix=colorstr('TensorFlow.js:')):
    # YOLOv5 TensorFlow.js export
    check_requirements('tensorflowjs')
    import re

    import tensorflowjs as tfjs

    LOGGER.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
    f = str(file).replace('.pt', '_web_model')  # js dir
    f_pb = file.with_suffix('.pb')  # *.pb path
    f_json = f'{f}/model.json'  # *.json path

    cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
          f'--output_node_names=Identity,Identity_1,Identity_2,Identity_3 {f_pb} {f}'
    subprocess.run(cmd.split())

    json = Path(f_json).read_text()
    with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
        subst = re.sub(
            r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}}}', r'{"outputs": {"Identity": {"name": "Identity"}, '
                                                           r'"Identity_1": {"name": "Identity_1"}, '
                                                           r'"Identity_2": {"name": "Identity_2"}, '
                                                           r'"Identity_3": {"name": "Identity_3"}}}', json)
        j.write(subst)
    return f, None


@smart_inference_mode()
def run(
        data=ROOT / 'data/coco128.yaml',  # 'dataset.yaml path'
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('torchscript', 'onnx'),  # include formats
        half=False,  # FP16 half-precision export
        inplace=False,  # set YOLOv5 Detect() inplace=True
        keras=False,  # use Keras
        optimize=False,  # TorchScript: optimize for mobile
        int8=False,  # CoreML/TF INT8 quantization
        dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
        simplify=False,  # ONNX: simplify model
        opset=12,  # ONNX: opset version
        verbose=False,  # TensorRT: verbose log
        workspace=4,  # TensorRT: workspace size (GB)
        nms=False,  # TF: add NMS to model
        agnostic_nms=False,  # TF: add agnostic NMS to model
        topk_per_class=100,  # TF.js NMS: topk per class to keep
        topk_all=100,  # TF.js NMS: topk for all classes to keep
        iou_thres=0.45,  # TF.js NMS: IoU threshold
        conf_thres=0.25,  # TF.js NMS: confidence threshold
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    # imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    # imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    if xml:  # OpenVINO
        f[3], _ = export_openvino(file, metadata, half)
    if coreml:  # CoreML
        f[4], _ = export_coreml(model, im, file, int8, half)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
        assert not tflite or not tfjs, 'TFLite and TF.js models must be exported separately, please pass only one type.'
        assert not isinstance(model, ClassificationModel), 'ClassificationModel export to TF formats not yet supported.'
        f[5], s_model = export_saved_model(model.cpu(),
                                           im,
                                           file,
                                           dynamic,
                                           tf_nms=nms or agnostic_nms or tfjs,
                                           agnostic_nms=agnostic_nms or tfjs,
                                           topk_per_class=topk_per_class,
                                           topk_all=topk_all,
                                           iou_thres=iou_thres,
                                           conf_thres=conf_thres,
                                           keras=keras)
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6], _ = export_pb(s_model, file)
        if tflite or edgetpu:
            f[7], _ = export_tflite(s_model, im, file, int8 or edgetpu, data=data, nms=nms, agnostic_nms=agnostic_nms)
        if edgetpu:
            f[8], _ = export_edgetpu(file)
        if tfjs:
            f[9], _ = export_tfjs(file)
    if paddle:  # PaddlePaddle
        f[10], _ = export_paddle(model, im, file, metadata)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = '--half' if half else ''  # --half FP16 inference arg
        LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                    f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                    f"\nDetect:          python detect.py --weights {f[-1]} {h}"
                    f"\nValidate:        python val.py --weights {f[-1]} {h}"
                    f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
                    f"\nVisualize:       https://netron.app")
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire-smoke.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/dataset_0801_neg_07_0329_finetune100/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720, 1280], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--keras', action='store_true', help='TF: use Keras')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=12, help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument(
        '--include',
        nargs='+',
        default=['onnx'],
        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)