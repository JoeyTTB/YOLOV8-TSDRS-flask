import random
import torch
import numpy as np
import time
import cv2
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
# 配置flask路由
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
# io
from io import BytesIO
# subprocess
import subprocess
# 配置OSS
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
# 显示图片
from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt
from qt_material import apply_stylesheet
from main_window_camera import Ui_MainWindow

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

# 已有功能：
# 权重加载，权重初始化
# 图像导入，图像检测，图像结果展示，图像导出，图像检测退出
# 视频导入，视频检测，视频结果展示，视频导出，视频检测退出
# 摄像头导入，摄像头检测，摄像头结果展示，摄像头导出，摄像头检测退出
# 检测时间，检测目标数，检测目标类别，检测目标位置信息


# 初始化flask路由
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')  # 初始化websocket
clients = {}  # socket连接池

# 导入进程池检测器
from process_pool_detector import ProcessPoolDetector, TaskType
import torch.multiprocessing as mp
import threading

# 全局进程池（延迟初始化）
detector_pool = None
pool_lock = threading.Lock()


def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        # self.input_width = self.input.width()
        # self.input_height = self.input.height()
        # self.output_width = self.output.width()
        # self.output_height = self.output.height()
        self.imgsz = 640
        # self.timer = QTimer()
        # self.timer.setInterval(1)
        # self.timer_c = QTimer(self)
        # self.timer_c.timeout.connect(self.detect_camera)
        self.video = None
        self.out = None
        self.device = "cuda:0"
        self.num_stop = 1
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.results = []
        self.camera = None
        self.running = False
        self.file_path = None
        # self.bind_slots()
        # self.init_icons()
        self.load_model()

    def upload_file(self, filename, local_path):
        endpoint = "https://oss-cn-beijing.aliyuncs.com"
        bucket_path = "http://web-empsys.oss-cn-beijing.aliyuncs.com/" + filename
        bucket_name = "web-empsys"
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        try:
            bucket = oss2.Bucket(auth, endpoint, bucket_name)
            bucket.put_object_from_file(filename, local_path)
            print("文件上传成功\n")
            return bucket_path
        except Exception as e:
            print("文件上传失败: {}\n ", e)

    def open_image(self, file_path):
        self.file_path = file_path
        print("window类的file_path: " + self.file_path + "\n")

    def open_video(self, file_path):
        self.video_path = file_path
        self.video = cv2.VideoCapture(self.video_path)
        if self.video.isOpened():
            self.out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),
                                       20, (int(self.video.get(3)), int(self.video.get(4))))
            print("视频成功打开\n")
        else:
            print("视频打开失败\n")

    def load_model(self):
        self.openfile_name_model = "./trained/weights/best.pt"
        if not self.openfile_name_model:
            print("权重打开失败", end="\n")
        else:
            print("权重打开成功", end="\n")
            self.init_model()

    def init_model(self):
        print("处理前路径", end=self.openfile_name_model + "\n")
        self.weights_path = str(self.openfile_name_model)
        print("处理后路径", end=self.weights_path + "\n")
        self.device = select_device(self.device)
        # self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True
        self.model = DetectMultiBackend(self.weights_path, device=self.device)  # load FP32 model
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]
        print("模型初始化成功", end="\n")

    def detect_begin(self):
        print("路径: " + self.file_path)
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
            print("图片检测完毕\n")

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
    
    # 视频检测
    def detect_video(self):
        name_list = []
        frames = []
        while True:
            ret, frame = self.video.read()
            if ret:
                print("检测一帧视频...\n")
                with torch.no_grad():
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
            else:
                break
        self.out.release()
        self.video.release()
        print("视频检测成功\n")

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

    def export_images(self, output_filename):
        self.OutputDir = os.path.join("../detected_storage/", output_filename)
        try:
            cv2.imwrite(self.OutputDir, self.img_showimg)
            print("图片导出成功\n")
        except Exception as e:
            print("图片导出失败: ", e)

    def export_videos(self, output_filename):
        self.OutputDirs = os.path.join("../detected_storage/" + output_filename)
        try:
            shutil.copy(str(ROOT) + '/prediction.mp4', self.OutputDirs)
            os.remove(str(ROOT) + '/prediction.mp4')
            print("本地视频文件保存成功")
        except Exception as e:
            print("保存本地视频文件失败")
            print(e)

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

    def start_encoder_process(self, output_format='webm', codec='libvpx-vp9'):
        # 启动 ffmpeg 子进程，设置参数以接受来自 stdin 的 yuv420p 原始视频流，并输出 WebM 格式的字节流
        command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件，如果存在
            '-f', 'rawvideo',  # 输入格式
            '-vcodec', 'rawvideo',  # 输出原始视频(格式)
            '-s', f'{300}x{300}',  # 设置分辨率
            '-pix_fmt', 'bgr24',  # 设置像素格式，因为 OpenCV 使用 BGR
            '-r', '30',  # 设置帧率
            '-i', 'pipe:0',  # 输入来自 stdin
            '-c:v', codec,  # 视频编解码器
            # '-pix_fmt', 'yuv420p',  # 输出像素格式
            '-f', output_format,  # 输出格式
            'pipe:1'  # 输出到 stdout
        ]
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def start_decoder_process(self):
        # 启动FFmpeg解码进程：输入WebM字节流，输出原始帧数据
        cmd = [
            'ffmpeg',
            '-i', 'pipe:0',          # 从stdin读取输入
            '-f', 'image2pipe',      # 输出为连续帧
            '-pix_fmt', 'bgr24',     # OpenCV兼容的像素格式
            '-vcodec', 'rawvideo',   # 输出原始视频
            'pipe:1'                 # 输出到stdout
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def detect_camera_frame(self, data):
        # 处理单帧JPEG图片
        try:
            import time
            t_start = time.time()
            
            # 将JPEG字节数据转换为numpy数组
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return b''
            
            t_decode = time.time()
            
            # 进行检测
            with torch.no_grad():
                # 直接在原图上绘制，避免copy
                # Letterbox
                img = letterbox(frame, new_shape=self.imgsz)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                t_preprocess = time.time()
                
                # Inference
                pred = self.model(img)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.numcon, self.numiou, max_det=1000)
                
                t_inference = time.time()
                
                # Process detections
                detection_count = 0
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], frame.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            detection_count += 1
                            plot_one_box(xyxy, frame, label=label,
                                       color=self.colors[int(cls)], line_thickness=1)  # 线条更细
                
                t_draw = time.time()
            
            # 将处理后的帧编码为JPEG（降低质量以加快速度）
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result = buffer.tobytes()
            
            t_encode = time.time()
            
            # 每10帧打印一次性能统计
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            
            if self._frame_count % 10 == 0:
                total_time = t_encode - t_start
                print(f"帧处理时间: 总计{total_time*1000:.1f}ms | "
                      f"解码{(t_decode-t_start)*1000:.1f}ms | "
                      f"预处理{(t_preprocess-t_decode)*1000:.1f}ms | "
                      f"推理{(t_inference-t_preprocess)*1000:.1f}ms | "
                      f"绘制{(t_draw-t_inference)*1000:.1f}ms | "
                      f"编码{(t_encode-t_draw)*1000:.1f}ms | "
                      f"检测到{detection_count}个目标")
            
            return result
            
        except Exception as e:
            print(f"帧检测错误: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    def detect_camera(self, data):
        # 将data转换为可识别检测字节流
        # 进行检测识别工作,将输出的数据转换成webm格式的字节流
        # 返回字节流
        try:
            import time
            import uuid
            
            # 使用唯一文件名避免冲突
            temp_input = f'temp_input_{uuid.uuid4().hex[:8]}.webm'
            temp_output = f'temp_output_{uuid.uuid4().hex[:8]}.webm'
            
            print(f"接收数据大小: {len(data)} bytes")
            
            # 将接收到的Blob数据写入临时文件
            with open(temp_input, 'wb') as f:
                f.write(data)
            
            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(temp_input)
            if not cap.isOpened():
                print("无法打开视频流")
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                return b''
            
            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"视频属性: {width}x{height} @ {fps}fps")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            frame_count = 0
            detection_results = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                print(f"检测第 {frame_count} 帧...")
                
                with torch.no_grad():
                    frame_showing = frame.copy()
                    # Letterbox
                    img = letterbox(frame, new_shape=self.imgsz)[0]
                    # Convert
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.device)
                    img = img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    
                    # Inference
                    pred = self.model(img)[0]
                    # Apply NMS
                    pred = non_max_suppression(pred, self.numcon, self.numiou, max_det=1000)
                    
                    # Process detections
                    for i, det in enumerate(pred):
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], frame_showing.shape).round()

                            for *xyxy, conf, cls in reversed(det):
                                label = '%s %.2f' % (self.names[int(cls)], conf)
                                detection_results.append({
                                    'class': self.names[int(cls)],
                                    'confidence': float(conf),
                                    'bbox': [int(x) for x in xyxy]
                                })
                                plot_one_box(xyxy, frame_showing, label=label,
                                           color=self.colors[int(cls)], line_thickness=2)
                    
                    # 写入处理后的帧
                    out.write(frame_showing)
            
            cap.release()
            out.release()
            
            print(f"处理了 {frame_count} 帧，检测到 {len(detection_results)} 个目标")
            
            # 读取输出文件
            result = b''
            if os.path.exists(temp_output):
                with open(temp_output, 'rb') as f:
                    result = f.read()
                print(f"输出文件大小: {len(result)} bytes")
            else:
                print("输出文件不存在")
            
            # 清理临时文件
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            return result
            
        except Exception as e:
            print(f"摄像头检测错误: {e}")
            import traceback
            traceback.print_exc()
            return b''

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
        self.label_environment_detect.setPixmap(QPixmap(':/image/icons/weight.png'))
        self.label_environment_detect.setScaledContents(True)
        self.label_environment_denoise.setPixmap(QPixmap(':/image/icons/init.png'))
        self.label_environment_denoise.setScaledContents(True)
        self.label_image_select.setPixmap(QPixmap('./icon/选择图像.png'))
        self.label_image_select.setScaledContents(True)
        self.label_video_select.setPixmap(QPixmap('./icon/选择视频.png'))
        self.label_video_select.setScaledContents(True)
        self.label_image_detect.setPixmap(QPixmap('./icon/图像检测.png'))
        self.label_image_detect.setScaledContents(True)
        self.label_video_detect.setPixmap(QPixmap('./icon/视频检测.png'))
        self.label_video_detect.setScaledContents(True)
        self.label_image_show.setPixmap(QPixmap('./icon/结果展示.png'))
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
        self.input.setPixmap(QPixmap("input.jpg"))
        self.input.setScaledContents(True)
        self.output.setPixmap(QPixmap("input.jpg"))
        self.output.setScaledContents(True)
        self.label_main.setPixmap(QPixmap("bg.jpg"))
        self.label_main.setScaledContents(True)


class ServApi:
    @app.route("/detect_image/<string:uuid>", methods=['POST'])
    def detect_img(uuid):
        """图像识别 - 使用进程池"""
        global detector_pool
        
        file_name = "img" + uuid + ".jpg"
        output_filename = "det_img" + uuid + ".jpg"
        print(f"\n{'='*60}")
        print(f"图像识别请求: {file_name}")
        print(f"{'='*60}")
        
        try:
            file = request.files[file_name]
        except Exception as e:
            print(f"文件接收失败: {e}")
            return jsonify({'code': 0, 'msg': '文件接收失败', 'data': None})
        
        if not file:
            return jsonify({'code': 0, 'msg': '图片保存失败', 'data': None})
        
        file_path = os.path.join("../storage/", file_name)
        dec_save_path = os.path.join("../detected_storage/", output_filename)
        
        try:
            # 保存上传的文件
            file.save(file_path)
            print(f"文件已保存: {file_path}")
            
            # 读取图像并编码为JPEG
            img = cv2.imread(file_path)
            if img is None:
                raise Exception("无法读取图像文件")
            
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_data = buffer.tobytes()
            
            # 提交到进程池并等待结果
            print(f"提交到进程池处理...")
            start_time = time.time()
            
            result = detector_pool.submit_task_blocking(
                client_id=uuid,
                image_data=image_data,
                task_type=TaskType.IMAGE,
                timeout=30
            )
            
            if result is None:
                raise Exception("处理超时或队列已满")
            
            task_id, processed_data, process_time = result
            total_time = time.time() - start_time
            
            print(f"处理完成: 推理耗时{process_time*1000:.1f}ms, 总耗时{total_time:.2f}s")
            
            # 保存处理后的图像
            nparr = np.frombuffer(processed_data, np.uint8)
            result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imwrite(dec_save_path, result_img)
            
            # 上传到云存储
            bucket_path = window.upload_file(output_filename, dec_save_path)
            
            # 清理临时文件
            os.remove(file_path)
            os.remove(dec_save_path)
            print("临时文件已删除")
            print(f"{'='*60}\n")
            
            return jsonify({
                'code': 1,
                'msg': '图片检测成功',
                'data': bucket_path
            })
            
        except Exception as e:
            print(f"图像识别错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理文件
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(dec_save_path):
                os.remove(dec_save_path)
            
            return jsonify({
                'code': 0,
                'msg': f'图片检测失败: {str(e)}',
                'data': None
            })

    @app.route("/detect_video/<string:uuid>", methods=['POST'])
    def detect_video(uuid):
        """视频识别 - 使用进程池逐帧处理"""
        global detector_pool
        
        file_name = "video" + uuid + ".mp4"
        output_filename = "det_video" + uuid + ".mp4"
        print(f"\n{'='*60}")
        print(f"视频识别请求: {file_name}")
        print(f"{'='*60}")
        
        try:
            file = request.files[file_name]
        except Exception as e:
            print(f"文件接收失败: {e}")
            return jsonify({'code': 0, 'msg': '文件接收失败', 'data': None})
        
        if not file:
            return jsonify({'code': 0, 'msg': '视频保存失败', 'data': None})
        
        file_path = os.path.join("../storage/", file_name)
        dec_save_path = os.path.join("../detected_storage/", output_filename)
        
        try:
            # 保存上传的文件
            file.save(file_path)
            print(f"文件已保存: {file_path}")
            
            # 打开视频
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: {width}x{height}, {fps}fps, 共{total_frames}帧")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dec_save_path, fourcc, fps, (width, height))
            
            # 逐帧处理
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 编码帧为JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_data = buffer.tobytes()
                
                # 提交到进程池处理
                result = detector_pool.submit_task_blocking(
                    client_id=f"{uuid}_frame{frame_count}",
                    image_data=image_data,
                    task_type=TaskType.VIDEO_FRAME,
                    timeout=30
                )
                
                if result is None:
                    print(f"警告: 第{frame_count}帧处理失败，使用原始帧")
                    processed_frame = frame
                else:
                    task_id, processed_data, process_time = result
                    nparr = np.frombuffer(processed_data, np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # 写入处理后的帧
                out.write(processed_frame)
                
                # 每10帧打印一次进度
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_count) * (total_frames - frame_count)
                    print(f"进度: {frame_count}/{total_frames} ({progress:.1f}%), 已用时{elapsed:.1f}s, 预计剩余{eta:.1f}s")
            
            # 释放资源
            cap.release()
            out.release()
            
            total_time = time.time() - start_time
            print(f"视频处理完成: 共{frame_count}帧, 总耗时{total_time:.2f}s, 平均{total_time/frame_count:.2f}s/帧")
            
            # 上传到云存储
            bucket_path = window.upload_file(output_filename, dec_save_path)
            
            # 清理临时文件
            os.remove(file_path)
            os.remove(dec_save_path)
            print("临时文件已删除")
            print(f"{'='*60}\n")
            
            return jsonify({
                'code': 1,
                'msg': '视频检测成功',
                'data': bucket_path
            })
            
        except Exception as e:
            print(f"视频识别错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理文件
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(dec_save_path):
                os.remove(dec_save_path)
            
            return jsonify({
                'code': 0,
                'msg': f'视频检测失败: {str(e)}',
                'data': None
            })

    @app.route("/")
    @socketio.on("connect")
    def handle_connect(self):
        """客户端连接时的初始化"""
        client_id = request.sid
        print(f"Client connected: {client_id}")
        clients[client_id] = None  # 初始化客户端映射

    @socketio.on('disconnect')
    def handle_disconnect(self):
        """客户端断开连接时的清理"""
        client_id = request.sid
        print(f'Client disconnected: {client_id}')
        
        # 从客户端映射中移除
        if client_id in clients:
            del clients[client_id]
        
        # 注意：进程池中该客户端的待处理任务会自动被丢弃
        # 因为 process_results() 线程会检查客户端是否还在 clients 中
        # 如果不在，结果会被忽略
        
        print(f'Client {client_id} 资源已清理')

    @socketio.on('video_frame')
    def handle_camera_frame(data):
        # 处理接收到的单帧图片数据
        global detector_pool
        
        try:
            # 提交任务到进程池
            client_id = request.sid
            task_id = detector_pool.submit_task(client_id, data)
            
            if task_id is None:
                print(f"任务队列已满，客户端 {client_id} 的帧被丢弃")
                
        except Exception as e:
            print(f'Error submitting frame: {e}')
            import traceback
            traceback.print_exc()


def process_results():
    """后台线程：持续从进程池获取结果并发送给客户端"""
    global detector_pool
    print("结果处理线程已启动")
    
    frame_count = 0
    discarded_count = 0  # 统计被丢弃的结果数
    
    while True:
        try:
            result = detector_pool.get_result(timeout=0.1)
            
            if result is not None:
                task_id, client_id, processed_data, process_time = result
                frame_count += 1
                
                # 检查是否是阻塞等待的任务（图像/视频识别）
                with detector_pool.result_lock:
                    if task_id in detector_pool.pending_results:
                        # 这是阻塞等待的任务，填充结果
                        detector_pool.pending_results[task_id] = (task_id, processed_data, process_time)
                        continue  # 不发送WebSocket，由阻塞函数返回
                
                # 检查客户端是否还在线（摄像头识别）
                if client_id not in clients:
                    discarded_count += 1
                    # 客户端已断开，丢弃结果
                    if discarded_count % 10 == 0:
                        print(f"警告: 已丢弃 {discarded_count} 个断开客户端的结果")
                    continue
                
                # 每10帧打印一次统计
                if frame_count % 10 == 0:
                    stats = detector_pool.get_stats()
                    print(f"处理了10帧 | 最后一帧耗时: {process_time*1000:.1f}ms | "
                          f"队列: 输入{stats['input_queue_size']} 输出{stats['output_queue_size']}")
                
                # 发送结果给对应客户端（摄像头识别）
                if len(processed_data) > 0:
                    socketio.emit('processed_frame', processed_data, room=client_id)
                    
        except Exception as e:
            print(f'Error processing result: {e}')
            import traceback
            traceback.print_exc()
            import time
            time.sleep(0.1)


def init_detector_pool():
    """初始化进程池（预热）"""
    global detector_pool
    
    if detector_pool is not None:
        print("进程池已经初始化")
        return
    
    print("="*60)
    print("预热进程池...")
    print("="*60)
    
    detector_pool = ProcessPoolDetector(
        weights_path="./trained/weights/best.pt",
        num_workers=None,  # 自动计算
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45
    )
    detector_pool.start()
    
    # 启动结果处理线程
    result_thread = threading.Thread(
        target=process_results,
        daemon=True
    )
    result_thread.start()
    
    print("="*60)
    print("进程池预热完成，系统已就绪！")
    print("="*60)


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    qApp = QApplication(sys.argv)
    window = MainWindow()
    
    # 预热进程池（在服务启动前）
    print("\n正在启动服务...")
    init_detector_pool()
    print("\n服务启动中，监听端口 5000...\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n收到中断信号")
    finally:
        # 优雅关闭进程池
        if detector_pool is not None:
            print("\n正在关闭进程池...")
            detector_pool.stop()
            print("进程池已关闭")
        print("服务已停止")