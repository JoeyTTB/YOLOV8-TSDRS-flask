"""
多进程YOLO检测器
使用进程池实现并发推理，解决线程安全问题
支持摄像头帧、图像文件、视频帧的并发处理
"""

import torch
import torch.multiprocessing as mp
import cv2
import numpy as np
from queue import Empty
import time
import os
import threading
from pathlib import Path
from enum import Enum

# 导入YOLO相关模块
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.augmentations import letterbox
from utils.torch_utils import select_device


class TaskType(Enum):
    """任务类型"""
    CAMERA_FRAME = 1  # 摄像头帧（实时流）
    IMAGE = 2         # 图像文件
    VIDEO_FRAME = 3   # 视频帧


class YOLOWorker:
    """单个worker进程的检测器"""
    
    def __init__(self, weights_path, device_id, imgsz=640, conf_thres=0.25, iou_thres=0.45):
        """
        初始化worker
        
        Args:
            weights_path: 模型权重路径
            device_id: GPU设备ID (如果有多个GPU，可以分配不同ID)
            imgsz: 输入图像尺寸
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        self.weights_path = weights_path
        self.device_id = device_id
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 模型会在进程启动后加载
        self.model = None
        self.device = None
        self.names = None
        self.colors = None
    
    def init_model(self):
        """在worker进程中初始化模型（避免跨进程传递模型）"""
        print(f"Worker PID {os.getpid()} 正在加载模型到设备 {self.device_id}...")
        
        # 选择设备
        if torch.cuda.is_available():
            # 如果有多个GPU，可以分配到不同GPU
            gpu_count = torch.cuda.device_count()
            actual_device_id = self.device_id % gpu_count
            self.device = select_device(f'cuda:{actual_device_id}')
        else:
            self.device = select_device('cpu')
        
        # 加载模型
        self.model = DetectMultiBackend(self.weights_path, device=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # 生成颜色
        import random
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        print(f"Worker PID {os.getpid()} 模型加载完成")
    
    def detect_frame(self, image_data):
        """
        检测单帧图像
        
        Args:
            image_data: JPEG编码的图像字节数据
            
        Returns:
            处理后的JPEG图像字节数据
        """
        try:
            # 解码图像
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return b''
            
            # 预处理
            img = letterbox(frame, new_shape=self.imgsz)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=1000)
            
            # 绘制检测结果
            detection_count = 0
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        detection_count += 1
                        plot_one_box(xyxy, frame, label=label,
                                   color=self.colors[int(cls)], line_thickness=1)
            
            # 编码为JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buffer.tobytes()
            
        except Exception as e:
            print(f"Worker PID {os.getpid()} 检测错误: {e}")
            import traceback
            traceback.print_exc()
            return b''


def worker_process(worker_id, device_id, weights_path, input_queue, output_queue, 
                   imgsz, conf_thres, iou_thres):
    """
    Worker进程主函数
    
    Args:
        worker_id: Worker编号
        device_id: GPU设备ID
        weights_path: 模型权重路径
        input_queue: 输入任务队列
        output_queue: 输出结果队列
        imgsz: 图像尺寸
        conf_thres: 置信度阈值
        iou_thres: IOU阈值
    """
    # 创建worker实例
    worker = YOLOWorker(weights_path, device_id, imgsz, conf_thres, iou_thres)
    
    # 初始化模型（在子进程中）
    worker.init_model()
    
    print(f"Worker {worker_id} (PID {os.getpid()}) 已启动，等待任务...")
    
    # 处理任务循环
    while True:
        try:
            # 从队列获取任务
            task_id, client_id, image_data, task_type = input_queue.get(timeout=1)
            
            # 处理任务
            start_time = time.time()
            processed_data = worker.detect_frame(image_data)
            process_time = time.time() - start_time
            
            # 将结果放入输出队列
            output_queue.put((task_id, client_id, processed_data, process_time))
            
        except Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker {worker_id} 错误: {e}")
            import traceback
            traceback.print_exc()


class ProcessPoolDetector:
    """进程池检测器管理类"""
    
    def __init__(self, weights_path, num_workers=None, imgsz=640, 
                 conf_thres=0.25, iou_thres=0.45):
        """
        初始化进程池
        
        Args:
            weights_path: 模型权重路径
            num_workers: worker进程数，None则自动计算
            imgsz: 输入图像尺寸
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
        """
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 自动计算worker数量
        if num_workers is None:
            num_workers = self._calculate_optimal_workers()
        
        self.num_workers = num_workers
        
        # 创建任务队列和结果队列（增加队列大小以支持混合场景）
        # 队列大小 = worker数 * 4，在硬件允许的情况下提供更大缓冲
        queue_size = num_workers * 4
        print(f"队列大小: {queue_size}")
        self.input_queue = mp.Queue(maxsize=queue_size)
        self.output_queue = mp.Queue(maxsize=queue_size)      
        # Worker进程列表
        self.workers = []
        
        # 任务计数器
        self.task_counter = 0
        
        # 用于阻塞等待的结果字典
        self.pending_results = {}  # {task_id: result}
        self.result_lock = threading.Lock()
        
        print(f"初始化进程池，worker数量: {self.num_workers}")
    
    def _calculate_optimal_workers(self):
        """自动计算最优worker数量"""
        # CPU核心数
        cpu_count = mp.cpu_count()
        
        # GPU数量和显存
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            # 估算单个模型显存占用（根据模型大小调整）
            # YOLOv8s约1GB，YOLOv8m约2GB，YOLOv8l约3GB
            estimated_memory_per_model = 1.5  # GB，保守估计
            
            # 获取GPU总显存
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # 留20%显存余量
            available_memory = total_memory * 0.8
            
            # 根据显存计算最大worker数
            max_workers_by_memory = int(available_memory / estimated_memory_per_model)
            
            # 综合考虑
            optimal = min(
                max_workers_by_memory,  # 显存限制
                cpu_count,              # CPU限制
                8                       # 经验上限
            )
            
            print(f"GPU显存: {total_memory:.1f}GB, 可用: {available_memory:.1f}GB")
            print(f"估算单模型占用: {estimated_memory_per_model}GB")
            print(f"建议worker数: {optimal}")
            
            return max(1, optimal)
        else:
            # CPU模式，使用CPU核心数
            return min(cpu_count, 4)
    
    def start(self):
        """启动所有worker进程"""
        print("启动worker进程...")
        
        for i in range(self.num_workers):
            # 如果有多个GPU，可以分配到不同GPU
            device_id = i % (torch.cuda.device_count() if torch.cuda.is_available() else 1)
            
            p = mp.Process(
                target=worker_process,
                args=(i, device_id, self.weights_path, self.input_queue, 
                      self.output_queue, self.imgsz, self.conf_thres, self.iou_thres)
            )
            p.start()
            self.workers.append(p)
        
        print(f"已启动 {len(self.workers)} 个worker进程")
    
    def submit_task(self, client_id, image_data, task_type=TaskType.CAMERA_FRAME):
        """
        提交检测任务
        
        Args:
            client_id: 客户端ID
            image_data: JPEG图像字节数据
            task_type: 任务类型（摄像头帧/图像/视频帧）
            
        Returns:
            task_id: 任务ID，如果队列满则返回None
        """
        try:
            task_id = f"{task_type.name}_{client_id}_{int(time.time() * 1000000)}"
            self.input_queue.put_nowait((task_id, client_id, image_data, task_type))
            self.task_counter += 1
            return task_id
        except:
            # 队列满，丢弃任务
            return None
    
    def submit_task_blocking(self, client_id, image_data, task_type=TaskType.IMAGE, timeout=30):
        """
        提交任务并等待结果（用于图像/视频识别）
        
        Args:
            client_id: 客户端ID
            image_data: JPEG图像字节数据
            task_type: 任务类型
            timeout: 超时时间（秒）
            
        Returns:
            (task_id, processed_data, process_time) 或 None（超时）
        """
        task_id = self.submit_task(client_id, image_data, task_type)
        if task_id is None:
            print(f"警告: 输入队列已满，无法提交任务 {task_id}")
            return None
        
        # 注册等待的任务
        with self.result_lock:
            self.pending_results[task_id] = None
        
        # 等待结果
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.result_lock:
                if self.pending_results.get(task_id) is not None:
                    result = self.pending_results.pop(task_id)
                    return result
            time.sleep(0.01)  # 短暂休眠，避免CPU空转
        
        # 超时，清理
        with self.result_lock:
            if task_id in self.pending_results:
                del self.pending_results[task_id]
        
        print(f"警告: 任务 {task_id} 等待超时")
        return None  # 超时
    
    def get_result(self, timeout=0.1):
        """
        获取检测结果
        
        Args:
            timeout: 超时时间
            
        Returns:
            (task_id, client_id, result, process_time) 或 None
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def stop(self):
        """停止所有worker进程"""
        print("停止worker进程...")
        
        # 发送结束信号
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # 等待所有进程结束
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        print("所有worker进程已停止")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'num_workers': self.num_workers,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'total_tasks': self.task_counter
        }


if __name__ == '__main__':
    # 测试代码
    mp.set_start_method('spawn', force=True)
    
    weights_path = "./trained/weights/best.pt"
    
    # 创建进程池（自动计算worker数量）
    detector = ProcessPoolDetector(weights_path)
    detector.start()
    
    print("进程池已启动，按Ctrl+C退出")
    
    try:
        # 模拟任务提交
        import time
        while True:
            time.sleep(1)
            stats = detector.get_stats()
            print(f"统计: {stats}")
    except KeyboardInterrupt:
        print("\n收到中断信号")
    finally:
        detector.stop()
