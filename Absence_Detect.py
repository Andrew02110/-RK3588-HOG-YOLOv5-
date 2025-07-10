import torch  # PyTorch深度学习框架，用于加载并运行YOLOv5目标检测模型
import cv2  # OpenCV库，用于图像处理和摄像头操作
import time  # 用于时间相关操作
import numpy as np  # 数值计算库
from datetime import datetime, timedelta  # 日期和时间处理
from threading import Thread  # 多线程支持
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, BOTTOM, LEFT, Frame  # GUI组件
import tkinter.font as tkFont  # Tkinter字体支持
import os  # 操作系统功能接口
from PIL import Image, ImageTk  # 图像处理库，用于Tkinter中显示图像
import sqlite3  # SQLite数据库支持
import face_recognition  # 人脸识别库
import json  # JSON数据处理
from tkinter import ttk  # Tkinter高级组件
from pypinyin import lazy_pinyin  # 中文转拼音工具
import requests  # HTTP请求库
from tqdm import tqdm  # 进度条显示
import pandas as pd  # 数据处理库，用于导出CSV
import threading  # 线程管理
import queue  # 线程间通信
import signal  # 信号处理
import torch.serialization  # 添加此行，用于处理安全加载
import sys  # 添加此行，用于系统退出功能

# 全局变量声明，用于报警窗口
alert_window = None
alert_label = None

def init_database():
    """初始化数据库，创建必要的表"""
    try:
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            
            # 创建人脸数据表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE,
                name TEXT,
                position TEXT,
                face_id TEXT UNIQUE,
                face_encoding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 创建离岗记录表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS absence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                name TEXT,
                absence_duration TEXT,
                absence_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")

# 将HybridWorkerDetector类移到这里
class HybridWorkerDetector:
    """混合工人检测器，结合HOG和YOLOv5"""
    
    def __init__(self, yolo_model_path, face_library=None):
        self.yolo_model_path = yolo_model_path
        self.face_library = face_library or []
        self.yolo_model = None
        self.yolo_loaded = False
        self.yolo_loading = False
        self.detection_mode = "hog"  # 初始使用HOG检测器
        
        # 初始化HOG行人检测器
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("[INFO] HOG pedestrian detector initialized")
        
        # 启动YOLOv5模型异步加载
        self.start_yolo_loading()
    
    def start_yolo_loading(self):
        """异步加载YOLOv5模型"""
        if self.yolo_loading or self.yolo_loaded:
            return
        
        self.yolo_loading = True
        Thread(target=self._load_yolo_model, daemon=True).start()
    
    def _load_yolo_model(self):
        """在后台线程中加载YOLOv5模型"""
        try:
            print("[INFO] Starting to load YOLOv5 model...")
            # 尝试使用离线模式加载模型
            try:
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                path=self.yolo_model_path, 
                                                trust_repo=True,
                                                force_reload=False)
            except Exception as e:
                print(f"[WARNING] torch.hub loading failed: {e}, trying to load model file directly")
                if os.path.exists(self.yolo_model_path):
                    # 修复PyTorch 2.6+ 模型加载问题
                    try:
                        # 方法1: 使用添加安全全局变量的方式
                        try:
                            # 尝试添加YOLOv5模型类到安全全局列表
                            torch.serialization.add_safe_globals(['models.yolo.Model'])
                            self.yolo_model = torch.load(self.yolo_model_path)
                        except Exception:
                            # 如果添加安全全局变量失败，回退到方法2
                            print("[WARNING] Failed to add safe globals, trying with weights_only=False")
                            self.yolo_model = torch.load(self.yolo_model_path, weights_only=False)
                    
                    except Exception as load_err:
                        print(f"[ERROR] Model loading error: {load_err}")
                        print("[INFO] Trying to download the model from torch hub...")
                        # 最后尝试从在线仓库下载
                        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                                         trust_repo=True, 
                                                         force_reload=True)
                    
                    # 确保模型处于评估模式
                    if hasattr(self.yolo_model, 'model'):
                        self.yolo_model = self.yolo_model.model
                    self.yolo_model.eval()
            
            self.yolo_loaded = True
            self.detection_mode = "hybrid"  # 切换到混合模式
            print("[INFO] YOLOv5 model loaded successfully, switched to hybrid detection mode")
        except Exception as e:
            print(f"[ERROR] YOLOv5 model loading failed: {e}")
        finally:
            self.yolo_loading = False
    
    def detect_workers(self, frame):
        """检测画面中的工人
        
        Args:
            frame: 输入视频帧
            
        Returns:
            worker_detected: 是否检测到工人
            detections: 检测到的人体位置列表 [(x1,y1,x2,y2), ...]
            frame: 处理后的帧（带有标注）
        """
        # 确保输入帧有效
        if frame is None or frame.size == 0:
            return False, [], frame
        
        # 复制一份帧用于绘制
        result_frame = frame.copy()
        all_detections = []
        worker_detected = False
        confidence_threshold = 0.5  # 置信度阈值
        
        # 使用HOG检测器
        hog_detections = self._detect_with_hog(frame)
        for (x, y, w, h) in hog_detections:
            all_detections.append((x, y, x+w, y+h, 0.6, "hog"))  # HOG默认置信度0.6
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_frame, "Person(HOG)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 如果YOLOv5已加载，则融合检测结果
        if self.yolo_loaded and self.yolo_model is not None:
            try:
                # 使用YOLOv5检测
                yolo_results = self.yolo_model(frame)
                yolo_detections = yolo_results.xyxy[0].cpu().numpy()
                
                # 提取人体检测结果
                for detection in yolo_detections:
                    x1, y1, x2, y2, conf, class_id = detection
                    if conf >= confidence_threshold:
                        label = yolo_results.names[int(class_id)]
                        if (label == 'person'):
                            # 记录检测结果
                            all_detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), "yolo"))
                            # 在图像上绘制边界框
                            cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(result_frame, f"Person(YOLO): {conf:.2f}", (int(x1), int(y1)-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print(f"[ERROR] YOLOv5 detection error: {e}")
        
        # 非极大值抑制 (NMS) 融合重叠检测
        if len(all_detections) > 1:
            # 提取坐标和置信度
            boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _, _) in all_detections])
            confidences = np.array([conf for (_, _, _, _, conf, _) in all_detections])
            
            # 应用NMS
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.5, 0.4).flatten()
            
            # 保留NMS后的检测结果
            filtered_detections = [all_detections[i] for i in indices]
            all_detections = filtered_detections
        
        # 基于人脸识别增强检测
        if len(all_detections) > 0:
            worker_detected = True
            self._enhance_with_face_recognition(frame, result_frame, all_detections)
        
        # 返回检测结果
        return worker_detected, all_detections, result_frame
    
    def _detect_with_hog(self, frame):
        """使用HOG检测器检测人体"""
        # 调整图像大小以提高性能
        height, width = frame.shape[:2]
        
        # 如果图像太大，则调整大小
        if width > 640:
            scale = 640.0 / width
            frame_resized = cv2.resize(frame, (640, int(height * scale)))
        else:
            scale = 1.0
            frame_resized = frame
        
        # HOG检测
        boxes, weights = self.hog.detectMultiScale(
            frame_resized, 
            winStride=(8, 8), 
            padding=(4, 4), 
            scale=1.05
        )
        
        # 如果进行了缩放，则将结果映射回原始尺寸
        if scale != 1.0:
            boxes = np.array([[int(x/scale), int(y/scale), int(w/scale), int(h/scale)] for (x, y, w, h) in boxes])
        
        return boxes
    
    def _enhance_with_face_recognition(self, frame, result_frame, detections):
        """使用人脸识别增强检测结果"""
        if not self.face_library:
            return
        
        # 检测人脸
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # 获取人脸特征点
        face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
        
        # 识别人脸
        for face_encoding, (top, right, bottom, left), landmarks in zip(face_encodings, face_locations, face_landmarks_list):
            name = "Unknown"
            for face_data in self.face_library:
                known_encoding = np.array(face_data["encoding"])
                matches = face_recognition.compare_faces([known_encoding], face_encoding)
                if matches[0]:
                    name = face_data["name"]
                    break
            
            # 中文转拼音显示
            display_name = " ".join(lazy_pinyin(name)) if name != "Unknown" else name
            
            # 在图像上绘制人脸框和标签
            cv2.rectangle(result_frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(result_frame, display_name, (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 显示人脸特征点
            self._draw_facial_landmarks(result_frame, landmarks)

    def _draw_facial_landmarks(self, frame, landmarks, show_labels=False):
        """绘制人脸特征点和连线
        
        参数:
            frame: 要绘制的图像帧
            landmarks: 人脸特征点字典
            show_labels: 是否显示特征标签，默认False
        """
        # 绘制各个特征点
        for feature, points in landmarks.items():
            # 根据不同特征设置不同颜色
            if feature == 'chin':  # 下巴
                color = (255, 0, 0)  # 蓝色
                label = "Chin"
            elif feature == 'left_eye':  # 左眼
                color = (0, 255, 0)  # 绿色
                label = "Left Eye"
            elif feature == 'right_eye':  # 右眼
                color = (0, 255, 0)  # 绿色
                label = "Right Eye"
            elif feature == 'left_eyebrow':  # 左眉毛
                color = (255, 255, 0)  # 青色
                label = "Left Eyebrow"
            elif feature == 'right_eyebrow':  # 右眉毛
                color = (255, 255, 0)  # 青色
                label = "Right Eyebrow"
            elif feature == 'nose_bridge':  # 鼻梁
                color = (0, 0, 255)  # 红色
                label = "Nose Bridge"
            elif feature == 'nose_tip':  # 鼻尖
                color = (0, 0, 255)  # 红色
                label = "Nose Tip"
            elif feature == 'top_lip':  # 上嘴唇
                color = (255, 0, 255)  # 紫色
                label = "Top Lip"
            elif feature == 'bottom_lip':  # 下嘴唇
                color = (255, 0, 255)  # 紫色
                label = "Bottom Lip"
            else:
                color = (255, 255, 255)  # 白色
                label = feature
            
            # 绘制连线（将点依次连接）
            prev_point = None
            for i, point in enumerate(points):
                # 绘制特征点
                cv2.circle(frame, point, 2, color, -1)
                
                # 连接点
                if prev_point is not None:
                    cv2.line(frame, prev_point, point, color, 1)
                prev_point = point
                
            # 如果是闭环特征（如眼睛、嘴唇），则连接首尾点
            if feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip'] and len(points) > 2:
                cv2.line(frame, points[-1], points[0], color, 1)
            
            # 显示特征标签
            if show_labels and points:
                # 选择一个合适的点来放置标签
                label_point = points[0]  # 使用第一个点的位置
                if feature == 'left_eye':
                    label_point = points[3]  # 左眼外侧
                elif feature == 'right_eye':
                    label_point = points[0]  # 右眼外侧
                
                # 添加标签文本
                cv2.putText(frame, label, 
                           (label_point[0], label_point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def check_face_exists(face_encoding, similarity_threshold=0.6):
    """
    检查人脸是否已存在于数据库中
    
    参数:
        face_encoding: 待检查的人脸特征向量
        similarity_threshold: 相似度阈值，默认0.6
        
    返回:
        (bool, dict): (是否存在, 匹配的人脸数据)
    """
    try:
        # 从数据库获取所有人脸数据
        face_data_list = get_all_faces_from_db()
        
        # 对比每一个已存在的人脸
        for face_data in face_data_list:
            known_encoding = np.array(face_data["encoding"])
            # 计算欧氏距离，距离越小表示越相似
            face_distance = face_recognition.face_distance([known_encoding], face_encoding)
            
            # 如果距离小于阈值，则认为是同一个人
            if face_distance[0] < similarity_threshold:
                return True, face_data
                
        return False, None
    except Exception as e:
        print(f"Error checking if face exists: {e}")
        return False, None

def save_face_to_db(employee_id, name, position, face_encoding):
    """保存人脸特征到数据库
    
    参数:
        employee_id: 员工ID
        name: 员工姓名
        position: 职位
        face_encoding: 面部特征向量 (numpy array)
    
    返回:
        bool: 是否保存成功
    """
    try:
        face_id = f"face_{employee_id}"
        # 将numpy数组转换为JSON字符串
        encoding_json = json.dumps(face_encoding.tolist())
        
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO face_records
            (employee_id, name, position, face_id, face_encoding)
            VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, name, position, face_id, encoding_json))
            conn.commit()
            print(f"Successfully saved face data to database: {name} ({employee_id})")
            return True
    except Exception as e:
        print(f"Failed to save face data: {e}")
        return False

def get_all_faces_from_db():
    """从数据库获取所有人脸数据
    
    返回:
        list: 人脸数据列表
    """
    try:
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT employee_id, name, position, face_id, face_encoding FROM face_records')
            rows = cursor.fetchall()
            
            face_data_list = []
            for row in rows:
                employee_id, name, position, face_id, encoding_json = row
                # 将JSON字符串转回numpy数组
                try:
                    encoding = np.array(json.loads(encoding_json))
                    
                    face_data = {
                        "employee_id": employee_id,
                        "name": name,
                        "position": position,
                        "face_id": face_id,
                        "encoding": encoding
                    }
                    face_data_list.append(face_data)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse face encoding: {e}, data: {encoding_json[:30]}...")
            
            print(f"Loaded {len(face_data_list)} face records from database")
            return face_data_list
    except Exception as e:
        print(f"Failed to get face data: {e}")
        return []

def migrate_json_to_db():
    """将现有JSON文件数据迁移到数据库"""
    try:
        # 确保人脸库目录存在
        if not os.path.exists("face_library"):
            print("Face library directory doesn't exist, no migration needed")
            return
            
        # 迁移计数
        migrated_count = 0
        failed_count = 0
        
        for file in os.listdir("face_library"):
            if file.endswith(".json"):
                try:
                    with open(os.path.join("face_library", file), "r") as f:
                        face_data = json.load(f)
                        
                    # 提取数据
                    employee_id = face_data.get("employee_id")
                    name = face_data.get("name")
                    position = face_data.get("position")
                    encoding = np.array(face_data.get("encoding"))
                    
                    # 保存到数据库
                    if save_face_to_db(employee_id, name, position, encoding):
                        migrated_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"Failed to migrate file {file}: {e}")
                    failed_count += 1
        
        print(f"Migration completed: {migrated_count} successful, {failed_count} failed")
    except Exception as e:
        print(f"Error during migration process: {e}")

def alert_popup_dynamic(message):
    """
    动态更新的报警弹窗函数
    
    参数:
        message (str): 要显示的报警信息
    """
    global alert_window, alert_label

    # 检查窗口是否已经存在
    if alert_window is None or not alert_window.winfo_exists():
        # 如果不存在，创建一个新的窗口
        alert_window = Tk()
        alert_window.title("Alert")

        # 创建一个新的标签，设置字体为Arial，颜色为红色
        alert_label = Label(alert_window, text=message, padx=20, pady=20, font=("Arial", 16), fg="red")
        alert_label.pack()

    else:
        # 如果窗口已存在，则更新标签的文本
        alert_label.config(text=message)

    # 更新窗口显示内容
    alert_window.update()


def enroll_face(employee_id, name, position):
    """
    录入人脸并保存到数据库
    
    参数:
        employee_id (str): 员工ID
        name (str): 员工姓名
        position (str): 员工职位
    """
    cap = cv2.VideoCapture(11)  # 初始化摄像头
    print("Please face the camera, press 'q' to complete enrollment")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame")
            break

        cv2.imshow("Face Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                if face_encodings:
                    # 保存到数据库
                    if save_face_to_db(employee_id, name, position, face_encodings[0]):
                        print(f"Face enrollment successful: {name} ({employee_id})")
                    else:
                        print("Database save failed, please try again")
                else:
                    print("Could not extract face features, please try again")
            else:
                print("No face detected, please try again")
            break

    cap.release()
    cv2.destroyAllWindows()

def download_model(model_url, save_path, timeout=60):
    """
    下载模型文件，并显示下载进度
    
    参数:
        model_url (str): 模型下载地址
        save_path (str): 保存路径
        timeout (int): 下载超时时间，单位秒，默认 60 秒
    """
    try:
        print(f"[DEBUG] Starting model download: {model_url}")
        response = requests.get(model_url, stream=True, timeout=timeout)
        response.raise_for_status()

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 如果没有 content-length 头，直接下载并保存
        if (total_size == 0):
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"[DEBUG] Model download complete: {save_path}")
            return

        # 启动进度条
        with open(save_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=save_path.split('/')[-1]) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"[DEBUG] Model download complete: {save_path}")
        
    except requests.exceptions.Timeout:
        print("[ERROR] Download timeout, please check network connection or increase timeout")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Network request error: {e}")
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")

def record_absence(employee_id, name, absence_duration, absence_date):
    """记录离岗信息到统一数据库"""
    try:
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO absence_records 
            (employee_id, name, absence_duration, absence_date)
            VALUES (?, ?, ?, ?)
            ''', (employee_id, name, absence_duration, absence_date))
            conn.commit()
            return True
    except Exception as e:
        print(f"[ERROR] Failed to record absence information: {e}")
        return False

def monitor_worker():
    """实时监测工人是否离岗"""
    # 从数据库加载人脸库
    face_library = get_all_faces_from_db()
    if not face_library:
        print("[WARNING] Face library is empty, unable to identify worker identities")
    
    # 检查模型文件
    model_path = '/home/elf/reid_project/yolov5s.pt'  # 本地模型路径
    model_url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'  # 模型下载地址

    print(f"[DEBUG] Checking if model file exists: {os.path.exists(model_path)}")
    if not os.path.exists(model_path):
        print("[DEBUG] Local model doesn't exist, attempting to download yolov5s model...")
        download_model(model_url, model_path)
        print("[DEBUG] Model download complete")
    
    # 创建混合检测器
    detector = HybridWorkerDetector(model_path, face_library)

    # 打开摄像头
    cap = cv2.VideoCapture(11)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    last_seen_time = datetime.now()  # 记录上次检测到工人的时间
    absence_threshold_minutes = 1  # 离岗阈值（分钟）
    absence_threshold = timedelta(minutes=absence_threshold_minutes)

    # 创建 Tkinter 窗口
    root = Tk()
    root.title("Absence Monitoring - Hybrid Mode")
    root.geometry("800x600")
    font = set_english_font()

    # 信息标签
    info_label = Label(root, text="Status: Initializing...", font=font)
    info_label.pack(pady=5)
    
    mode_label = Label(root, text="Detection Mode: HOG", font=font)
    mode_label.pack(pady=5)

    video_label = Label(root)
    video_label.pack(fill="both", expand=True)

    # 用于跟踪定时器ID，便于清理
    update_id = None

    def update_frame():
        """更新视频帧并进行目标检测和人脸识别"""
        nonlocal update_id
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame")
            update_id = root.after(10, update_frame)
            return

        start_time = time.time()
        
        # 使用混合检测器检测工人
        worker_detected, detections, result_frame = detector.detect_workers(frame)
        
        # 更新检测模式显示
        mode_label.config(text=f"Detection Mode: {detector.detection_mode.upper()}")
        
        # 处理检测结果
        nonlocal last_seen_time
        if worker_detected:
            last_seen_time = datetime.now()
            info_label.config(text=f"Status: Worker Present - Detected {len(detections)} people")
        else:
            elapsed_time = datetime.now() - last_seen_time
            minutes = int(elapsed_time.total_seconds() // 60)
            seconds = int(elapsed_time.total_seconds() % 60)
            info_label.config(text=f"Status: Worker Absent - {minutes}m {seconds}s")

        # 计算离岗时间并检查是否超过阈值
        elapsed_time = datetime.now() - last_seen_time
        if (elapsed_time > absence_threshold):
            absence_date = last_seen_time.strftime("%Y-%m-%d")
            absence_duration = str(elapsed_time).split('.')[0]

            # 记录离岗信息到数据库
            record_absence("unknown", "unknown", absence_duration, absence_date)

            # 显示报警信息
            minutes = int(elapsed_time.total_seconds() // 60)
            message = f"Worker absent for over {minutes} minutes, please check!"
            print(message)
            Thread(target=alert_popup_dynamic, args=(message,)).start()

        # 计算检测性能
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 转换图像为 Tkinter 格式并显示
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(result_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # 继续更新帧
        update_id = root.after(10, update_frame)

    # 返回首页按钮
    Button(root, text="Return to Main Menu", command=lambda: [
        root.after_cancel(update_id) if update_id else None,
        cap.release(), 
        root.destroy(), 
        main_menu()
    ], font=font).pack(pady=10)

    # 开始更新帧
    update_id = root.after(10, update_frame)
    root.mainloop()

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def set_english_font():
    """
    设置适合英文显示的字体，字号为12号
    
    返回:
        tkFont.Font: 设置好的字体对象
    """
    return tkFont.Font(family="Arial", size=12)  # 默认字体

def create_face_enrollment_page():
    """创建人脸录入页面"""
    root = Tk()
    root.title("Face Enrollment")
    root.geometry("800x600")

    font = set_english_font()

    # 创建输入框和标签
    Label(root, text="Employee ID:", font=font).pack(pady=5)
    employee_id_var = StringVar()
    Entry(root, textvariable=employee_id_var, font=font).pack(pady=5)

    Label(root, text="Name:", font=font).pack(pady=5)
    name_var = StringVar()
    Entry(root, textvariable=name_var, font=font).pack(pady=5)

    Label(root, text="Position:", font=font).pack(pady=5)
    position_var = StringVar()
    Entry(root, textvariable=position_var, font=font).pack(pady=5)

    # 状态标签
    status_var = StringVar()
    status_var.set("Ready")
    status_label = Label(root, textvariable=status_var, font=font, fg="blue")
    status_label.pack(pady=5)

    # 创建视频显示区域
    video_label = Label(root)
    video_label.pack(pady=10, fill="both", expand=True)

    # 创建已录入人员表格
    columns = ("Employee ID", "Name", "Position", "Face ID")
    table = ttk.Treeview(root, columns=columns, show="headings")
    for col in columns:
        table.heading(col, text=col)
    table.pack(pady=10, fill="x")

    # 初始化摄像头
    cap = cv2.VideoCapture(11)
    
    # 用于存储当前帧中检测到的人脸特征
    current_face_encoding = None
    
    # 用于跟踪定时器ID，便于清理
    update_id = None
    
    def update_frame():
        """更新视频帧，显示人脸特征点"""
        nonlocal current_face_encoding, update_id
        
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame")
            update_id = root.after(10, update_frame)
            return

        # 检测人脸位置
        face_locations = face_recognition.face_locations(frame)
        
        # 如果检测到人脸，提取特征并显示特征点
        if face_locations:
            # 提取人脸特征
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            if face_encodings:
                current_face_encoding = face_encodings[0]
                
                # 检查人脸是否已存在
                exists, existing_face = check_face_exists(current_face_encoding)
                if exists:
                    status_var.set(f"Warning: Face already enrolled! Name: {existing_face['name']}, ID: {existing_face['employee_id']}")
                    status_label.config(fg="red")
                else:
                    status_var.set("Face detected, ready to enroll")
                    status_label.config(fg="green")
            
            # 获取人脸特征点
            face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
            
            # 在图像上绘制人脸框和特征点
            for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks_list):
                # 绘制人脸框
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # 绘制各个特征点
                for feature, points in landmarks.items():
                    # 根据不同特征设置不同颜色
                    if feature == 'chin':  # 下巴
                        color = (255, 0, 0)  # 蓝色
                    elif feature in ['left_eye', 'right_eye']:  # 眼睛
                        color = (0, 255, 0)  # 绿色
                    elif feature in ['left_eyebrow', 'right_eyebrow']:  # 眉毛
                        color = (255, 255, 0)  # 青色
                    elif feature == 'nose_bridge' or feature == 'nose_tip':  # 鼻子
                        color = (0, 0, 255)  # 红色
                    elif feature == 'top_lip' or feature == 'bottom_lip':  # 嘴唇
                        color = (255, 0, 255)  # 紫色
                    else:
                        color = (255, 255, 255)  # 白色
                    
                    # 绘制特征点
                    for point in points:
                        cv2.circle(frame, point, 1, color, -1)
                
                # 添加提示文本
                cv2.putText(frame, "Press 'Enroll' button to save face", (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            current_face_encoding = None
            status_var.set("No face detected, please face the camera")
            status_label.config(fg="blue")

        # 转换图像为Tkinter格式并显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        update_id = root.after(10, update_frame)

    def validate_inputs(*args):
        """验证输入是否有效，如果有效则启用录入按钮"""
        employee_id = employee_id_var.get()
        name = name_var.get()
        position = position_var.get()
        
        if employee_id and name and position:
            enroll_button.config(state="normal")
        else:
            enroll_button.config(state="disabled")

    def start_enrollment():
        """开始人脸录入过程"""
        nonlocal current_face_encoding
        
        if current_face_encoding is None:
            messagebox.showerror("Error", "No face detected, please face the camera and try again")
            return
            
        employee_id = employee_id_var.get()
        name = name_var.get()
        position = position_var.get()
        
        # 再次检查人脸是否已存在
        exists, existing_face = check_face_exists(current_face_encoding)
        if exists:
            messagebox.showwarning("Warning", 
                                  f"This face is already enrolled!\nName: {existing_face['name']}\nID: {existing_face['employee_id']}")
            return
        
        # 保存到数据库
        if save_face_to_db(employee_id, name, position, current_face_encoding):
            messagebox.showinfo("Success", f"Successfully enrolled face data for {name}")
            
            # 更新表格显示
            for row in table.get_children():
                table.delete(row)
            
            # 重新从数据库加载数据
            with sqlite3.connect('worker_monitoring.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT employee_id, name, position, face_id FROM face_records')
                rows = cursor.fetchall()
                
                for row in rows:
                    table.insert("", "end", values=row)
                    
            # 清空输入框
            employee_id_var.set("")
            name_var.set("")
            position_var.set("")
            status_var.set("Enrollment successful! Continue to enroll new faces")
            status_label.config(fg="green")
        else:
            messagebox.showerror("Error", "Database save failed, please try again")

    # 设置输入变更监听
    employee_id_var.trace_add("write", validate_inputs)
    name_var.trace_add("write", validate_inputs)
    position_var.trace_add("write", validate_inputs)

    # 按钮框架
    button_frame = Frame(root)
    button_frame.pack(pady=10)
    
    # 创建按钮
    enroll_button = Button(button_frame, text="Enroll", command=start_enrollment, font=font, state="disabled")
    enroll_button.pack(side=LEFT, padx=10)

    Button(button_frame, text="Return to Main Menu", command=lambda: [
        root.after_cancel(update_id) if update_id else None,
        cap.release(), 
        root.destroy(), 
        main_menu()
    ], font=font).pack(side=LEFT, padx=10)

    # 加载现有人脸数据
    try:
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT employee_id, name, position, face_id FROM face_records')
            rows = cursor.fetchall()
            
            for row in rows:
                table.insert("", "end", values=row)
    except Exception as e:
        print(f"Failed to load existing face data: {e}")

    # 开始更新帧
    update_id = root.after(10, update_frame)
    root.mainloop()

def create_monitoring_page():
    """创建离岗检测页面"""
    monitor_worker()

def create_data_management_page():
    """
    创建数据管理页面，提供数据库操作功能
    """
    root = Tk()
    root.title("Data Management")
    root.geometry("800x600")
    font = set_english_font()
    
    Label(root, text="Worker Data Management", font=tkFont.Font(family="Arial", size=16)).pack(pady=10)
    
    # 创建选项卡
    tab_control = ttk.Notebook(root)
    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab3 = ttk.Frame(tab_control)
    
    tab_control.add(tab1, text='Face Data')
    tab_control.add(tab2, text='Absence Records')
    tab_control.add(tab3, text='Import/Export')
    tab_control.pack(expand=1, fill="both")
    
    # 人脸数据选项卡
    def load_face_data():
        # 清空表格
        for row in face_table.get_children():
            face_table.delete(row)
            
        # 从数据库加载数据
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT employee_id, name, position, face_id FROM face_records')
            rows = cursor.fetchall()
            
            for row in rows:
                face_table.insert("", "end", values=row)
    
    def delete_face_record():
        selected = face_table.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a record to delete")
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to delete the selected record(s)?"):
            for item in selected:
                employee_id = face_table.item(item, "values")[0]
                
                # 从数据库删除
                with sqlite3.connect('worker_monitoring.db') as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM face_records WHERE employee_id = ?', (employee_id,))
                    conn.commit()
                
                # 从表格删除
                face_table.delete(item)
    
    # 创建人脸数据表格
    columns = ("Employee ID", "Name", "Position", "Face ID")
    face_table = ttk.Treeview(tab1, columns=columns, show="headings")
    for col in columns:
        face_table.heading(col, text=col)
    face_table.pack(pady=10, fill="both", expand=True)
    
    # 添加按钮
    button_frame = Frame(tab1)
    button_frame.pack(pady=10)
    
    Button(button_frame, text="Refresh", command=load_face_data, font=font).pack(side=LEFT, padx=5)
    Button(button_frame, text="Delete", command=delete_face_record, font=font).pack(side=LEFT, padx=5)
    
    # 离岗记录选项卡
    def load_absence_data():
        # 清空表格
        for row in absence_table.get_children():
            absence_table.delete(row)
            
        # 从数据库加载数据
        with sqlite3.connect('worker_monitoring.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT employee_id, name, absence_date, absence_duration FROM absence_records ORDER BY absence_date DESC')
            rows = cursor.fetchall()
            
            for row in rows:
                absence_table.insert("", "end", values=row)
    
    # 创建离岗记录表格
    columns = ("Employee ID", "Name", "Absence Date", "Absence Duration")
    absence_table = ttk.Treeview(tab2, columns=columns, show="headings")
    for col in columns:
        absence_table.heading(col, text=col)
    absence_table.pack(pady=10, fill="both", expand=True)
    
    # 添加按钮
    button_frame2 = Frame(tab2)
    button_frame2.pack(pady=10)
    
    Button(button_frame2, text="Refresh", command=load_absence_data, font=font).pack(side=LEFT, padx=5)
    Button(button_frame2, text="Export CSV", command=lambda: export_to_csv("absence_records"), font=font).pack(side=LEFT, padx=5)
    
    # 数据导入/导出选项卡
    Label(tab3, text="Data Migration Tools", font=font).pack(pady=10)
    
    Button(tab3, text="Import JSON Files to Database", command=migrate_json_to_db, font=font).pack(pady=10)
    Button(tab3, text="Export Face Data to JSON", command=lambda: export_to_json("face_records"), font=font).pack(pady=10)
    Button(tab3, text="Backup Entire Database", command=backup_database, font=font).pack(pady=10)
    
    # 加载初始数据
    load_face_data()
    load_absence_data()
    
    # 返回按钮
    Button(root, text="Return to Main Menu", command=lambda: [root.destroy(), main_menu()], font=font).pack(pady=10)
    
    root.mainloop()

def export_to_csv(table_name):
    """导出表数据到CSV文件"""
    try:
        with sqlite3.connect('worker_monitoring.db') as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        filename = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        messagebox.showinfo("Success", f"Data exported to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error: {e}")

def export_to_json(table_name):
    """导出表数据到JSON文件"""
    try:
        filename = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if (table_name == "face_records"):
            # 导出人脸数据
            face_data_list = get_all_faces_from_db()
            
            # 将numpy数组转换为列表以便序列化
            for face_data in face_data_list:
                face_data["encoding"] = face_data["encoding"].tolist()
            
            with open(filename, "w") as f:
                json.dump(face_data_list, f, indent=2)
        else:
            # 导出其他表
            with sqlite3.connect('worker_monitoring.db') as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                df.to_json(filename, orient="records", indent=2)
        
        messagebox.showinfo("Success", f"Data exported to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error: {e}")

def backup_database():
    """备份整个数据库"""
    try:
        backup_file = f"worker_monitoring_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        with sqlite3.connect('worker_monitoring.db') as conn:
            # 创建备份连接
            backup_conn = sqlite3.connect(backup_file)
            conn.backup(backup_conn)
            backup_conn.close()
            
        messagebox.showinfo("Success", f"Database backed up to {backup_file}")
    except Exception as e:
        messagebox.showerror("Error", f"Error: {e}")

def exit_system(root=None):
    """
    完全退出系统，确保释放所有资源
    
    参数:
        root: Tkinter根窗口，如果提供则销毁
    """
    # 关闭可能存在的任何报警窗口
    global alert_window
    if (alert_window is not None and alert_window.winfo_exists()):
        alert_window.destroy()
    
    # 尝试关闭任何可能打开的摄像头
    try:
        # 列出所有可能打开的摄像头
        for i in range(12):  # 尝试关闭0-4号摄像头
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
    except Exception as e:
        print(f"Warning during camera cleanup: {e}")
    
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    
    # 销毁当前窗口
    if root is not None and root.winfo_exists():
        root.destroy()
    
    print("Exiting Worker Monitoring System...")
    
    # 完全退出程序
    sys.exit(0)

def main_menu():
    """主菜单页面"""
    root = Tk()
    root.title("Worker Absence Monitoring - Hybrid Mode")
    root.geometry("400x550")
    
    font = set_english_font()
    
    # 确保数据库初始化
    init_database()
    
    Label(root, text="Worker Absence Monitoring", font=tkFont.Font(family="Arial", size=18)).pack(pady=20)
    Label(root, text="HOG + YOLOv5 Hybrid Detection", font=tkFont.Font(family="Arial", size=12)).pack(pady=5)
    
    Button(root, text="Face Enrollment", command=lambda: [root.destroy(), create_face_enrollment_page()], font=font).pack(pady=15)
    Button(root, text="Absence Monitoring", command=lambda: [root.destroy(), create_monitoring_page()], font=font).pack(pady=15)
    Button(root, text="Data Management", command=lambda: [root.destroy(), create_data_management_page()], font=font).pack(pady=15)
    
    # 修改退出按钮，使用新的exit_system函数
    Button(root, text="Exit System", command=lambda: exit_system(root), font=font).pack(pady=15)
    
    # 版本和版权信息
    version_label = Label(root, text="Version: v1.1 - Hybrid Detection Enhanced", font=("Arial", 8))
    version_label.pack(side=BOTTOM, pady=5)
    copyright_label = Label(root, text="© 2025 Embedded AI Specialty", font=("Arial", 8))
    copyright_label.pack(side=BOTTOM)
    
    # 添加窗口关闭事件处理
    root.protocol("WM_DELETE_WINDOW", lambda: exit_system(root))
    
    root.mainloop()

if __name__ == "__main__":
    # 确保数据库初始化
    init_database()
    
    # 自动执行JSON到数据库的迁移（仅首次运行）
    if os.path.exists("face_library") and not os.path.exists(".migration_completed"):
        migrate_json_to_db()
        # 标记迁移已完成
        with open(".migration_completed", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    main_menu()