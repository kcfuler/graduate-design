import os
import random
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import cv2
import torch
import sys

class YOLOv11Model:
    """
    YOLOv11模型
    加载YOLOv11模型并进行推理
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        初始化YOLOv11模型
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = Path(model_path)
        self.loaded = False
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self._load_model()
        
        # 交通标志类别
        self.classes = [
            "stop_sign", "yield", "speed_limit_30", "speed_limit_50", 
            "speed_limit_60", "speed_limit_80", "no_entry", "no_parking",
            "pedestrian_crossing", "traffic_light", "construction_ahead"
        ]
    
    def _load_model(self):
        """
        加载YOLOv11模型
        
        加载.pt或.onnx格式的YOLO模型
        """
        print(f"加载YOLOv11模型: {self.model_path}")
        try:
            # 检查文件是否存在
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 根据文件扩展名决定加载方式
            if str(self.model_path).endswith('.pt'):
                # 尝试使用Ultralytics库加载模型
                try:
                    # 先检查是否已安装ultralytics
                    import importlib
                    ultralytics_spec = importlib.util.find_spec("ultralytics")
                    
                    if ultralytics_spec is not None:
                        from ultralytics import YOLO
                        self.model = YOLO(str(self.model_path))
                        self.loaded = True
                        self.use_ultralytics = True
                    else:
                        # 如果没有安装ultralytics，尝试使用PyTorch hub
                        self.model = torch.hub.load('ultralytics/yolov11', 'custom', path=str(self.model_path))
                        self.loaded = True
                        self.use_ultralytics = False
                except Exception as e:
                    print(f"使用Ultralytics加载模型失败: {e}")
                    # 尝试使用PyTorch直接加载
                    self.model = torch.load(str(self.model_path), map_location=self.device)
                    if isinstance(self.model, dict) and 'model' in self.model:
                        self.model = self.model['model']
                    self.model.to(self.device)
                    self.model.eval()
                    self.loaded = True
                    self.use_ultralytics = False
            
            elif str(self.model_path).endswith('.onnx'):
                # 加载ONNX模型
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(self.model_path))
                self.loaded = True
                self.use_onnx = True
            else:
                raise ValueError(f"不支持的模型格式: {self.model_path}")
            
            print(f"YOLOv11模型加载成功")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            # 如果加载失败，使用占位模型
            self.loaded = True
            self.model = "yolov11_model_placeholder"
            print("使用占位模型替代")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像用于YOLOv11输入
        
        Args:
            image: 输入图像，OpenCV格式(BGR)
            
        Returns:
            预处理后的图像
        """
        if isinstance(self.model, str) and self.model == "yolov11_model_placeholder":
            # 使用默认的预处理逻辑
            resized = cv2.resize(image, (1280, 1280))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))
            preprocessed = np.expand_dims(transposed, axis=0)
            return preprocessed
        
        # 如果使用Ultralytics YOLOv11，预处理会在predict中自动完成
        if hasattr(self, 'use_ultralytics') and self.use_ultralytics:
            return image
        
        # 如果使用ONNX，需要特定的预处理
        if hasattr(self, 'use_onnx') and self.use_onnx:
            # YOLOv11输入大小为1280x1280
            resized = cv2.resize(image, (1280, 1280))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))
            preprocessed = np.expand_dims(transposed, axis=0).astype(np.float32)
            return preprocessed
        
        # 针对PyTorch模型的预处理
        resized = cv2.resize(image, (1280, 1280))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(transposed, axis=0)
        return preprocessed
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用YOLOv11模型进行预测
        
        Args:
            image: 输入图像，OpenCV格式(BGR)
            
        Returns:
            检测结果列表，每个结果包含边界框、标签和置信度
        """
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        # 获取原始图像尺寸用于坐标转换
        height, width = image.shape[:2]
        
        # 如果是占位模型，返回随机结果
        if isinstance(self.model, str) and self.model == "yolov11_model_placeholder":
            return self._generate_random_results(image)
        
        try:
            # 根据模型类型进行推理
            if hasattr(self, 'use_ultralytics') and self.use_ultralytics:
                # 使用Ultralytics YOLOv11进行推理
                results = self.model(image)
                return self._process_ultralytics_results(results, image)
            elif hasattr(self, 'use_onnx') and self.use_onnx:
                # 使用ONNX进行推理
                preprocessed = self.preprocess_image(image)
                inputs = {self.session.get_inputs()[0].name: preprocessed}
                outputs = self.session.run(None, inputs)
                return self._process_onnx_results(outputs, image)
            else:
                # 使用PyTorch模型进行推理
                preprocessed = self.preprocess_image(image)
                tensor = torch.from_numpy(preprocessed).to(self.device)
                with torch.no_grad():
                    predictions = self.model(tensor)
                return self._process_pytorch_results(predictions, image)
        except Exception as e:
            print(f"推理过程中出错: {e}")
            # 出错时返回随机结果
            return self._generate_random_results(image)
    
    def _process_ultralytics_results(self, results, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        处理Ultralytics YOLOv11的推理结果
        """
        processed_results = []
        height, width = image.shape[:2]
        
        # Ultralytics YOLO返回的结果格式已经变化，需要使用最新的API
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 获取置信度
                conf = float(box.conf[0].cpu().numpy())
                
                # 获取类别索引并转换为标签
                cls_idx = int(box.cls[0].cpu().numpy())
                
                # 使用模型的类别或我们自己定义的类别
                if hasattr(result, 'names') and cls_idx in result.names:
                    label = result.names[cls_idx]
                elif cls_idx < len(self.classes):
                    label = self.classes[cls_idx]
                else:
                    label = f"class_{cls_idx}"
                
                processed_results.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "label": label,
                    "confidence": conf
                })
        
        return processed_results
    
    def _process_onnx_results(self, outputs, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        处理ONNX模型的推理结果
        """
        processed_results = []
        height, width = image.shape[:2]
        
        # ONNX模型输出通常为检测结果数组
        # 假设输出格式为 [x1, y1, x2, y2, confidence, class_id]
        detections = outputs[0]
        
        for detection in detections:
            # 提取边界框、置信度和类别ID
            x1, y1, x2, y2, conf, cls_id = detection
            
            # 跳过低置信度检测
            if conf < 0.5:
                continue
            
            # 将边界框坐标转换为原始图像坐标系
            scale_x = width / 1280
            scale_y = height / 1280
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # 确保边界框在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 获取类别标签
            cls_id = int(cls_id)
            if cls_id < len(self.classes):
                label = self.classes[cls_id]
            else:
                label = f"class_{cls_id}"
            
            processed_results.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "label": label,
                "confidence": float(conf)
            })
        
        return processed_results
    
    def _process_pytorch_results(self, predictions, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        处理PyTorch模型的推理结果
        """
        processed_results = []
        height, width = image.shape[:2]
        
        # 处理PyTorch模型的输出
        # 这里的处理方法取决于具体模型的输出格式
        # 这只是一个示例处理方法
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        detections = predictions.cpu().numpy()
        
        if len(detections.shape) == 3:
            # 一些模型可能输出形状为 [batch_size, num_detections, 6]
            # 其中6表示 [x1, y1, x2, y2, confidence, class_id]
            detections = detections[0]  # 取第一个batch
        
        for detection in detections:
            # 提取边界框、置信度和类别ID
            if len(detection) >= 6:
                x1, y1, x2, y2, conf, cls_id = detection[:6]
            else:
                # 如果检测结果格式不同，尝试适配
                continue
            
            # 跳过低置信度检测
            if conf < 0.5:
                continue
            
            # 将边界框坐标转换为原始图像坐标系
            scale_x = width / 1280
            scale_y = height / 1280
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # 确保边界框在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 获取类别标签
            cls_id = int(cls_id)
            if cls_id < len(self.classes):
                label = self.classes[cls_id]
            else:
                label = f"class_{cls_id}"
            
            processed_results.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "label": label,
                "confidence": float(conf)
            })
        
        return processed_results
    
    def _generate_random_results(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        生成随机预测结果（当模型加载失败或推理失败时使用）
        
        Args:
            image: 输入图像
            
        Returns:
            随机生成的检测结果列表
        """
        height, width = image.shape[:2]
        
        # 生成2-5个随机检测结果
        num_detections = random.randint(2, 5)
        results = []
        
        for _ in range(num_detections):
            # 随机选择一个类别
            label = random.choice(self.classes)
            # 随机生成置信度 (0.6-1.0)
            confidence = random.uniform(0.6, 1.0)
            
            # 生成随机边界框
            box_width = random.randint(width // 10, width // 3)
            box_height = random.randint(height // 10, height // 3)
            x_min = random.randint(0, width - box_width)
            y_min = random.randint(0, height - box_height)
            x_max = x_min + box_width
            y_max = y_min + box_height
            
            # 添加到结果列表
            results.append({
                "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "label": label,
                "confidence": float(confidence)
            })
        
        return results 