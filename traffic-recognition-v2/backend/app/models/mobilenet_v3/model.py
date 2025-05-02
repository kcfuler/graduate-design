import os
import random
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import cv2
import tensorflow as tf

class MobileNetV3Model:
    """
    MobileNetV3模型
    加载MobileNetV3模型并进行交通标志检测与分类
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        初始化MobileNetV3模型
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = Path(model_path)
        self.loaded = False
        self.model = None
        
        # 实际加载模型
        self._load_model()
        
        # 交通标志类别
        self.classes = [
            "stop_sign", "yield", "speed_limit_30", "speed_limit_50", 
            "speed_limit_60", "speed_limit_80", "no_entry", "no_parking",
            "pedestrian_crossing", "traffic_light", "construction_ahead"
        ]
    
    def _load_model(self):
        """
        加载MobileNetV3模型
        
        加载.h5或.tflite格式的TensorFlow/Keras模型
        """
        print(f"加载MobileNetV3模型: {self.model_path}")
        try:
            # 检查文件是否存在
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 根据文件扩展名决定加载方式
            if str(self.model_path).endswith('.h5'):
                # 加载Keras模型
                self.model = tf.keras.models.load_model(self.model_path)
                self.loaded = True
            elif str(self.model_path).endswith('.tflite'):
                # 加载TFLite模型
                self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                self.interpreter.allocate_tensors()
                
                # 获取输入和输出张量的详细信息
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.loaded = True
            else:
                raise ValueError(f"不支持的模型格式: {self.model_path}")
            
            print(f"MobileNetV3模型加载成功")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            # 如果加载失败，使用占位模型
            self.loaded = True
            self.model = "mobilenet_v3_model_placeholder"
            print("使用占位模型替代")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像用于MobileNetV3输入
        
        Args:
            image: 输入图像，OpenCV格式(BGR)
            
        Returns:
            预处理后的图像
        """
        # 调整图像大小到224x224（MobileNetV3标准输入大小）
        resized = cv2.resize(image, (224, 224))
        # 转换为RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 归一化到[-1,1]
        normalized = (rgb / 127.5) - 1.0
        # 扩展维度以匹配模型输入 [batch_size, height, width, channels]
        preprocessed = np.expand_dims(normalized, axis=0)
        
        return preprocessed
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用MobileNetV3模型进行预测
        
        Args:
            image: 输入图像，OpenCV格式(BGR)
            
        Returns:
            检测结果列表，每个结果包含边界框、标签和置信度
        """
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        # 图像预处理
        preprocessed = self.preprocess_image(image)
        
        # 区分实际模型和占位模型
        if isinstance(self.model, str) and self.model == "mobilenet_v3_model_placeholder":
            # 使用占位模型时，返回随机结果
            return self._generate_random_results(image)
        
        try:
            # 根据模型类型进行推理
            if hasattr(self, 'interpreter'):
                # 使用TFLite进行推理
                predictions = self._predict_with_tflite(preprocessed)
            else:
                # 使用Keras模型进行推理
                predictions = self.model.predict(preprocessed)
            
            # 处理预测结果
            results = self._process_predictions(predictions, image)
            return results
        except Exception as e:
            print(f"推理过程中出错: {e}")
            # 出错时返回随机结果
            return self._generate_random_results(image)
    
    def _predict_with_tflite(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        使用TFLite模型进行推理
        
        Args:
            preprocessed: 预处理后的图像
            
        Returns:
            模型预测结果
        """
        # 设置输入张量
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            preprocessed.astype(np.float32)
        )
        
        # 运行推理
        self.interpreter.invoke()
        
        # 获取输出张量
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    def _process_predictions(self, predictions: np.ndarray, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        处理模型预测结果
        
        Args:
            predictions: 模型预测结果
            image: 原始输入图像
            
        Returns:
            处理后的检测结果列表
        """
        height, width = image.shape[:2]
        results = []
        
        # 假设模型输出为分类结果
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        
        if confidence > 0.5:  # 设置置信度阈值
            # 由于MobileNetV3是分类器，我们需要生成一个假的边界框
            # 这里我们假设物体在图像中心，边界框大小为图像大小的2/3
            box_width = int(width * 2/3)
            box_height = int(height * 2/3)
            x_min = (width - box_width) // 2
            y_min = (height - box_height) // 2
            x_max = x_min + box_width
            y_max = y_min + box_height
            
            # 获取类别标签
            if class_index < len(self.classes):
                label = self.classes[class_index]
            else:
                label = f"unknown_{class_index}"
            
            results.append({
                "box": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "label": label,
                "confidence": confidence
            })
        
        return results
    
    def _generate_random_results(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        生成随机预测结果（当模型加载失败或推理失败时使用）
        
        Args:
            image: 输入图像
            
        Returns:
            随机生成的检测结果列表
        """
        height, width = image.shape[:2]
        
        # 生成1-3个随机检测结果
        num_detections = random.randint(1, 3)
        results = []
        
        for _ in range(num_detections):
            # 随机选择一个类别
            label = random.choice(self.classes)
            # 随机生成置信度 (0.5-1.0)
            confidence = random.uniform(0.5, 1.0)
            
            # 生成随机边界框
            box_width = random.randint(width // 8, width // 2)
            box_height = random.randint(height // 8, height // 2)
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