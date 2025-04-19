import numpy as np
from typing import Any, Dict, List, Optional
from ultralytics import YOLO
import logging

from .base import BaseModel

# 设置日志记录器
logger = logging.getLogger(__name__)

class YOLOModel(BaseModel):
    """Ultralytics YOLO 模型适配器类"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化 YOLO 模型适配器
        
        Args:
            model_path: YOLO 模型文件路径 (例如 'yolov8n.pt') 或模型名称
            device: 运行设备 ("cpu" or "cuda")
        """
        # 注意：BaseModel 的 __init__ 会调用 load_model，所以这里必须先设置model_path然后传递给super
        super().__init__(model_path=model_path, device=device) 
        # BaseModel 的 __init__ 调用了 load_model，模型应已加载
        if self.model is None:
             logger.error(f"YOLO 模型未能通过 BaseModel 的 __init__ 加载: {self.model_path}")
             raise RuntimeError(f"Failed to load YOLO model: {self.model_path}")

    def load_model(self) -> None:
        """
        加载 Ultralytics YOLO 模型
        
        Args:
           # No longer takes model_path argument
        """
        try:
            logger.info(f"尝试加载 YOLO 模型: {self.model_path} 到设备 {self.device}")
            self.model = YOLO(self.model_path)
            self.to(self.device) # 确保模型移动到正确的设备
            logger.info(f"YOLO 模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"加载 YOLO 模型失败: {self.model_path}. 错误: {e}", exc_info=True)
            self.model = None # 确保模型状态为 None
            # 不在此处重新引发异常，让 __init__ 中的检查处理
            # raise RuntimeError(f"Failed to load YOLO model: {self.model_path}") from e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理 - 对于 YOLO，直接返回 NumPy 数组
        
        Args:
            image: 输入图像 (NumPy array)
            
        Returns:
            输入图像 (NumPy array)
        """
        # YOLO 模型可以直接处理 NumPy 数组
        return image

    def postprocess(self, output: Any) -> List[Dict[str, Any]]:
        """
        后处理 Ultralytics YOLO 推理结果
        
        Args:
            output: Ultralytics model's prediction results (通常是 list of Results objects)
            
        Returns:
            格式化的结果列表: List[Dict[str, Any]]
            每个字典包含: 'box', 'confidence', 'class_id', 'class_name'
        """
        results_list = []
        if not output or not isinstance(output, list) or len(output) == 0:
            logger.warning("YOLO 推理输出为空或格式不正确")
            return results_list
            
        # 假设批处理大小为 1，取第一个结果
        results = output[0] 
        
        # 检查是否有检测框
        if results.boxes is None:
            logger.info("YOLO 推理未检测到任何对象")
            return results_list

        try:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            class_names_map = results.names # Dict: {class_id: class_name}
            
            for i in range(len(boxes)):
                results_list.append({
                    "box": boxes[i].tolist(), # 转换为列表以便 JSON 序列化
                    "confidence": float(confidences[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": class_names_map.get(int(class_ids[i]), "未知类别") 
                })
        except Exception as e:
             logger.error(f"解析 YOLO 结果时出错: {e}", exc_info=True)
             # 返回空列表或部分结果，取决于需求
             return []
             
        return results_list

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        执行 YOLO 推理 (重写基类方法以优化)
        
        Args:
            image: 输入图像 (NumPy array)
            
        Returns:
            推理结果列表
        """
        if self.model is None:
             logger.error("YOLO 模型未加载，无法执行推理。")
             return [{"error": "Model not loaded"}]

        try:
            # YOLO v8 predict 方法接受 NumPy 数组
            # verbose=False 减少控制台输出
            outputs = self.model.predict(image, device=self.device, verbose=False) 
            
            # 后处理
            results = self.postprocess(outputs)
            return results
            
        except Exception as e:
            logger.error(f"YOLO 推理过程中发生错误: {e}", exc_info=True)
            return [{"error": f"Prediction failed: {str(e)}"}]

    def to(self, device: str) -> None:
        """
        将模型移动到指定设备
        
        Args:
            device: "cpu" or "cuda"
        """
        if self.model:
            try:
                # Ultralytics 模型有自己的 to 方法或在 predict 时指定 device
                # 这里我们主要更新内部状态，并在 predict 时传递 device
                # 如果需要强制移动模型本身（可能影响性能或非预期行为），可以使用：
                # self.model.to(device) 
                self.device = device
                logger.info(f"YOLO 模型的目标设备设置为: {device}")
            except Exception as e:
                 logger.error(f"移动 YOLO 模型到设备 {device} 时出错: {e}", exc_info=True)
        else:
             self.device = device # 即使模型未加载，也更新目标设备 