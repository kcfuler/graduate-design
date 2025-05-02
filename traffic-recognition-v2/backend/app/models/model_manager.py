import os
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.config import get_model_by_id, get_registered_models, ROOT_DIR
from app.models.mobilenet.model import MobileNetModel
from app.models.yolo.model import YoloModel
from app.models.mobilenet_v3.model import MobileNetV3Model
from app.models.yolov11.model import YOLOv11Model

class ModelManager:
    """
    模型管理器
    负责加载和管理不同类型的模型
    """
    
    def __init__(self):
        # 已加载模型的缓存
        self._loaded_models = {}
        # 当前活动模型ID
        self._active_model_id = None
    
    def get_available_models(self):
        """
        获取所有可用模型的信息
        """
        return get_registered_models()
    
    def load_model(self, model_id: str) -> Any:
        """
        加载指定的模型
        如果模型已经加载，则从缓存返回
        """
        # 检查模型是否已加载
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        # 获取模型信息
        model_info = get_model_by_id(model_id)
        if not model_info:
            raise ValueError(f"未找到ID为'{model_id}'的模型")
        
        # 根据模型类型加载不同的模型
        model_type = model_info.get("type")
        model_path = Path(ROOT_DIR) / model_info.get("path")
        
        # 确保模型文件存在
        if not os.path.exists(model_path):
            # 为了演示，我们允许模型文件不存在
            print(f"警告：模型文件不存在: {model_path}，使用占位模型")
        
        # 根据模型类型和ID加载不同的模型
        if model_id == "mobilenet_v3_tsr":
            model = MobileNetV3Model(model_path)
        elif model_id == "yolov11_tsr":
            model = YOLOv11Model(model_path)
        elif model_type == "mobilenet":
            model = MobileNetModel(model_path)
        elif model_type == "yolo":
            model = YoloModel(model_path)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 将模型添加到缓存
        self._loaded_models[model_id] = model
        
        return model
    
    def get_model(self, model_id: str) -> Any:
        """
        获取模型实例
        如果模型未加载，则自动加载
        """
        return self.load_model(model_id)
    
    def set_active_model(self, model_id: str) -> None:
        """
        设置当前活动模型
        """
        # 验证模型是否存在
        if not get_model_by_id(model_id):
            raise ValueError(f"未找到ID为'{model_id}'的模型")
        
        self._active_model_id = model_id
        
        # 预加载模型
        self.load_model(model_id)
    
    def get_active_model(self) -> Optional[Any]:
        """
        获取当前活动模型
        如果未设置活动模型，返回None
        """
        if not self._active_model_id:
            return None
        
        return self.get_model(self._active_model_id)
    
    def get_active_model_id(self) -> Optional[str]:
        """
        获取当前活动模型ID
        """
        return self._active_model_id

# 创建全局模型管理器实例
model_manager = ModelManager()

# 默认设置第一个可用的模型为活动模型
available_models = model_manager.get_available_models()
if available_models:
    try:
        model_manager.set_active_model(available_models[0]["id"])
    except Exception as e:
        print(f"警告：无法设置默认模型: {e}") 