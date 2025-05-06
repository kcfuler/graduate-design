import os
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.config import get_model_by_id, get_registered_models, ROOT_DIR, load_model_config
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
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        model_info = get_model_by_id(model_id)
        if not model_info:
            raise ValueError(f"未找到ID为'{model_id}'的模型")
        
        # Load the full config to get settings.model_dir
        full_config = load_model_config()
        settings = full_config.get("settings", {})
        model_base_dir_name = settings.get("model_dir")

        if not model_base_dir_name:
            # Default to "models" if not specified, or raise an error
            # For now, let's be strict as per typical configurations.
            raise ValueError("配置文件 settings 中缺少 'model_dir' 字段")

        model_file_name = model_info.get("file_name")
        if not model_file_name:
            raise ValueError(f"模型 '{model_id}' 的配置中缺少 'file_name' 字段")

        # ROOT_DIR from config.py is project_root/backend.
        # Model files are expected in project_root/models_folder_name/
        actual_models_root_dir = ROOT_DIR.parent 
        model_path = actual_models_root_dir / model_base_dir_name / model_file_name
        
        # Get model type for potential error message (config.json uses "model_type")
        effective_model_type = model_info.get("model_type") 
        
        # Ensure model file exists (or handle gracefully if model classes support it)
        if not os.path.exists(model_path):
            # Original code printed a warning and used a "placeholder model".
            # Here, we'll print a warning. The model class itself will determine if it can proceed.
            print(f"警告：模型文件不存在: {model_path}。模型加载可能会失败或使用默认行为。")
        
        # Instantiate based on model_id (which are keys from config, e.g., "yolov11")
        # Pass model_path as a string to constructors
        if model_id == "mobilenet_v3":
            model = MobileNetV3Model(str(model_path))
        elif model_id == "yolov11":
            model = YOLOv11Model(str(model_path))
        else:
            # Use effective_model_type (which comes from "model_type" in config) in the error message
            raise ValueError(f"不支持的模型ID '{model_id}' (类型: '{effective_model_type}')")
        
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