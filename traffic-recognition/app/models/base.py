from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """基础模型接口类"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        初始化模型
        
        Args:
            model_path: 模型文件路径
            device: 运行设备，可选 "cpu" 或 "cuda"
        """
        self.device = device
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理
        
        Args:
            image: 输入图像，numpy 数组格式
            
        Returns:
            预处理后的张量
        """
        pass
    
    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        """
        后处理推理结果
        
        Args:
            output: 模型输出张量
            
        Returns:
            处理后的结果列表，每个结果包含类别、置信度等信息
        """
        pass
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        执行推理
        
        Args:
            image: 输入图像，numpy 数组格式
            
        Returns:
            推理结果列表
        """
        # 预处理
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 后处理
        results = self.postprocess(output)
        return results
    
    def to(self, device: str) -> None:
        """
        切换运行设备
        
        Args:
            device: 目标设备，"cpu" 或 "cuda"
        """
        if self.model:
            self.model = self.model.to(device)
        self.device = device


class ModelFactory:
    """模型工厂类"""
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """
        注册模型类
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """
        创建模型实例
        
        Args:
            name: 模型名称
            **kwargs: 模型初始化参数
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果模型名称未注册
        """
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered")
        return cls._models[name](**kwargs)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        获取所有可用的模型名称
        
        Returns:
            模型名称列表
        """
        return list(cls._models.keys()) 