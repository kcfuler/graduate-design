from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from pathlib import Path
import os
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    基础模型接口类
    
    这个抽象类定义了所有模型实现的基本接口。子类必须实现抽象方法，
    包括加载模型、预处理输入和后处理输出的方法。
    
    模型路径处理:
    1. 可以是完整的文件路径 (例如 "/path/to/model.pt")
    2. 可以是模型名称 (例如 "yolov8n")，某些特定模型实现可能会将其转换为适当的路径
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        初始化模型
        
        Args:
            model_path: 模型文件路径或模型名称。可以是：
                        - 完整的文件路径 (如 "/path/to/model.pt")
                        - 不带扩展名的文件名 (如 "yolov8n")，会自动检查对应扩展名
                        - 预训练模型标识符 (某些特定模型如YOLO支持)
            device: 运行设备，可选 "cpu" 或 "cuda"
                    
        注意:
            - 如果提供了model_path，将自动调用load_model()方法
            - 子类应确保正确实现load_model()方法
        """
        self.device = device
        self.model = None
        self.model_path = model_path
        
        if model_path:
            # 检查模型路径是否为文件路径
            path_obj = Path(model_path)
            if path_obj.exists() and path_obj.is_file():
                logger.info(f"找到模型文件: {model_path}")
            else:
                # 检查是否有.pt扩展名
                if not model_path.endswith('.pt'):
                    potential_path = f"{model_path}.pt"
                    potential_path_obj = Path(potential_path)
                    if potential_path_obj.exists() and potential_path_obj.is_file():
                        logger.info(f"使用模型文件: {potential_path}")
                        self.model_path = potential_path
                    else:
                        logger.warning(f"模型文件未找到: {model_path}，将尝试使用模型名称直接加载")
                else:
                    logger.warning(f"模型文件未找到: {model_path}，将尝试使用模型名称直接加载")
            
            # 尝试加载模型
            self.load_model()
    
    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型
        
        子类必须实现此方法以加载其特定的模型类型。
        模型路径可以通过self.model_path属性获取，该属性在__init__中设置。
        
        子类实现应该:
        1. 检查self.model_path是否有效
        2. 加载模型到self.model
        3. 处理任何加载过程中的异常
        4. 将模型移动到正确的设备(self.device)
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
    """
    模型工厂类
    
    此类负责创建和管理模型实例。它提供了一个注册机制，允许应用程序注册
    不同类型的模型实现，然后通过名称创建这些模型的实例。
    
    使用方法:
    1. 首先注册模型类: ModelFactory.register_model("yolo", YOLOModel)
    2. 然后创建模型实例: model = ModelFactory.create_model("yolo", model_path="yolov8n.pt")
    
    模型注册应该在应用程序初始化时完成，如在模型模块的__init__.py文件中。
    """
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """
        注册模型类
        
        Args:
            name: 模型名称，用于后续创建模型实例时标识模型类型
            model_class: 模型类，必须是BaseModel的子类
        """
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        """
        创建模型实例
        
        此方法执行以下步骤:
        1. 检查模型名称是否已注册
        2. 处理模型路径参数，确保至少提供基本的模型标识符
        3. 对特定模型类型(如YOLO)进行特殊处理
        4. 创建并返回模型实例
        
        Args:
            name: 模型名称，必须已通过register_model注册
            **kwargs: 模型初始化参数，常用参数包括:
                      - model_path: 模型文件路径或模型名称
                      - device: 运行设备 ("cpu" 或 "cuda")
            
        Returns:
            创建的模型实例
            
        Raises:
            ValueError: 如果模型名称未注册
            RuntimeError: 如果模型创建过程中发生错误
        """
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered")
        
        # 处理模型路径
        # 如果没有提供model_path参数，则使用模型名称作为路径
        # 这样确保在创建模型时至少有一个基本的模型标识符
        if 'model_path' not in kwargs:
            logger.info(f"未提供model_path参数，使用模型名称 '{name}' 作为默认路径")
            kwargs['model_path'] = name
        
        # 检查模型路径后缀
        model_path = kwargs.get('model_path')
        if model_path and isinstance(model_path, str):
            if name.startswith('yolo') and not model_path.endswith('.pt'):
                # YOLO模型特殊处理，确保有.pt后缀（如果不是特殊的预训练名称）
                if not Path(model_path).exists() and not Path(f"{model_path}.pt").exists():
                    logger.info(f"为YOLO模型 '{name}' 添加.pt后缀")
                    kwargs['model_path'] = f"{model_path}.pt"
        
        try:
            return cls._models[name](**kwargs)
        except Exception as e:
            logger.error(f"创建模型 '{name}' 失败: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create model {name}: {e}") from e
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        获取所有可用的模型名称
        
        Returns:
            已注册的模型名称列表
        """
        return list(cls._models.keys()) 