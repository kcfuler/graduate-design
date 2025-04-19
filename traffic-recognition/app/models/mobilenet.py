import torch
import torchvision
import numpy as np
from typing import Any, Dict, List
import os
import logging
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from .base import BaseModel

# 忽略 urllib3 v2 关于 OpenSSL 版本低于 1.1.1 的警告
# 这通常是因为系统 Python 使用了 LibreSSL (如 macOS)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# 设置日志记录器
logger = logging.getLogger(__name__)

class MobileNetModel(BaseModel):
    """MobileNet 模型实现"""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        初始化 MobileNet 模型
        
        Args:
            model_path: (可选) 自定义模型权重文件路径。如果为 None，则加载预训练的 ImageNet 权重。
            device: 运行设备
        """
        # --- 在 super().__init__ 之前加载类别名称 ---
        self.class_names = []
        class_names_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            logger.info(f"成功从 {class_names_path} 加载 {len(self.class_names)} 个类别名称。")
        except FileNotFoundError:
            logger.error(f"错误：类别名称文件未找到于 {class_names_path}。MobileNetModel 将无法正确分类。")
            # 根据需要，可以抛出错误或使用默认列表
            # raise FileNotFoundError(f"Class names file not found at {class_names_path}")
            self.class_names = [] # 或者提供一个小的默认列表以允许程序继续运行
        except Exception as e:
            logger.error(f"加载类别名称文件时出错: {e}", exc_info=True)
            self.class_names = []
            
        # --- 现在调用 super().__init__ --- 
        # 这将触发 load_model，此时 self.class_names 已经存在
        super().__init__(model_path, device)
        
        # 移除这里的 class_names 定义，因为它已经在上面加载了
        # self.class_names = [
        #     "限速20", "限速30", ..., "其他"
        # ]

    def load_model(self) -> None:
        """
        加载模型
        
        使用self.model_path作为模型文件路径加载模型
        """
        # 加载预训练的 MobileNet 模型
        # 使用 weights 参数替换旧的 pretrained=True
        self.model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # 确保 self.class_names 已被加载
        if not self.class_names:
            logger.warning("MobileNetModel 的 class_names 为空，分类器层将使用默认大小 (1000) 或可能失败。")
            # 尝试从预训练模型的元数据获取类别数（如果可用）
            try:
                num_classes = len(torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"])
            except Exception:
                 num_classes = 1000 # ImageNet 默认
        else:
             num_classes = len(self.class_names)
             
        logger.info(f"设置 MobileNet 分类器输出类别数为: {num_classes}")
        
        # 修改最后一层以适应我们的分类任务
        # num_classes = len(self.class_names) # 旧代码
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )
        
        # 如果提供了自定义权重路径，则加载
        if self.model_path:
            try:
                 logger.info(f"尝试从 {self.model_path} 加载自定义权重...")
                 state_dict = torch.load(self.model_path, map_location=self.device)
                 self.model.load_state_dict(state_dict)
                 logger.info(f"成功加载自定义权重: {self.model_path}")
            except FileNotFoundError:
                 logger.error(f"错误：自定义权重文件 {self.model_path} 未找到。将使用 ImageNet 预训练权重。")
            except Exception as e:
                 logger.error(f"加载自定义权重 {self.model_path} 时出错: {e}。将使用 ImageNet 预训练权重。", exc_info=True)
        else:
             logger.info("未提供自定义权重路径，使用 ImageNet 预训练权重。")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        图像预处理
        
        Args:
            image: 输入图像，numpy 数组格式
            
        Returns:
            预处理后的张量
        """
        # 检查图像是否为None
        if image is None:
            raise ValueError("输入图像不能为None")
            
        # 确保图像是numpy数组
        if not isinstance(image, np.ndarray):
            raise TypeError("输入图像必须是numpy数组")
        
        # 检查图像数据
        if image.size == 0 or len(image.shape) < 2:
            raise ValueError("输入图像数据异常，无法处理")
        
        # 转换为 RGB 格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) > 2 and image.shape[2] == 4:
            image = image[:, :, :3]
        elif len(image.shape) < 3:
            raise ValueError("输入图像格式错误，无法转换为RGB格式")
        
        try:
            # 调整大小
            image = torchvision.transforms.functional.resize(
                torch.from_numpy(image).permute(2, 0, 1),
                (224, 224)
            )
            
            # 标准化
            image = image.float() / 255.0
            image = torchvision.transforms.functional.normalize(
                image,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # 添加批次维度
            image = image.unsqueeze(0)
            return image
        except Exception as e:
            raise ValueError(f"图像预处理失败: {str(e)}")
    
    def postprocess(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        """
        后处理推理结果
        
        Args:
            output: 模型输出张量
            
        Returns:
            处理后的结果列表
        """
        # 检查输出是否为None
        if output is None:
            return []
            
        # 检查输出是否为张量
        if not isinstance(output, torch.Tensor):
            raise TypeError("输出必须是PyTorch张量")
            
        try:
            # 获取预测结果
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_class = torch.topk(probabilities, 1)
            
            # 转换为列表格式
            results = []
            for i in range(len(top_class)):
                class_id = int(top_class[i].item())
                # 确保类别ID在有效范围内
                if 0 <= class_id < len(self.class_names):
                    result = {
                        "class_id": class_id,
                        "class_name": self.class_names[class_id],
                        "confidence": float(top_prob[i].item())
                    }
                else:
                    # 如果类别ID超出范围，使用默认值
                    result = {
                        "class_id": class_id,
                        "class_name": "未知类别",
                        "confidence": float(top_prob[i].item())
                    }
                results.append(result)
            
            return results
        except Exception as e:
            # 异常情况下返回空列表
            print(f"后处理错误: {str(e)}")
            return [] 