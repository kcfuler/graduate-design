import torch
import torchvision
import numpy as np
from typing import Any, Dict, List
from .base import BaseModel


class MobileNetModel(BaseModel):
    """MobileNet 模型实现"""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        初始化 MobileNet 模型
        
        Args:
            model_path: 模型文件路径
            device: 运行设备
        """
        super().__init__(model_path, device)
        self.class_names = [
            "限速20", "限速30", "限速50", "限速60", "限速70", "限速80", "限速100", "限速120",
            "禁止通行", "禁止左转", "禁止右转", "禁止直行", "禁止掉头",
            "注意行人", "注意儿童", "注意非机动车", "注意野生动物",
            "前方施工", "前方拥堵", "前方事故",
            "停车让行", "减速让行",
            "直行", "左转", "右转", "掉头",
            "人行横道", "非机动车道",
            "公交专用", "应急车道",
            "其他"
        ]
    
    def load_model(self, model_path: str) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
        """
        # 加载预训练的 MobileNet 模型
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        
        # 修改最后一层以适应我们的分类任务
        num_classes = len(self.class_names)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )
        
        # 如果有预训练权重，则加载
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
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
        # 转换为 RGB 格式
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
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
    
    def postprocess(self, output: torch.Tensor) -> List[Dict[str, Any]]:
        """
        后处理推理结果
        
        Args:
            output: 模型输出张量
            
        Returns:
            处理后的结果列表
        """
        # 获取预测结果
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = torch.topk(probabilities, 1)
        
        # 转换为列表格式
        results = []
        for i in range(len(top_class)):
            result = {
                "class_id": int(top_class[i].item()),
                "class_name": self.class_names[int(top_class[i].item())],
                "confidence": float(top_prob[i].item())
            }
            results.append(result)
        
        return results 