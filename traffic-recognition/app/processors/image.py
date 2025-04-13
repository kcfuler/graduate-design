import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path


class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self):
        """初始化图像处理器"""
        self._supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            加载的图像数组，如果加载失败则返回 None
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            # 转换为 RGB 格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def preprocess(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像
        """
        # 调整大小
        image = cv2.resize(image, target_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def batch_preprocess(self, images: List[np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
        """
        批量图像预处理
        
        Args:
            images: 输入图像列表
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像列表
        """
        return [self.preprocess(img, target_size) for img in images]
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """
        检查文件格式是否支持
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否支持该格式
        """
        return Path(file_path).suffix.lower() in self._supported_formats
    
    def draw_detection(self, image: np.ndarray, results: List[dict], 
                      font_scale: float = 0.6, thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            results: 检测结果列表
            font_scale: 字体大小
            thickness: 线条粗细
            
        Returns:
            绘制了检测结果的图像
        """
        # 转换为 BGR 格式用于显示
        display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for result in results:
            class_name = result.get('class_name', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            # 在图像上添加文本
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 255, 0), thickness)
        
        return display_image 