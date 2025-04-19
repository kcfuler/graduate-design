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
        # 检查图像是否为None
        if image is None:
            raise ValueError("输入图像不能为None")
        
        # 确保图像是numpy数组并且有正确的维度
        if not isinstance(image, np.ndarray):
            raise TypeError("输入图像必须是numpy数组")
        
        # 检查图像是否为空或尺寸异常
        if image.size == 0 or len(image.shape) < 2:
            raise ValueError("输入图像数据异常，无法处理")
        
        # 调整大小
        try:
            image = cv2.resize(image, target_size)
        except Exception as e:
            raise ValueError(f"图像大小调整失败: {str(e)}")
        
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
                      font_scale: float = 0.6, thickness: int = 2,
                      text_color: Tuple[int, int, int] = (0, 255, 0),
                      box_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            results: 检测结果列表
            font_scale: 字体大小
            thickness: 线条粗细
            text_color: 文本颜色 (BGR)
            box_color: 边界框颜色 (BGR)
            
        Returns:
            绘制了检测结果的图像
        """
        # 检查输入
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("输入图像必须是有效的numpy数组")
            
        if results is None or not isinstance(results, list):
            # 如果没有结果，返回原始图像
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 转换为 BGR 格式用于显示
        display_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        
        # 图像尺寸
        h, w = image.shape[:2]
        
        for i, result in enumerate(results):
            # 获取类别名称和置信度，使用get方法避免KeyError
            class_name = result.get('class_name', 'Unknown')
            confidence = result.get('confidence', 0.0)
            
            # 提取边界框信息（如果存在）
            box = result.get('box', None)
            
            # 文本显示位置
            if box is not None and len(box) == 4:
                # 如果有边界框，在框顶部显示文本
                x1, y1, x2, y2 = [int(coord) for coord in box]
                text_pos = (x1, y1 - 10)
                
                # 确保边界框坐标在图像范围内
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # 绘制边界框
                cv2.rectangle(display_image, (x1, y1), (x2, y2), box_color, thickness)
            else:
                # 如果没有边界框，在图像顶部显示文本
                text_pos = (10, 30 + i * 30)  # 每个结果显示在不同位置
            
            # 准备显示文本
            text = f"{class_name}: {confidence:.2f}"
            
            # 绘制文本背景（提高可读性）
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(display_image, 
                         (text_pos[0], text_pos[1] - text_h - 5),
                         (text_pos[0] + text_w, text_pos[1] + 5),
                         (0, 0, 0), -1)  # 黑色背景
            
            # 绘制文本
            cv2.putText(display_image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, text_color, thickness)
        
        return display_image 