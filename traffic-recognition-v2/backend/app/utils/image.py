import base64
import io
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image

def read_image_file(file_content: bytes) -> np.ndarray:
    """
    从上传的文件内容读取图像
    
    Args:
        file_content: 文件二进制内容
        
    Returns:
        OpenCV格式的图像（BGR）
    """
    # 使用PIL读取图像
    image = Image.open(io.BytesIO(file_content))
    # 转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # 转换为numpy数组
    img_array = np.array(image)
    # 转换为BGR（OpenCV格式）
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_bgr

def read_image_base64(base64_str: str) -> np.ndarray:
    """
    从Base64字符串读取图像
    
    Args:
        base64_str: Base64编码的图像字符串
        
    Returns:
        OpenCV格式的图像（BGR）
    """
    # 解码Base64字符串
    if base64_str.startswith('data:image'):
        # 处理data URI
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    return read_image_file(img_data)

def read_image_url(url: str) -> np.ndarray:
    """
    从URL读取图像
    
    Args:
        url: 图像URL
        
    Returns:
        OpenCV格式的图像（BGR）
    """
    import urllib.request
    
    # 下载图像数据
    with urllib.request.urlopen(url) as response:
        img_data = response.read()
    
    return read_image_file(img_data)

def resize_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None, 
                 max_size: Optional[int] = None) -> np.ndarray:
    """
    调整图像大小，保持宽高比
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)，如果指定则精确调整到此尺寸
        max_size: 最大尺寸限制，如果指定则按比例缩放至长边不超过max_size
        
    Returns:
        调整大小后的图像
    """
    height, width = image.shape[:2]
    
    if target_size:
        return cv2.resize(image, target_size)
    
    if max_size:
        # 按比例缩放
        scale = min(max_size / width, max_size / height)
        if scale < 1:  # 只在需要缩小时调整
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
    
    return image

def encode_image_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """
    将OpenCV格式图像编码为Base64字符串
    
    Args:
        image: OpenCV格式的图像（BGR）
        format: 输出格式 ('jpeg' 或 'png')
        
    Returns:
        Base64编码的图像字符串
    """
    # 转换为RGB（PIL格式）
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 创建PIL图像
    pil_img = Image.fromarray(img_rgb)
    # 创建内存缓冲区
    buffer = io.BytesIO()
    # 保存到缓冲区
    pil_img.save(buffer, format=format)
    # 获取Base64编码
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/{format};base64,{img_str}"

def draw_detection_boxes(image: np.ndarray, detections: list, 
                        color_map: Optional[dict] = None) -> np.ndarray:
    """
    在图像上绘制检测框和标签
    
    Args:
        image: 输入图像
        detections: 检测结果列表，每个元素应包含 'box', 'label', 'confidence'
        color_map: 类别颜色映射字典，格式为 {类别: (B,G,R)}
        
    Returns:
        添加了检测框的图像
    """
    # 创建图像副本
    output = image.copy()
    height, width = output.shape[:2]
    
    # 默认颜色映射
    if color_map is None:
        color_map = {
            "stop_sign": (0, 0, 255),       # 红色
            "yield": (0, 165, 255),         # 橙色
            "speed_limit_30": (0, 255, 0),  # 绿色
            "speed_limit_50": (0, 255, 0),  # 绿色
            "speed_limit_60": (0, 255, 0),  # 绿色
            "speed_limit_80": (0, 255, 0),  # 绿色
            "no_entry": (0, 0, 255),        # 红色
            "no_parking": (0, 0, 255),      # 红色
            "pedestrian_crossing": (255, 0, 0),  # 蓝色
            "traffic_light": (255, 255, 0),  # 青色
            "construction_ahead": (128, 0, 255)  # 紫色
        }
        # 默认颜色（如果标签不在映射中）
        default_color = (255, 255, 255)  # 白色
    
    # 绘制每个检测框
    for det in detections:
        # 获取框坐标
        box = det["box"]
        x_min, y_min, x_max, y_max = map(int, box)
        
        # 确保坐标在图像范围内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        # 获取标签和置信度
        label = det["label"]
        confidence = det["confidence"]
        
        # 获取颜色
        color = color_map.get(label, default_color)
        
        # 绘制边界框
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color, 2)
        
        # 准备标签文本
        text = f"{label}: {confidence:.2f}"
        
        # 获取文本大小
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 绘制文本背景
        cv2.rectangle(output, (x_min, y_min - text_height - 10), 
                    (x_min + text_width, y_min), color, -1)
        
        # 绘制文本
        cv2.putText(output, text, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return output 