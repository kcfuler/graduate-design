from typing import Dict, List, Any, Optional, Union
import numpy as np

from app.models import model_manager
from app.utils.image import read_image_file, read_image_base64, read_image_url, draw_detection_boxes

async def run_inference(
    model_id: str,
    image_data: Optional[bytes] = None,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    draw_boxes: bool = False
) -> Dict[str, Any]:
    """
    运行推理
    
    Args:
        model_id: 要使用的模型ID
        image_data: 图像二进制数据，优先级最高
        image_base64: Base64编码的图像，优先级第二
        image_url: 图像URL，优先级最低
        draw_boxes: 是否在图像上绘制检测框
        
    Returns:
        推理结果字典，包含模型ID、检测结果，
        如果draw_boxes=True，还包含添加了检测框的图像的Base64编码
    """
    # 获取模型
    model = model_manager.get_model(model_id)
    
    # 读取图像
    if image_data is not None:
        image = read_image_file(image_data)
    elif image_base64 is not None:
        image = read_image_base64(image_base64)
    elif image_url is not None:
        image = read_image_url(image_url)
    else:
        raise ValueError("必须至少提供一种图像输入")
    
    # 运行推理
    results = model.predict(image)
    
    # 组织响应
    response = {
        "model_id": model_id,
        "results": results
    }
    
    # 如果需要，绘制检测框并添加到响应
    if draw_boxes and results:
        from app.utils.image import encode_image_base64
        
        # 绘制检测框
        image_with_boxes = draw_detection_boxes(image, results)
        
        # 添加到响应
        response["image_with_boxes"] = encode_image_base64(image_with_boxes)
    
    return response

def get_model_metrics(model_id: str) -> Dict[str, Any]:
    """
    获取模型性能指标
    
    Args:
        model_id: 模型ID
        
    Returns:
        包含模型性能指标的字典
    """
    # 验证模型是否存在
    model_info = model_manager.get_model_by_id(model_id)
    if not model_info:
        raise ValueError(f"未找到ID为'{model_id}'的模型")
    
    # 根据模型类型返回不同的指标
    model_type = model_info.get("type")
    
    # 返回模拟的性能指标
    if model_type == "mobilenet":
        metrics = {
            "accuracy": 0.92,
            "precision": 0.94,
            "recall": 0.91,
            "f1_score": 0.92,
            "inference_time_ms": 15
        }
    elif model_type == "yolo":
        metrics = {
            "map_50": 0.86,
            "map_50_95": 0.65,
            "precision": 0.88,
            "recall": 0.85,
            "inference_time_ms": 35
        }
    else:
        metrics = {}
    
    # 添加通用信息
    metrics.update({
        "model_id": model_id,
        "type": model_type,
        "name": model_info.get("name")
    })
    
    return metrics 