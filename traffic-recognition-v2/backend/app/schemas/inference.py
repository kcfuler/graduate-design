from pydantic import BaseModel, Field
from typing import List, Optional

class DetectionBox(BaseModel):
    """检测框信息"""
    box: List[float] = Field(..., description="边界框坐标 [x_min, y_min, x_max, y_max]")
    label: str = Field(..., description="标签名称")
    confidence: float = Field(..., description="置信度")

class InferenceResponse(BaseModel):
    """推理结果响应"""
    model_id: str = Field(..., description="使用的模型ID")
    results: List[DetectionBox] = Field(..., description="检测结果列表")

class InferenceRequest(BaseModel):
    """推理请求（不包含图像数据，图像通过表单上传）"""
    model_id: str = Field(..., description="要使用的模型ID") 