from pydantic import BaseModel, Field
from typing import List, Optional

class ModelBase(BaseModel):
    """模型基础信息"""
    id: str = Field(..., description="模型唯一ID")
    name: str = Field(..., description="模型名称")
    type: str = Field(..., description="模型类型，如'mobilenet'或'yolo'")
    description: Optional[str] = Field(None, description="模型描述")

class ModelDetail(ModelBase):
    """模型详细信息，包含路径"""
    path: str = Field(..., description="模型文件路径")

class ModelList(BaseModel):
    """模型列表响应"""
    models: List[ModelBase] = Field(..., description="可用模型列表")

class SetActiveModelRequest(BaseModel):
    """设置当前活动模型请求"""
    model_id: str = Field(..., description="要激活的模型ID") 