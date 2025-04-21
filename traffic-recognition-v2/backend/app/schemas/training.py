from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TrainingRequest(BaseModel):
    """训练请求"""
    model_id: str = Field(..., description="要训练的模型ID")
    dataset_path: Optional[str] = Field(None, description="训练数据集路径")
    epochs: int = Field(10, description="训练轮数")
    batch_size: int = Field(32, description="批量大小")
    parameters: Optional[Dict[str, Any]] = Field(None, description="其他训练参数")

class TrainingResponse(BaseModel):
    """训练响应"""
    status: str = Field(..., description="训练状态")
    job_id: str = Field(..., description="训练任务ID")
    model_id: str = Field(..., description="模型ID")

class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    job_id: str = Field(..., description="训练任务ID")
    model_id: str = Field(..., description="模型ID")
    status: str = Field(..., description="任务状态")
    progress: Optional[float] = Field(None, description="训练进度（百分比）")
    metrics: Optional[Dict[str, Any]] = Field(None, description="当前训练指标") 