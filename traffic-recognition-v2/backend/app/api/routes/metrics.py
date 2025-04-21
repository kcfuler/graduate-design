from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

from app.models import model_manager

router = APIRouter()

# 模拟数据 - 在实际应用中，这些数据应该从数据库或模型评估中获取
model_metrics = {
    "mobilenet_v2_tsr": {
        "accuracy": 0.92,
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "inference_time": 45,  # ms
        "last_updated": "2023-11-15"
    },
    "yolov5s_tsr": {
        "accuracy": 0.89,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "inference_time": 75,  # ms
        "mAP50": 0.87,
        "last_updated": "2023-11-10"
    }
}

@router.get("")
async def get_all_models_metrics():
    """
    获取所有模型的性能指标
    """
    available_models = model_manager.get_available_models()
    
    result = []
    for model in available_models:
        model_id = model["id"]
        metrics = model_metrics.get(model_id, {})
        result.append({
            "model_id": model_id,
            "model_name": model["name"],
            "metrics": metrics
        })
    
    return {"models_metrics": result}

@router.get("/{model_id}")
async def get_model_metrics(model_id: str):
    """
    获取特定模型的性能指标
    """
    # 验证模型存在
    model_info = next((m for m in model_manager.get_available_models() 
                      if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"未找到ID为'{model_id}'的模型")
    
    # 获取模型指标
    metrics = model_metrics.get(model_id, {})
    
    return {
        "model_id": model_id,
        "model_name": model_info["name"],
        "metrics": metrics
    }

@router.post("/{model_id}/evaluate")
async def evaluate_model(model_id: str):
    """
    触发模型评估 (模拟实现)
    """
    # 验证模型存在
    model_info = next((m for m in model_manager.get_available_models() 
                      if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"未找到ID为'{model_id}'的模型")
    
    # 在实际应用中，这里将触发模型在测试集上的评估
    # 这里只是简单返回成功消息
    
    return {
        "status": "success",
        "message": f"已为模型 '{model_id}' 触发评估。实际评估需要在后续版本中实现。"
    } 