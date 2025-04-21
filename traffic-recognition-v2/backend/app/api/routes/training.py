from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import uuid
import time
from datetime import datetime

from app.models import model_manager
from app.schemas.training import TrainingRequest, TrainingResponse, TrainingStatusResponse

router = APIRouter()

# 模拟训练作业数据库
training_jobs = {}

def mock_training_task(job_id: str, model_id: str, epochs: int):
    """模拟训练任务的后台处理函数"""
    # 在实际应用中，这里会调用真正的训练逻辑
    # 这里仅用于模拟训练过程
    
    for epoch in range(epochs):
        # 更新训练作业状态
        progress = (epoch + 1) / epochs * 100
        
        # 模拟一些训练指标
        loss = 0.5 - (0.3 * progress / 100)
        accuracy = 0.7 + (0.25 * progress / 100)
        
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = progress
        training_jobs[job_id]["metrics"] = {
            "current_epoch": epoch + 1,
            "total_epochs": epochs,
            "loss": loss,
            "accuracy": accuracy,
            "last_updated": datetime.now().isoformat()
        }
        
        # 模拟训练时间间隔
        time.sleep(0.5)  # 在实际应用中，这将是真正的训练时间
        
        # 检查是否要求停止训练
        if training_jobs[job_id].get("should_stop", False):
            training_jobs[job_id]["status"] = "stopped"
            return
    
    # 训练完成
    training_jobs[job_id]["status"] = "completed"
    training_jobs[job_id]["progress"] = 100.0
    training_jobs[job_id]["metrics"]["final_accuracy"] = accuracy
    training_jobs[job_id]["metrics"]["final_loss"] = loss

@router.post("", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    启动模型训练任务
    """
    # 验证模型存在
    model_info = next((m for m in model_manager.get_available_models() 
                      if m["id"] == request.model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"未找到ID为'{request.model_id}'的模型")
    
    # 创建唯一的任务ID
    job_id = str(uuid.uuid4())
    
    # 初始化任务状态
    training_jobs[job_id] = {
        "model_id": request.model_id,
        "status": "queued",
        "progress": 0.0,
        "created_at": datetime.now().isoformat(),
        "parameters": {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "dataset_path": request.dataset_path,
            **request.parameters if request.parameters else {}
        },
        "metrics": {}
    }
    
    # 启动后台训练任务
    background_tasks.add_task(
        mock_training_task, 
        job_id, 
        request.model_id,
        request.epochs
    )
    
    return {
        "status": "queued",
        "job_id": job_id,
        "model_id": request.model_id
    }

@router.get("/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """
    获取训练任务状态
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"未找到ID为'{job_id}'的训练任务")
    
    job = training_jobs[job_id]
    
    return {
        "job_id": job_id,
        "model_id": job["model_id"],
        "status": job["status"],
        "progress": job["progress"],
        "metrics": job["metrics"]
    }

@router.delete("/{job_id}")
async def stop_training(job_id: str):
    """
    停止训练任务
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"未找到ID为'{job_id}'的训练任务")
    
    job = training_jobs[job_id]
    
    if job["status"] not in ["queued", "running"]:
        raise HTTPException(status_code=400, detail=f"无法停止处于'{job['status']}'状态的任务")
    
    # 标记任务应该停止
    job["should_stop"] = True
    
    return {
        "status": "stopping",
        "job_id": job_id,
        "message": "已请求停止训练任务"
    }

@router.get("")
async def list_training_jobs():
    """
    获取所有训练任务列表
    """
    jobs_list = []
    for job_id, job in training_jobs.items():
        jobs_list.append({
            "job_id": job_id,
            "model_id": job["model_id"],
            "status": job["status"],
            "progress": job["progress"],
            "created_at": job["created_at"]
        })
    
    return {"jobs": jobs_list} 