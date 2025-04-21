from fastapi import APIRouter, HTTPException

from app.models import model_manager
from app.schemas.model import ModelList, SetActiveModelRequest

router = APIRouter()

@router.get("", response_model=ModelList)
async def get_models():
    """
    获取所有可用模型
    """
    models = model_manager.get_available_models()
    return {"models": models}

@router.get("/active")
async def get_active_model():
    """
    获取当前活动模型
    """
    active_model_id = model_manager.get_active_model_id()
    if not active_model_id:
        return {"active_model": None}
    
    model_info = next((m for m in model_manager.get_available_models() 
                        if m["id"] == active_model_id), None)
    
    return {"active_model": model_info}

@router.post("/active")
async def set_active_model(request: SetActiveModelRequest):
    """
    设置当前活动模型
    """
    try:
        model_manager.set_active_model(request.model_id)
        return {"message": f"模型 '{request.model_id}' 已设置为当前活动模型"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) 