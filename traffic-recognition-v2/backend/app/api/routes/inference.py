from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional

from app.utils.inference import run_inference
from app.schemas.inference import InferenceResponse

router = APIRouter()

@router.post("", response_model=InferenceResponse)
async def infer_image(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    draw_boxes: bool = Form(False)
):
    """
    使用指定模型对图像进行推理
    """
    try:
        # 检查输入
        if image is None and image_url is None:
            raise HTTPException(
                status_code=400, 
                detail="必须提供图像文件或图像URL"
            )
        
        # 获取图像数据
        image_data = None
        if image:
            image_data = await image.read()
        
        # 运行推理
        result = await run_inference(
            model_id=model_id,
            image_data=image_data,
            image_url=image_url,
            draw_boxes=draw_boxes
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理过程中发生错误: {str(e)}")

@router.post("/base64")
async def infer_base64_image(
    background_tasks: BackgroundTasks,
    model_id: str,
    image_base64: str,
    draw_boxes: bool = False
):
    """
    使用Base64编码的图像进行推理
    """
    try:
        # 运行推理
        result = await run_inference(
            model_id=model_id,
            image_base64=image_base64,
            draw_boxes=draw_boxes
        )
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理过程中发生错误: {str(e)}") 