from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes import models, inference

app = FastAPI(
    title="交通标志识别API",
    description="深度学习交通标志识别系统的后端API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(models.router, prefix="/models", tags=["模型管理"])
app.include_router(inference.router, prefix="/infer", tags=["推理"])

@app.get("/", tags=["健康检查"])
async def root():
    return {"message": "交通标志识别API在线"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 