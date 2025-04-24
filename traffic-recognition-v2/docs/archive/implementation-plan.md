# 深度学习交通标志识别系统 - 实施计划

## 实施准备

1. 确认项目环境要求
   - Node.js (18+)
   - Python (3.8+)
   - 包管理工具: npm, pip

## 阶段一：项目基础结构搭建

### 1. 创建项目目录结构
```bash
mkdir -p frontend middleware backend
mkdir -p backend/app/api/routes backend/app/core backend/app/models backend/app/schemas backend/app/utils backend/models
mkdir -p frontend/src/components frontend/src/services frontend/src/hooks frontend/src/types frontend/src/assets
mkdir -p middleware/src/utils
```

### 2. 设置后端(Python FastAPI)

1. 创建Python虚拟环境
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. 创建requirements.txt
```bash
echo "fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.2
python-multipart>=0.0.6
opencv-python-headless>=4.8.0.74
numpy>=1.24.0
pillow>=10.0.0" > requirements.txt
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 创建模型配置文件
```bash
echo '{
  "models": [
    {
      "id": "mobilenet_v2_tsr",
      "name": "MobileNetV2 (轻量级)",
      "type": "mobilenet",
      "description": "轻量级模型，适合资源受限场景",
      "path": "models/mobilenet/model.h5"
    },
    {
      "id": "yolov5s_tsr",
      "name": "YOLOv5s (通用)",
      "type": "yolo",
      "description": "通用对象检测模型，精度与速度平衡",
      "path": "models/yolo/model.pt"
    }
  ]
}' > backend/models/config.json
```

### 3. 设置中间件(Node.js Express)

1. 初始化Node.js项目
```bash
cd middleware
npm init -y
```

2. 安装依赖
```bash
npm install express cors http-proxy-middleware dotenv
npm install --save-dev typescript ts-node @types/express @types/node @types/cors nodemon
```

3. 创建TypeScript配置文件
```bash
npx tsc --init
```

4. 创建环境配置文件
```bash
echo "PORT=3001
BACKEND_URL=http://localhost:8000" > .env
```

### 4. 设置前端(React + Vite)

1. 使用Vite创建React项目
```bash
cd frontend
npm create vite@latest . -- --template react-ts
```

2. 安装核心依赖
```bash
npm install
npm install axios
```

3. 安装Shadcn UI
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install @radix-ui/react-slot lucide-react class-variance-authority clsx tailwind-merge
```

## 阶段二：后端API开发

### 1. 创建FastAPI应用主入口
创建文件: backend/app/main.py
```python
# 实现FastAPI应用入口，设置CORS，导入路由
```

### 2. 实现核心配置模块
创建文件: backend/app/core/config.py
```python
# 实现应用配置，读取环境变量等
```

### 3. 实现模型管理器
创建文件: backend/app/models/model_manager.py
```python
# 实现模型配置加载和模型管理逻辑
```

### 4. 实现MobileNet模型占位符
创建文件: backend/app/models/mobilenet/model.py
```python
# 实现MobileNet模型加载和推理占位符
```

### 5. 实现YOLO模型占位符
创建文件: backend/app/models/yolo/model.py
```python
# 实现YOLO模型加载和推理占位符
```

### 6. 实现图像处理工具
创建文件: backend/app/utils/image.py
```python
# 实现图像处理函数，如读取、缩放等
```

### 7. 实现推理工具
创建文件: backend/app/utils/inference.py
```python
# 实现通用推理助手函数
```

### 8. 创建API数据模型
创建文件: backend/app/schemas/model.py, inference.py, training.py
```python
# 实现各API的Pydantic模型
```

### 9. 实现模型管理API
创建文件: backend/app/api/routes/models.py
```python
# 实现模型查询和管理API
```

### 10. 实现推理API
创建文件: backend/app/api/routes/inference.py
```python
# 实现图像推理API
```

### 11. 实现指标API
创建文件: backend/app/api/routes/metrics.py
```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional

from app.models import model_manager

# 创建路由
router = APIRouter()

# 模拟数据 - 存储模型性能指标
model_metrics = {
    "mobilenet_v2_tsr": {
        "accuracy": 0.92,
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "inference_time": 45  # ms
    },
    "yolov5s_tsr": {
        "accuracy": 0.89,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "inference_time": 75  # ms
    }
}

@router.get("")
async def get_all_models_metrics():
    """获取所有模型的性能指标"""
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
    """获取特定模型的性能指标"""
    # 验证模型存在
    model_info = next((m for m in model_manager.get_available_models() 
                      if m["id"] == model_id), None)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"未找到ID为'{model_id}'的模型")
    
    metrics = model_metrics.get(model_id, {})
    
    return {
        "model_id": model_id,
        "model_name": model_info["name"],
        "metrics": metrics
    }
```

### 12. 实现训练API
创建文件: backend/app/api/routes/training.py
```python
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
        time.sleep(0.5)
        
        # 检查是否要求停止训练
        if training_jobs[job_id].get("should_stop", False):
            training_jobs[job_id]["status"] = "stopped"
            return
    
    # 训练完成
    training_jobs[job_id]["status"] = "completed"
    training_jobs[job_id]["progress"] = 100.0

@router.post("", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """启动模型训练任务"""
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
            "batch_size": request.batch_size
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
    """获取训练任务状态"""
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
```

## 阶段三：中间件开发

### 1. 实现Express服务器
创建文件: middleware/src/server.ts
```typescript
// 实现Express服务器入口
```

### 2. 实现代理配置
创建文件: middleware/src/proxy.ts
```typescript
// 实现API代理配置
```

## 阶段四：前端开发

### 1. 配置Tailwind CSS
更新文件: frontend/tailwind.config.js
```javascript
// 配置Tailwind和Shadcn UI
```

### 2. 实现Shadcn UI组件
```bash
# 创建组件目录
mkdir -p frontend/src/components/ui
```

### 3. 定义TypeScript类型
创建文件: frontend/src/types/index.ts
```typescript
// 定义API响应类型等
```

### 4. 实现API服务
创建文件: frontend/src/services/api.ts
```typescript
// 实现API请求封装
```

### 5. 实现页眉组件
创建文件: frontend/src/components/Header.tsx
```typescript
// 实现页眉组件
```

### 6. 实现模型选择组件
创建文件: frontend/src/components/ModelSelector.tsx
```typescript
// 实现模型选择组件
```

### 7. 实现媒体上传组件
创建文件: frontend/src/components/MediaUploader.tsx
```typescript
// 实现图片/视频上传组件
```

### 8. 实现结果显示组件
创建文件: frontend/src/components/ResultDisplay.tsx
```typescript
// 实现识别结果显示组件
```

### 9. 实现指标显示组件
创建文件: frontend/src/components/MetricsDisplay.tsx
```typescript
// 实现模型指标显示组件
```

### 10. 实现训练面板组件
创建文件: frontend/src/components/TrainingPanel.tsx
```typescript
// 实现训练控制面板组件
```

### 11. 实现主应用组件
更新文件: frontend/src/App.tsx
```typescript
// 整合所有组件
```

## 阶段五：集成与测试

### 1. 启动后端服务
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 2. 启动中间件服务
```bash
cd middleware
npm run dev
```

### 3. 启动前端服务
```bash
cd frontend
npm run dev
```

### 4. 测试完整工作流程
- 测试模型列表获取
- 测试图片上传和推理
- 测试模型指标展示
- 测试训练功能占位符

## 实施顺序清单（按优先级）

```
实施清单：
1. 创建项目目录结构
2. 设置后端基础框架
   1. 创建Python虚拟环境
   2. 安装依赖
   3. 创建模型配置文件
3. 实现后端核心功能
   1. 创建FastAPI应用入口
   2. 实现模型管理器
   3. 实现模型占位符类
   4. 实现图像处理工具
   5. 实现API数据模型
   6. 实现各API路由
4. 设置中间件框架
   1. 初始化Node.js项目
   2. 安装依赖
   3. 实现Express服务器和代理
5. 设置前端基础框架
   1. 创建React项目
   2. 安装依赖
   3. 配置Tailwind CSS
6. 实现前端核心功能
   1. 定义TypeScript类型
   2. 实现API服务
   3. 实现各UI组件
   4. 整合应用
7. 系统集成与测试
   1. 启动各服务
   2. 测试完整流程
``` 