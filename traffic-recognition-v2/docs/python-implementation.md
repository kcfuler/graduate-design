# Python 后端实现文档

## 概述

本文档记录了深度学习交通标志识别系统后端 API 的实现。后端使用 FastAPI 框架构建，提供了模型管理、图像推理、模型指标和训练功能的 RESTful API。

## 项目结构

后端项目结构如下：

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py    # 推理 API
│   │   │   ├── metrics.py      # 指标 API
│   │   │   ├── models.py       # 模型管理 API
│   │   │   └── training.py     # 训练 API
│   │   └── __init__.py
│   ├── core/
│   │   └── config.py           # 应用配置
│   ├── models/
│   │   ├── mobilenet/
│   │   │   ├── __init__.py
│   │   │   └── model.py        # MobileNet 模型实现
│   │   ├── yolo/
│   │   │   ├── __init__.py
│   │   │   └── model.py        # YOLO 模型实现
│   │   ├── __init__.py
│   │   └── model_manager.py    # 模型管理器
│   ├── schemas/
│   │   ├── inference.py        # 推理数据模型
│   │   ├── model.py            # 模型数据模型
│   │   └── training.py         # 训练数据模型
│   ├── utils/
│   │   ├── image.py            # 图像处理工具
│   │   └── inference.py        # 推理工具
│   └── main.py                 # 应用入口
└── models/
    └── config.json             # 模型配置文件
```

## API 路由

### 1. 模型管理 API (`/models`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/models` | GET | 获取所有可用模型 |
| `/models/active` | GET | 获取当前活动模型 |
| `/models/active` | POST | 设置当前活动模型 |

### 2. 推理 API (`/infer`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/infer` | POST | 使用指定模型对图像进行推理 |
| `/infer/base64` | POST | 使用 Base64 编码的图像进行推理 |

### 3. 指标 API (`/metrics`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/metrics` | GET | 获取所有模型的性能指标 |
| `/metrics/{model_id}` | GET | 获取特定模型的性能指标 |
| `/metrics/{model_id}/evaluate` | POST | 触发模型评估（模拟实现） |

### 4. 训练 API (`/train`)

| 端点 | 方法 | 描述 |
|------|------|------|
| `/train` | GET | 获取所有训练任务列表 |
| `/train` | POST | 启动模型训练任务 |
| `/train/{job_id}` | GET | 获取训练任务状态 |
| `/train/{job_id}` | DELETE | 停止训练任务 |

## 实现细节

### 1. 指标 API 实现

指标 API 提供模型性能指标的访问。在当前实现中，使用模拟数据来演示功能。主要端点包括：

- 获取所有模型指标：提供系统中所有模型的性能数据
- 获取特定模型指标：提供特定模型的详细性能数据
- 模型评估：模拟触发模型评估过程

指标数据包括准确率、精确率、召回率、F1分数和推理时间等。

```python
# 模拟数据示例
model_metrics = {
    "mobilenet_v2_tsr": {
        "accuracy": 0.92,
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "inference_time": 45,  # ms
        "last_updated": "2023-11-15"
    }
}
```

### 2. 训练 API 实现

训练 API 提供模型训练功能的接口。在当前实现中，使用后台任务模拟训练过程。主要端点包括：

- 获取训练任务列表：列出所有训练任务
- 启动训练任务：创建并启动新的训练任务
- 获取训练状态：查询特定训练任务的状态和进度
- 停止训练任务：停止正在进行的训练任务

训练功能使用 FastAPI 的后台任务特性来模拟异步训练过程，任务状态和指标会随时间更新。

```python
# 训练任务模拟示例
def mock_training_task(job_id: str, model_id: str, epochs: int):
    for epoch in range(epochs):
        progress = (epoch + 1) / epochs * 100
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
        
        # 模拟训练时间
        time.sleep(0.5)
```

## 注意事项

- 当前实现主要用于演示目的，使用模拟数据和占位符逻辑
- 实际部署时，应替换模拟实现为真实的模型训练和评估代码
- 系统设计支持未来扩展，如添加新的模型类型、训练方法等 