# 交通标志识别系统 - API参考文档

本文档详细描述了交通标志识别系统的API接口。前端应用通过中间层（运行在`/api`前缀下）与后端交互。

## 基本信息

- 基础URL: `/api`
- 内容类型: 请求和响应主体均为JSON，除非另有说明（如文件上传）
- 错误响应格式:
  ```json
  {
    "error": "错误类型",
    "message": "错误详细描述",
    "code": "错误码"
  }
  ```

## 健康检查

### 检查API状态

```
GET /
```

#### 响应

```json
{
  "message": "交通标志识别API在线"
}
```

## 模型管理

### 获取所有可用模型

```
GET /models
```

#### 响应

```json
{
  "models": [
    {
      "id": "mobilenet_v2_tsr",
      "name": "MobileNet V2 TSR",
      "description": "轻量级交通标志分类模型",
      "type": "classification",
      "version": "1.0.0"
    },
    {
      "id": "yolov5s_tsr",
      "name": "YOLOv5s TSR",
      "description": "实时交通标志检测模型",
      "type": "detection",
      "version": "1.0.0"
    }
  ]
}
```

### 获取当前活动模型

```
GET /models/active
```

#### 响应

```json
{
  "active_model": {
    "id": "yolov5s_tsr",
    "name": "YOLOv5s TSR",
    "description": "实时交通标志检测模型",
    "type": "detection",
    "version": "1.0.0"
  }
}
```

### 设置当前活动模型

```
POST /models/active
```

#### 请求体

```json
{
  "model_id": "mobilenet_v2_tsr"
}
```

#### 响应

```json
{
  "message": "模型 'mobilenet_v2_tsr' 已设置为当前活动模型"
}
```

## 模型指标

### 获取所有模型的性能指标

```
GET /metrics
```

#### 响应

```json
{
  "models_metrics": [
    {
      "model_id": "mobilenet_v2_tsr",
      "model_name": "MobileNet V2 TSR",
      "metrics": {
        "accuracy": 0.92,
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "inference_time": 45,
        "last_updated": "2023-11-15"
      }
    },
    {
      "model_id": "yolov5s_tsr",
      "model_name": "YOLOv5s TSR",
      "metrics": {
        "accuracy": 0.89,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "inference_time": 75,
        "mAP50": 0.87,
        "last_updated": "2023-11-10"
      }
    }
  ]
}
```

### 获取特定模型的性能指标

```
GET /metrics/{model_id}
```

#### 响应

```json
{
  "model_id": "mobilenet_v2_tsr",
  "model_name": "MobileNet V2 TSR",
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.94,
    "recall": 0.91,
    "f1_score": 0.925,
    "inference_time": 45,
    "last_updated": "2023-11-15"
  }
}
```

### 触发模型评估

```
POST /metrics/{model_id}/evaluate
```

#### 响应

```json
{
  "status": "success",
  "message": "已为模型 '{model_id}' 触发评估"
}
```

## 推理服务

### 使用图像文件进行推理

```
POST /infer
```

#### 请求体 (multipart/form-data)

- `model_id`: 要使用的模型ID
- `image`: 图像文件
- `draw_boxes`: 是否在结果图像上绘制边界框 (可选，默认为`false`)

#### 响应

```json
{
  "model_id": "yolov5s_tsr",
  "predictions": [
    {
      "label": "限速30",
      "confidence": 0.95,
      "bounding_box": {
        "x1": 100,
        "y1": 120,
        "x2": 200,
        "y2": 220
      }
    },
    {
      "label": "禁止通行",
      "confidence": 0.88,
      "bounding_box": {
        "x1": 300,
        "y1": 150,
        "x2": 400,
        "y2": 250
      }
    }
  ],
  "inference_time": 75,
  "image_result": "data:image/jpeg;base64,..."  // 仅当draw_boxes=true时返回
}
```

### 使用图像URL进行推理

```
POST /infer
```

#### 请求体 (multipart/form-data)

- `model_id`: 要使用的模型ID
- `image_url`: 图像URL
- `draw_boxes`: 是否在结果图像上绘制边界框 (可选，默认为`false`)

#### 响应

与图像文件推理相同。

### 使用Base64编码图像进行推理

```
POST /infer/base64
```

#### 请求体

```json
{
  "model_id": "yolov5s_tsr",
  "image_base64": "data:image/jpeg;base64,...",
  "draw_boxes": true
}
```

#### 响应

与图像文件推理相同。

### 使用视频文件进行推理

```
POST /infer/video
```

#### 请求体 (multipart/form-data)

- `model_id`: 要使用的模型ID
- `video`: 视频文件
- `draw_boxes`: 是否在结果视频上绘制边界框 (可选，默认为`true`)

#### 响应

```json
{
  "model_id": "yolov5s_tsr",
  "frames_processed": 120,
  "total_detections": 45,
  "processing_time": 5600,
  "video_result_url": "/api/results/video/12345.mp4",  // 处理后的视频URL
  "detections": [
    {
      "frame": 10,
      "predictions": [
        {
          "label": "限速30",
          "confidence": 0.95,
          "bounding_box": {
            "x1": 100,
            "y1": 120,
            "x2": 200,
            "y2": 220
          }
        }
      ]
    },
    // 更多帧的检测结果...
  ]
}
```

## 错误代码

| 错误代码                | 描述                    | HTTP状态码 |
|------------------------|-------------------------|------------|
| `VALIDATION_ERROR`     | 请求参数验证失败        | 400        |
| `MODEL_NOT_FOUND`      | 指定的模型不存在        | 404        |
| `PROCESSING_ERROR`     | 处理图像/视频时出错     | 500        |
| `NO_FILE_UPLOADED`     | 没有上传文件            | 400        |
| `INVALID_IMAGE_FORMAT` | 图像格式无效            | 400        |
| `BACKEND_UNAVAILABLE`  | 后端服务不可用          | 503        |
| `BACKEND_TIMEOUT`      | 后端服务响应超时        | 504        |

## 中间层健康检查

### 检查中间层状态

```
GET /health
```

#### 响应

```json
{
  "status": "OK",
  "message": "中间件服务运行正常"
}
``` 