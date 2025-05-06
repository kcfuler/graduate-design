# 交通标志识别系统 - 前端开发指南

本文档总结了交通标志识别系统中与前端开发相关的关键信息，包括项目架构、技术栈、组件设计和API接口。

## 项目概述

交通标志识别系统是一个深度学习应用，允许用户上传图片或视频，选择AI模型进行交通标志识别，并查看识别结果和模型性能指标。系统采用三层架构实现，前端使用React+TypeScript构建用户界面。

## 技术栈

- **UI框架**: React + TypeScript
- **构建工具**: rsbuild
- **包管理工具**: bun
- **CSS方案**: Tailwind CSS
- **组件库**: antd
- **react hooks**: ahooks
- **代码质量**: oxlint

## 前端目录结构

```
frontend/
├── public/                 # 静态资源
├── src/
│   ├── components/         # UI组件
│   │   ├── Header.tsx      # 页眉组件
│   │   ├── ModelSelector.tsx  # 模型选择组件
│   │   ├── MediaUploader.tsx  # 媒体上传组件
│   │   ├── ResultDisplay.tsx  # 结果显示组件
│   │   ├── MetricsDisplay.tsx # 指标显示组件
│   │   ├── InferenceLayout.tsx # 推理主界面组件
│   ├── services/           # API服务
│   │   └── api.ts          # API请求封装
│   ├── hooks/              # 自定义钩子
│   ├── types/              # TypeScript类型定义
│   ├── assets/             # 本地资源
│   ├── App.tsx             # 主应用组件
│   ├── main.tsx            # 入口文件
│   └── index.css           # 全局样式
├── package.json            # 依赖配置
├── tsconfig.json           # TypeScript配置
└── rsbuild.config.ts       # rsbuild配置
```

## 主界面布局设计

主界面采用固定顶部Header和2×2网格布局设计，优化用户体验和交互流程。

### 布局结构

1. **顶部固定Header**
   - 内容为"Traffic Sign Recognition"
   - 高度约60px，浅灰背景色，文字居中
   - 使用Tailwind CSS实现固定定位和样式

2. **主区域2×2网格布局**
   - 采用CSS Grid实现，占满剩余高度
   - 四个区域有明确的grid-area命名
   - 各区域之间保持16px间距
   - 所有容器采用8px圆角，轻微阴影

### 功能区域详细设计

#### 推理结果区域 (grid-area: infer-result)
- 位于左上格
- 展示推理后的带标注图片
- 图片容器等比例缩放，保持白色边框和圆角
- 无图片时显示占位图或提示文字
- 图片加载中显示骨架屏或加载动画

#### 性能指标区域 (grid-area: perf-metrics)
- 位于右上格
- 卡片式容器，展示模型性能数据
- 包含关键指标：
  - FPS (每秒帧数)
  - mAP@0.5 (平均精度)
  - 推理时间 (毫秒)
- 每项指标采用标题+数值/进度条形式
- 支持指标数据动态更新

#### 上传区域 (grid-area: upload)
- 位于左下格
- 大尺寸按钮，文字"上传图片"
- 支持两种上传方式：
  - 点击触发文件选择对话框
  - 拖拽文件到区域内释放
- 拖拽状态有明显视觉反馈：边框高亮，显示"松手上传"提示
- 上传完成后自动预览并更新状态

#### 运行按钮区域 (grid-area: run-button)
- 位于右下格
- 醒目的"Run Inference"按钮
- 点击后调用onRun()回调函数
- 处理中状态显示加载动画
- 禁用状态有明确视觉提示(当无文件上传时)

### 状态管理

- 使用React状态管理文件上传和处理：
  ```typescript
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  ```

- 文件上传后自动生成预览
- 推理完成后返回JSON结果，用于绘制边框和标签
- 各组件间状态同步，保持UI一致性


## 关键组件设计

### 模型选择界面

结合下拉选择器和详情卡片两种方案：
- 主界面使用下拉选择器，简单直观

### 媒体上传组件

支持多种媒体输入方式：
1. **本地文件上传**: 使用`<input type="file">`结合拖放功能
2. **URL输入**: 支持直接输入图片/视频URL
3. **摄像头捕获**: 可选功能，使用WebRTC

### 结果展示

1. **图像结果**: 在原图上叠加边界框，显示标签和置信度
2. **视频结果**: 采用离线处理方式，上传视频后后台处理，完成后展示结果

## API接口

前端通过中间层与后端交互，所有API请求通过`/api/*`路由进行。主要API端点如下：

### 模型管理

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/models` | GET | 获取所有可用模型 |
| `/api/models/active` | GET | 获取当前活动模型 |
| `/api/models/active` | POST | 设置当前活动模型 |

### 推理服务

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/infer` | POST | 使用图像文件或URL进行推理 |
| `/api/infer/video` | POST | 使用视频文件进行推理 |

## 开发流程

### 开发环境部署

```bash
# 进入前端目录
cd frontend

# 安装依赖
bun install

# 启动开发服务器
bun run dev
```

前端服务将在 http://localhost:5173 运行。

### API请求流程

在开发环境中，请求流如下：
1. 浏览器访问 http://localhost:5173
2. 前端发起API请求到 http://localhost:5173/api/* (由Vite开发服务器代理)
3. 请求转发到中间层 http://localhost:3001/api/*
4. 中间层转发请求到后端 http://localhost:8000/*
5. 后端处理请求并返回响应

### 错误处理

所有错误响应遵循一致的格式：

```json
{
  "error": "错误类型",
  "message": "详细错误消息",
  "code": "错误代码"
}
```

常见错误代码：

| 错误代码 | 描述 | HTTP状态码 |
|----------|------|------------|
| `VALIDATION_ERROR` | 请求参数验证失败 | 400 |
| `MODEL_NOT_FOUND` | 指定的模型不存在 | 404 |
| `PROCESSING_ERROR` | 处理图像/视频时出错 | 500 |
| `NO_FILE_UPLOADED` | 没有上传文件 | 400 |
| `INVALID_IMAGE_FORMAT` | 图像格式无效 | 400 |
| `BACKEND_UNAVAILABLE` | 后端服务不可用 | 503 |
| `BACKEND_TIMEOUT` | 后端服务响应超时 | 504 |

## 生产环境部署

前端应用有多种部署选项：

1. **开发部署**: 独立运行前端、中间层和后端服务
2. **单体部署**: 中间层托管前端静态文件，代理API请求
3. **Docker部署**: 使用Docker Compose配置，同时启动所有三个服务
4. **Nginx部署**: 使用Nginx作为前端和API请求的入口点

在单体部署模式下，需要构建前端资源并将其放在中间层可访问的位置：

```bash
# 构建前端
cd frontend
bun build
```

中间层将在生产环境中托管这些静态文件。 