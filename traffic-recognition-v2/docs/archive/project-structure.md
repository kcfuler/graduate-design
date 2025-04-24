# 深度学习交通标志识别系统 - 项目结构

## 整体目录结构
```
traffic-recognition-v2/
├── frontend/               # React前端应用
├── middleware/             # Node.js中间件
└── backend/                # Python FastAPI后端
```

## 前端结构 (React + Shadcn UI + Vite)
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
│   │   └── TrainingPanel.tsx  # 训练面板组件
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
└── vite.config.ts          # Vite配置
```

## 中间件结构 (Node.js + Express + TypeScript)
```
middleware/
├── src/
│   ├── server.ts           # 服务器入口
│   ├── proxy.ts            # 代理配置
│   └── utils/              # 工具函数
├── package.json            # 依赖配置
├── tsconfig.json           # TypeScript配置
└── .env                    # 环境变量
```

## 后端结构 (Python + FastAPI)
```
backend/
├── app/
│   ├── main.py             # FastAPI应用入口
│   ├── api/
│   │   ├── routes/
│   │   │   ├── models.py   # 模型管理API
│   │   │   ├── inference.py # 推理API
│   │   │   ├── metrics.py  # 指标API
│   │   │   └── training.py # 训练API
│   │   └── dependencies.py # API依赖项
│   ├── core/
│   │   ├── config.py       # 应用配置
│   │   ├── security.py     # 安全相关
│   │   └── logging.py      # 日志配置
│   ├── models/
│   │   ├── model_manager.py # 模型管理器
│   │   ├── mobilenet/      # MobileNet相关代码
│   │   └── yolo/           # YOLO相关代码
│   ├── schemas/            # Pydantic模型
│   │   ├── model.py        # 模型相关schema
│   │   ├── inference.py    # 推理相关schema
│   │   └── training.py     # 训练相关schema
│   └── utils/              # 工具函数
│       ├── image.py        # 图像处理
│       └── inference.py    # 推理助手
├── models/                 # 模型文件存储
│   ├── config.json         # 模型配置
│   ├── mobilenet/          # MobileNet模型文件
│   └── yolo/               # YOLO模型文件
├── requirements.txt        # Python依赖
└── Dockerfile              # 容器化配置
```

## 数据流设计
1. 用户在前端上传图片/视频并选择模型
2. 前端将请求发送到中间件的API端点
3. 中间件将请求代理到Python后端
4. 后端加载指定模型，处理图像，执行推理
5. 推理结果通过相同路径返回前端
6. 前端展示识别结果（边界框、标签） 