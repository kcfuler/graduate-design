# 深度学习交通标志识别系统 - 任务清单

## 项目概述
创建一个全栈Web应用程序，使用深度学习技术识别图片或视频中的交通标志。系统允许用户上传媒体，选择模型，查看识别结果和模型指标。

## 技术栈
- **前端**: React + Shadcn UI + Vite + TypeScript
- **中间件**: Node.js + Express + TypeScript
- **后端**: Python + FastAPI

## 系统架构
- React前端处理用户交互和结果展示
- Node.js中间件(使用TypeScript)作为API网关，转发请求到Python后端
- Python后端管理AI模型和执行推理逻辑

## 项目基础结构

- [x] 1. 创建项目顶级目录
  ```bash
  mkdir -p frontend middleware backend
  ```

- [x] 2. 创建后端目录结构
  ```bash
  mkdir -p backend/app/api/routes backend/app/core backend/app/models backend/app/schemas backend/app/utils backend/models/mobilenet backend/models/yolo
  ```

- [x] 3. 创建前端目录结构
  ```bash
  mkdir -p frontend/src/components frontend/src/services frontend/src/hooks frontend/src/types frontend/src/assets
  ```

- [x] 4. 创建中间件目录结构
  ```bash
  mkdir -p middleware/src/utils
  ```

## 后端开发

- [x] 5. 创建Python虚拟环境
  ```bash
  cd backend
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

- [x] 6. 创建requirements.txt文件
  ```
  fastapi>=0.104.0
  uvicorn>=0.24.0
  pydantic>=2.4.2
  python-multipart>=0.0.6
  opencv-python-headless>=4.8.0.74
  numpy>=1.24.0
  pillow>=10.0.0
  ```

- [x] 7. 安装Python依赖
  ```bash
  pip install -r requirements.txt
  ```

- [x] 8. 创建模型配置文件 (backend/models/config.json)

- [x] 9. 创建FastAPI应用入口 (backend/app/main.py)

- [x] 10. 创建配置模块 (backend/app/core/config.py)

- [x] 11. 创建API数据模型:
  - [x] backend/app/schemas/model.py
  - [x] backend/app/schemas/inference.py
  - [x] backend/app/schemas/training.py

- [x] 12. 实现模型管理器 (backend/app/models/model_manager.py)

- [x] 13. 实现模型模块:
  - [x] backend/app/models/__init__.py
  - [x] backend/app/models/mobilenet/__init__.py
  - [x] backend/app/models/mobilenet/model.py
  - [x] backend/app/models/yolo/__init__.py
  - [x] backend/app/models/yolo/model.py

- [x] 14. 实现工具函数:
  - [x] backend/app/utils/image.py
  - [x] backend/app/utils/inference.py

- [x] 15. 实现API路由:
  - [x] backend/app/api/__init__.py
  - [x] backend/app/api/routes/__init__.py
  - [x] backend/app/api/routes/models.py
  - [x] backend/app/api/routes/inference.py
  - [x] backend/app/api/routes/metrics.py
  - [x] backend/app/api/routes/training.py
  
## 中间件开发

- [x] 16. 初始化Node.js项目
  ```bash
  cd middleware
  npm init -y
  ```

- [x] 17. 安装Node.js依赖
  ```bash
  npm install express cors http-proxy-middleware dotenv
  npm install --save-dev typescript ts-node @types/express @types/node @types/cors nodemon
  ```

- [x] 18. 创建TypeScript配置文件
  ```bash
  npx tsc --init
  ```

- [x] 19. 创建.env文件 (middleware/.env)
  ```
  PORT=3001
  BACKEND_URL=http://localhost:8000
  ```

- [x] 20. 创建package.json脚本 (middleware/package.json)
  ```json
  "scripts": {
    "start": "node dist/server.js",
    "build": "tsc",
    "dev": "ts-node-dev src/server.ts"
  }
  ```

- [x] 21. 实现Express服务器 (middleware/src/server.ts)

- [x] 22. 实现API代理 (middleware/src/proxy.ts)

## 前端开发

- [x] 23. 使用Vite创建React项目
  ```bash
  cd frontend
  npm create vite@latest . -- --template react-ts
  ```

- [x] 24. 安装React依赖
  ```bash
  npm install
  npm install axios
  ```

- [x] 25. 安装并配置Tailwind CSS (已按 v4 要求安装 CLI)
  ```bash
  # 原命令: npm install -D tailwindcss postcss autoprefixer && npx tailwindcss init -p
  ```

- [x] 26. 安装Shadcn UI依赖
  ```bash
  npm install @radix-ui/react-slot lucide-react class-variance-authority clsx tailwind-merge
  ```

- [x] 27. 配置Tailwind CSS (frontend/tailwind.config.js) (手动创建)

- [x] 28. 更新CSS入口文件 (frontend/src/index.css) (使用 v4 @import)

- [x] 29. 定义TypeScript类型 (frontend/src/types/index.ts)

- [x] 30. 实现API服务 (frontend/src/services/api.ts)

- [x] 31. 创建基础UI组件:
  ```bash
  mkdir -p frontend/src/components/ui
  ```

- [x] **32. 组件实现 (细化)**:
  - [x] **32.1**: 实现 `Header` 组件：添加基本布局和标题样式。
  - [x] **32.2**: 实现 `ModelSelector` 组件：
    - [x] 调用 `getModels` API 获取模型列表。
    - [x] 使用 Shadcn `Select` 组件展示模型选项。
    - [x] 管理选中的模型状态。
  - [x] **32.3**: 实现 `MediaUploader` 组件：
    - [x] 使用 Shadcn `Input type="file"` 处理图片文件选择。
    - [x] 实现图片文件预览功能。
    - [x] 添加 Shadcn `Button` 触发上传/推理。
    - [x] 管理文件和推理状态。
  - [x] **32.4**: 实现 `ResultDisplay` 组件：
    - [x] 接收推理结果数据 (`InferenceResult | ApiError | null`)。
    - [x] 在图片上使用 CSS 绝对定位绘制边界框（如果适用）。
    - [x] 使用 Shadcn `Card` 展示识别标签和置信度。
  - [x] **32.5**: 实现 `MetricsDisplay` 组件：
    - [x] 调用 `getMetrics` API 获取选定模型的指标。
    - [x] 使用 Shadcn `Card` 展示指标数据 (键值对)。
  - [x] **32.6**: 实现 `TrainingPanel` 组件：
    - [x] 添加触发训练的 Shadcn `Button`。
    - [x] 调用 `startTraining` API (占位符功能)。
    - [x] 显示训练状态或消息。

- [x] 33. 更新主应用组件 (frontend/src/App.tsx): 集成子组件，添加状态管理和 Props 传递。

- [x] 34. 配置环境变量 (frontend/.env) (手动创建)

## 系统集成与测试

- [] **35**: 启动后端服务
  ```bash
  cd backend
  source venv/bin/activate && uvicorn app.main:app --reload --port 8000
  ```

- [] **36**: 启动中间件服务
  ```bash
  cd middleware
  npm run dev
  ```

- [] **37**: 启动前端服务
  ```bash
  cd frontend
  npm run dev
  ```

- [-] **38**: 验证系统集成: (进行中...)
  - [ ] **38.1**: 测试模型列表获取和选择 (`ModelSelector`).
  - [ ] **38.2**: 测试图片上传和推理结果展示 (`MediaUploader`, `ResultDisplay`).
  - [ ] **38.3**: 测试模型指标查询和展示 (`MetricsDisplay`).
  - [ ] **38.4**: 测试训练功能占位符按钮 (`TrainingPanel`).

## 优化与修复

- [ ] **39**: 根据测试结果，优化前端交互、修复 Bug、完善样式。
