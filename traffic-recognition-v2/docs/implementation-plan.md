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
npm install --save-dev nodemon
```

3. 创建环境配置文件
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
# 实现模型性能指标API
```

### 12. 实现训练API占位符
创建文件: backend/app/api/routes/training.py
```python
# 实现训练API占位符
```

## 阶段三：中间件开发

### 1. 实现Express服务器
创建文件: middleware/src/server.js
```javascript
// 实现Express服务器入口
```

### 2. 实现代理配置
创建文件: middleware/src/proxy.js
```javascript
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