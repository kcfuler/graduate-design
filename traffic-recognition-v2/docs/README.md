# 交通标志识别系统

本项目是一个深度学习交通标志识别系统，使用三层架构实现。系统允许用户上传图片或视频，选择AI模型进行交通标志识别，并查看识别结果和模型性能指标。

## 系统架构

系统由三个主要组件组成：
1. **前端 (Frontend)**: React + TypeScript + Vite
2. **中间层 (Middleware)**: Node.js + Express
3. **后端 (Backend)**: Python + FastAPI

## 环境要求

### 开发环境
- Node.js >= 14.0.0
- Python >= 3.8
- npm >= 6.0.0
- pip >= 20.0.0

### 生产环境
- 上述所有依赖
- Docker (可选，用于容器化部署)
- Nginx (可选，用于生产环境的反向代理)

## 开发环境部署

### 1. 后端部署

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动后端服务
uvicorn app.main:app --reload --port 8000
```

后端服务将在 http://localhost:8000 运行，API文档可通过 http://localhost:8000/docs 访问。

### 2. 中间层部署

```bash
# 进入中间层目录
cd middleware

# 安装依赖
npm install

# 创建或修改.env文件
echo "PORT=3001\nBACKEND_URL=http://localhost:8000" > .env

# 启动中间层服务
npm run dev
```

中间层服务将在 http://localhost:3001 运行。

### 3. 前端部署

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端服务将在 http://localhost:5173 运行。

### 本地开发访问流程

在开发环境中，请求流如下：
1. 浏览器访问 http://localhost:5173
2. 前端发起API请求到 http://localhost:5173/api/* (由Vite开发服务器代理)
3. 请求转发到中间层 http://localhost:3001/api/*
4. 中间层转发请求到后端 http://localhost:8000/*
5. 后端处理请求并返回响应

## 生产环境部署

### 方案1: 单体部署

这种方案中，中间层同时负责托管前端静态文件和代理后端请求。

```bash
# 构建前端
cd frontend
npm run build

# 部署中间层，配置为生产环境
cd ../middleware
export NODE_ENV=production
# 或者在Windows上:
# set NODE_ENV=production

# 启动服务
npm start
```

在这种配置下，中间层将：
1. 在 http://localhost:3001 上托管前端静态文件
2. 处理API请求并转发到后端

### 方案2: Docker Compose部署

创建Docker Compose配置，同时启动所有三个服务。详细步骤请参考项目中的Docker相关文件。

系统将在以下端口运行：
- 前端: http://localhost
- 中间层: http://localhost:3001
- 后端: http://localhost:8000

### 方案3: 使用Nginx作为前端反向代理

在这种方案中，使用Nginx作为前端和API请求的入口点。详细的Nginx配置示例请参考项目文档。

## 常见问题

### 跨域问题

在开发环境中，跨域请求通过以下方式处理：
1. 前端Vite服务器配置代理
2. 中间层配置CORS头
3. 后端FastAPI启用CORS中间件

在生产环境中，如果使用单一域名部署，则跨域问题自然解决。

### 大文件上传限制

如果需要上传大型视频文件，需要调整配置：
- Nginx: 设置 `client_max_body_size`
- Express中间层: 设置 `limit` 参数
- FastAPI后端: 配置 CORS 中间件

## 项目文档

本项目包括以下详细文档：

- [解决方案设计](solution-design.md) - 技术方案、项目结构和实现细节
- [任务清单](task-checklist.md) - 项目任务和执行状态追踪
- [API参考](api-reference.md) - API接口详细说明

这些文档用于:
1. 项目规划和跟踪
2. 开发参考指南
3. 技术决策记录
4. 项目状态监控

请根据需要更新这些文档，确保它们反映当前项目状态。 