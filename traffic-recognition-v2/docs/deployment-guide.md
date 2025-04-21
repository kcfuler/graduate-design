# 交通标志识别系统 - 部署指南

本文档提供了交通标志识别系统的完整部署指南，包括开发环境和生产环境的部署步骤。

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

创建Docker Compose配置，同时启动所有三个服务。

#### 1. 创建docker-compose.yml

```yaml
version: '3'

services:
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - middleware
    networks:
      - app-network

  middleware:
    build: ./middleware
    environment:
      - NODE_ENV=production
      - PORT=3001
      - BACKEND_URL=http://backend:8000
    ports:
      - "3001:3001"
    depends_on:
      - backend
    networks:
      - app-network

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

#### 2. 为每个组件创建Dockerfile

##### 前端 Dockerfile
```dockerfile
FROM node:14 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

##### 中间层 Dockerfile
```dockerfile
FROM node:14
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3001
CMD ["npm", "start"]
```

##### 后端 Dockerfile
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3. 启动Docker Compose

```bash
docker-compose up -d
```

系统将在以下端口运行：
- 前端: http://localhost
- 中间层: http://localhost:3001
- 后端: http://localhost:8000

### 方案3: 使用Nginx作为前端反向代理

在这种方案中，使用Nginx作为前端和API请求的入口点。

#### Nginx配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # 前端静态文件
    location / {
        root /path/to/frontend/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    # API请求代理到中间层
    location /api/ {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 常见问题

### 跨域问题

在开发环境中，跨域请求通过以下方式处理：
1. 前端Vite服务器配置代理
2. 中间层配置CORS头
3. 后端FastAPI启用CORS中间件

在生产环境中，如果使用单一域名部署，则跨域问题自然解决。

### 大文件上传限制

如果需要上传大型视频文件，需要调整以下配置：

1. Nginx (如果使用):
```nginx
client_max_body_size 100M;
```

2. Express中间层:
```javascript
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
```

3. FastAPI后端:
```python
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)
```

## 监控与日志

### 开发环境
在开发环境中，各组件的日志直接输出到控制台。

### 生产环境
在生产环境中，推荐配置以下日志方案：

1. 使用PM2管理Node.js服务
```bash
npm install -g pm2
pm2 start middleware/src/index.js
```

2. 使用systemd管理Python服务
```
[Unit]
Description=Traffic Recognition Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/path/to/backend
ExecStart=/path/to/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 安全考虑

1. 在生产环境中，建议:
   - 限制CORS为特定域名
   - 添加请求速率限制
   - 实施适当的请求验证
   - 配置HTTPS

2. API访问控制:
   - 考虑添加API密钥或认证
   - 限制敏感操作的访问

## 性能优化

1. 启用压缩:
```javascript
// 中间层
const compression = require('compression');
app.use(compression());
```

2. 缓存策略:
```nginx
# Nginx
location ~* \.(js|css|png|jpg|jpeg|gif|ico)$ {
    expires 30d;
    add_header Cache-Control "public, no-transform";
}
```

3. 使用CDN托管静态资源 (生产环境) 