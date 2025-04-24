# 深度学习交通标志识别系统 - 执行清单

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

- [ ] 23. 使用Vite创建React项目
  ```bash
  cd frontend
  npm create vite@latest . -- --template react-ts
  ```

- [ ] 24. 安装React依赖
  ```bash
  npm install
  npm install axios
  ```

- [ ] 25. 配置Tailwind CSS
  ```bash
  npm install -D tailwindcss postcss autoprefixer
  npx tailwindcss init -p
  ```

- [ ] 26. 安装Shadcn UI依赖
  ```bash
  npm install @radix-ui/react-slot lucide-react class-variance-authority clsx tailwind-merge
  ```

- [ ] 27. 配置Tailwind CSS (frontend/tailwind.config.js)

- [ ] 28. 更新CSS入口文件 (frontend/src/index.css)

- [ ] 29. 定义TypeScript类型 (frontend/src/types/index.ts)

- [ ] 30. 实现API服务 (frontend/src/services/api.ts)

- [ ] 31. 创建基础UI组件:
  ```bash
  mkdir -p frontend/src/components/ui
  ```

- [ ] 32. 实现自定义组件:
  - [ ] frontend/src/components/Header.tsx
  - [ ] frontend/src/components/ModelSelector.tsx
  - [ ] frontend/src/components/MediaUploader.tsx
  - [ ] frontend/src/components/ResultDisplay.tsx
  - [ ] frontend/src/components/MetricsDisplay.tsx
  - [ ] frontend/src/components/TrainingPanel.tsx

- [ ] 33. 更新主应用组件 (frontend/src/App.tsx)

- [ ] 34. 配置环境变量 (frontend/.env)
  ```
  VITE_API_BASE_URL=http://localhost:3001/api
  ```

## 系统集成与测试

- [ ] 35. 启动后端服务
  ```bash
  cd backend
  uvicorn app.main:app --reload --port 8000
  ```

- [ ] 36. 启动中间件服务
  ```bash
  cd middleware
  npm run dev
  ```

- [ ] 37. 启动前端服务
  ```bash
  cd frontend
  npm run dev
  ```

- [ ] 38. 验证系统集成:
  - [ ] 测试模型列表获取
  - [ ] 测试图片上传和推理
  - [ ] 测试模型指标查询
  - [ ] 测试训练功能占位符

- [ ] 39. 优化和修复问题