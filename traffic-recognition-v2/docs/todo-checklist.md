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
  - [ ] backend/app/api/routes/metrics.py
  - [ ] backend/app/api/routes/training.py
  
## 中间件开发

- [ ] 16. 初始化Node.js项目
  ```bash
  cd middleware
  npm init -y
  ```

- [ ] 17. 安装Node.js依赖
  ```bash
  npm install express cors http-proxy-middleware dotenv
  npm install --save-dev nodemon
  ```

- [ ] 18. 创建.env文件 (middleware/.env)
  ```
  PORT=3001
  BACKEND_URL=http://localhost:8000
  ```

- [ ] 19. 创建package.json脚本 (middleware/package.json)
  ```json
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js"
  }
  ```

- [ ] 20. 实现Express服务器 (middleware/src/server.js)

- [ ] 21. 实现API代理 (middleware/src/proxy.js)

## 前端开发

- [ ] 22. 使用Vite创建React项目
  ```bash
  cd frontend
  npm create vite@latest . -- --template react-ts
  ```

- [ ] 23. 安装React依赖
  ```bash
  npm install
  npm install axios
  ```

- [ ] 24. 配置Tailwind CSS
  ```bash
  npm install -D tailwindcss postcss autoprefixer
  npx tailwindcss init -p
  ```

- [ ] 25. 安装Shadcn UI依赖
  ```bash
  npm install @radix-ui/react-slot lucide-react class-variance-authority clsx tailwind-merge
  ```

- [ ] 26. 配置Tailwind CSS (frontend/tailwind.config.js)

- [ ] 27. 更新CSS入口文件 (frontend/src/index.css)

- [ ] 28. 定义TypeScript类型 (frontend/src/types/index.ts)

- [ ] 29. 实现API服务 (frontend/src/services/api.ts)

- [ ] 30. 创建基础UI组件:
  ```bash
  mkdir -p frontend/src/components/ui
  ```

- [ ] 31. 实现自定义组件:
  - [ ] frontend/src/components/Header.tsx
  - [ ] frontend/src/components/ModelSelector.tsx
  - [ ] frontend/src/components/MediaUploader.tsx
  - [ ] frontend/src/components/ResultDisplay.tsx
  - [ ] frontend/src/components/MetricsDisplay.tsx
  - [ ] frontend/src/components/TrainingPanel.tsx

- [ ] 32. 更新主应用组件 (frontend/src/App.tsx)

- [ ] 33. 配置环境变量 (frontend/.env)
  ```
  VITE_API_BASE_URL=http://localhost:3001/api
  ```

## 系统集成与测试

- [ ] 34. 启动后端服务
  ```bash
  cd backend
  uvicorn app.main:app --reload --port 8000
  ```

- [ ] 35. 启动中间件服务
  ```bash
  cd middleware
  npm run dev
  ```

- [ ] 36. 启动前端服务
  ```bash
  cd frontend
  npm run dev
  ```

- [ ] 37. 验证系统集成:
  - [ ] 测试模型列表获取
  - [ ] 测试图片上传和推理
  - [ ] 测试模型指标查询
  - [ ] 测试训练功能占位符

- [ ] 38. 优化和修复问题 