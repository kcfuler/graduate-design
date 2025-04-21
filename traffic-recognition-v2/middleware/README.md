# 交通标志识别系统 - Node.js 中间层

该中间层负责处理前端和后端之间的通信，包括请求转发、错误处理和大文件上传优化。

## 功能特性

- 统一API入口，所有前端请求通过`/api/*`路由
- 将请求无缝转发到Python后端
- 统一的错误处理和响应格式
- 优化大文件上传（特别是视频文件）的流式处理
- 生产环境中托管前端静态文件

## 技术栈

- Express.js: 轻量级Node.js框架
- http-proxy-middleware: 提供请求代理功能
- multer: 处理文件上传
- axios: 处理HTTP请求
- cors: 跨域资源共享

## 环境要求

- Node.js >= 14.0.0
- npm >= 6.0.0

## 目录结构

```
middleware/
├── src/                # 源代码目录
│   ├── index.js        # 主入口文件
│   └── utils/          # 工具函数
│       ├── errorHandler.js  # 错误处理
│       └── streamHandler.js # 流处理
├── uploads/            # 上传文件临时目录（自动创建）
├── .env                # 环境变量
└── package.json        # 项目配置
```

## 安装和设置

1. 安装依赖
   ```bash
   cd middleware
   npm install
   ```

2. 配置环境变量
   在`.env`文件中设置以下变量:
   ```
   PORT=3001
   BACKEND_URL=http://localhost:8000
   ```

## 启动服务

### 开发环境
```bash
npm run dev
```

### 生产环境
```bash
npm start
```

## API路由

所有API请求都应发送到`/api`前缀，中间层会自动将请求转发到后端服务。

例如:
- 前端请求 `/api/models` → 后端接收 `/models`
- 前端请求 `/api/infer` → 后端接收 `/infer`

## 文件上传

对于普通图片上传，请求会直接代理到后端。
对于视频文件，使用专门的流式处理优化，请通过 `/api/infer/video` 端点上传。

## 错误处理

中间层提供统一的错误处理机制，所有错误响应都会按以下格式返回:

```json
{
  "error": "错误类型",
  "message": "详细错误消息",
  "code": "错误代码"
}
```

## 健康检查

可通过访问 `/health` 端点检查中间层服务状态。 