// 引入依赖
const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const cors = require("cors");
const morgan = require("morgan");
const dotenv = require("dotenv");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

// 引入自定义工具和中间件
const { handleApiError } = require("./utils/errorHandler");
const { setupStreamProxy } = require("./utils/streamHandler");

// 加载环境变量
dotenv.config();

// 创建Express应用
const app = express();
const PORT = process.env.PORT || 3001;
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

// 配置上传存储
const uploadDir = path.join(__dirname, "../uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage });

// 中间件配置
app.use(cors());
app.use(morgan("dev"));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 设置API代理
const apiProxy = createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  pathRewrite: {
    "^/api": "",
  },
  onError: handleApiError,
});

// 路由配置
app.use("/api", apiProxy);

// 特殊路由 - 视频上传使用流式处理
app.post(
  "/api/infer/video",
  upload.single("video"),
  setupStreamProxy(BACKEND_URL)
);

// 健康检查路由
app.get("/health", (req, res) => {
  res.status(200).json({ status: "OK", message: "中间件服务运行正常" });
});

// 在生产环境中，这里可以托管前端静态文件
if (process.env.NODE_ENV === "production") {
  const staticPath = path.join(__dirname, "../../frontend/dist");
  app.use(express.static(staticPath));

  // 处理单页应用路由
  app.get("*", (req, res) => {
    res.sendFile(path.join(staticPath, "index.html"));
  });
}

// 全局错误处理中间件
app.use((err, req, res, next) => {
  console.error("Server error:", err);
  res.status(500).json({
    error: "服务器内部错误",
    message: err.message || "发生了未知错误",
  });
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`中间件服务器运行在 http://localhost:${PORT}`);
});
