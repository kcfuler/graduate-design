import express, { Express, Request, Response, NextFunction } from "express";
import {
  createProxyMiddleware,
  Options as ProxyOptions,
} from "http-proxy-middleware";
import cors from "cors";
import morgan from "morgan";
import dotenv from "dotenv";
import multer, { StorageEngine } from "multer";
import path from "path";
import fs from "fs";
import { handleApiError } from "./utils/errorHandler"; // 假设 errorHandler.ts 存在且导出 handleApiError
import { setupStreamProxy } from "./utils/streamHandler"; // 假设 streamHandler.ts 存在且导出 setupStreamProxy

// 加载环境变量
dotenv.config();

// 创建Express应用
const app: Express = express();
const PORT: number = parseInt(process.env.PORT || "3001", 10);
const BACKEND_URL: string = process.env.BACKEND_URL || "http://localhost:8000";

// --- 上传配置 ---
const uploadDir = path.join(__dirname, "../uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage: StorageEngine = multer.diskStorage({
  destination: function (
    req: Request,
    file: Express.Multer.File,
    cb: (error: Error | null, destination: string) => void
  ) {
    cb(null, uploadDir);
  },
  filename: function (
    req: Request,
    file: Express.Multer.File,
    cb: (error: Error | null, filename: string) => void
  ) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage });

// --- 中间件配置 ---
app.use(cors()); // 启用CORS
app.use(morgan("dev")); // 日志记录
app.use(express.json()); // 解析JSON请求体
app.use(express.urlencoded({ extended: true })); // 解析URL编码的请求体

// --- API 代理配置 ---
const apiProxyOptions: ProxyOptions = {
  target: BACKEND_URL,
  changeOrigin: true, // 需要虚拟托管站点
  pathRewrite: { "^/api": "" }, // 重写路径：移除 /api 前缀
  onError: (err: Error, req: Request, res: Response) => {
    console.error("代理错误:", err);
    handleApiError(
      err,
      req,
      res as express.Response<any, Record<string, any>>,
      () => {}
    ); // 调用错误处理
  },
  onProxyReq: (proxyReq, req, res) => {
    console.log(
      `代理请求: ${req.method} ${req.path} -> ${BACKEND_URL}${proxyReq.path}`
    );
    // 如果有需要，可以在这里修改请求头等
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log(`收到代理响应: ${proxyRes.statusCode} for ${req.path}`);
    // 如果有需要，可以在这里修改响应头等
  },
};

// --- 路由 ---

// 健康检查
app.get("/health", (req: Request, res: Response) => {
  res.status(200).send("中间件服务运行正常");
});

// 文件上传代理 (特定处理)
app.post(
  "/api/upload",
  upload.single("file"),
  (req: Request, res: Response, next: NextFunction) => {
    // 文件上传由 multer 处理，这里可以将请求转发给后端
    // 注意：需要正确处理文件转发，http-proxy-middleware 可能不直接支持 multer 上传的文件转发
    // 可能需要使用 axios 或类似库手动转发
    console.log("收到文件上传请求:", req.file);
    // TODO: 实现文件转发到后端 /upload 端点的逻辑
    res.status(501).json({ message: "文件上传转发尚未实现" });
  }
);

// 流式API代理 (如果需要)
// setupStreamProxy(app, '/api/stream', BACKEND_URL + '/stream'); // 示例

// 通用API代理 (放在最后，捕获所有其他 /api/* 请求)
app.use("/api", createProxyMiddleware(apiProxyOptions));

// --- 全局错误处理 ---
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  handleApiError(err, req, res, next);
});

// --- 启动服务器 ---
app.listen(PORT, () => {
  console.log(`中间件服务器运行在 http://localhost:${PORT}`);
  console.log(`代理目标后端: ${BACKEND_URL}`);
});

export default app;
