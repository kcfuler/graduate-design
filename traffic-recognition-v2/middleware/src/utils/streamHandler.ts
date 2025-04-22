import fs from "fs";
import path from "path";
import axios, { AxiosRequestConfig, AxiosResponse } from "axios";
import FormData from "form-data";
import { Request, Response, NextFunction, Express } from "express";
import { formatBackendError, handleApiError } from "./errorHandler";

/**
 * 设置流式代理中间件，用于处理特定的大文件上传（例如视频）
 * 这个函数返回一个 Express 中间件
 * @param {string} backendUrl - 后端服务的基础 URL
 * @param {string} targetPath - 后端接收文件的目标路径 (例如 /infer/video)
 * @returns {Function} Express 中间件函数
 */
export const setupStreamProxy = (backendUrl: string, targetPath: string) => {
  return async (
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    // 检查是否有文件上传
    if (!req.file) {
      console.error("流式代理错误: 没有文件被上传。");
      res.status(400).json({
        error: "请求错误",
        message: "必须上传一个文件。",
        code: "NO_FILE_UPLOADED",
      });
      return;
    }

    const filePath = req.file.path;
    const fileName = req.file.originalname;
    const fileMimeType = req.file.mimetype;

    try {
      const fileStats = fs.statSync(filePath);
      console.log(
        `开始流式代理文件: ${fileName}, 大小: ${fileStats.size} bytes 到 ${backendUrl}${targetPath}`
      );

      // 创建 FormData 实例
      const formData = new FormData();

      // 将文件流添加到 FormData
      // 注意：字段名 ('video' 或 'file') 应与后端期望的匹配
      const fileStream = fs.createReadStream(filePath);
      formData.append("file", fileStream, {
        filename: fileName,
        contentType: fileMimeType,
      });

      // 添加请求体中的其他字段（如果存在）
      if (req.body) {
        for (const key in req.body) {
          if (Object.prototype.hasOwnProperty.call(req.body, key)) {
            formData.append(key, req.body[key]);
          }
        }
      }

      // 配置 Axios 请求
      const config: AxiosRequestConfig = {
        method: "post",
        url: `${backendUrl}${targetPath}`,
        data: formData,
        headers: {
          ...formData.getHeaders(), // 让 form-data 设置 Content-Type 和 Content-Length
          // 可以传递原始请求中的其他相关头部，例如认证信息
          // 'Authorization': req.headers.authorization || '',
        },
        maxBodyLength: Infinity, // 允许发送大文件
        maxContentLength: Infinity,
        responseType: "stream", // 期望后端响应也是流式（如果适用）
      };

      // 发送请求到后端
      const backendResponse: AxiosResponse = await axios(config);

      console.log(`后端响应状态码: ${backendResponse.status} for ${fileName}`);

      // 将后端响应流式传输回客户端
      res.status(backendResponse.status);
      // 复制后端响应头到客户端响应
      for (const header in backendResponse.headers) {
        if (
          Object.prototype.hasOwnProperty.call(backendResponse.headers, header)
        ) {
          res.setHeader(header, backendResponse.headers[header]);
        }
      }
      backendResponse.data.pipe(res);

      // 清理上传的临时文件
      fileStream.on("close", () => {
        fs.unlink(filePath, (unlinkErr) => {
          if (unlinkErr) {
            console.error(`清理临时文件失败: ${filePath}`, unlinkErr);
          } else {
            console.log(`成功清理临时文件: ${filePath}`);
          }
        });
      });
    } catch (error: any) {
      console.error(`流式代理时发生错误 (${fileName}):`, error.message);
      // 清理可能未成功发送的文件
      if (fs.existsSync(filePath)) {
        fs.unlink(filePath, (unlinkErr) => {
          if (unlinkErr)
            console.error(`错误处理中清理文件失败: ${filePath}`, unlinkErr);
        });
      }

      // 检查是否是 Axios 错误
      if (axios.isAxiosError(error) && error.response) {
        // 后端返回了错误响应
        console.error("后端错误响应数据:", error.response.data);
        // 尝试格式化后端错误
        const formattedError = formatBackendError(error.response.data);
        res.status(error.response.status || 500).json(formattedError);
      } else {
        // 其他类型的错误 (网络问题, 文件系统错误等)
        // 使用通用错误处理器处理
        handleApiError(error, req, res, next);
      }
    }
  };
};

/**
 * (可选) 如果需要一个通用的流式代理设置函数，可以添加到 Express app
 * @param {Express} app - Express 应用实例
 * @param {string} routePath - 需要代理的路由路径
 * @param {string} targetUrl - 目标后端的完整 URL
 */
export const setupGenericStreamProxy = (
  app: Express,
  routePath: string,
  targetUrl: string
) => {
  // 注意：这个通用代理可能需要更复杂的逻辑来处理请求体、头部等
  // 并且 http-proxy-middleware 本身对流式处理的支持有限，特别是对于请求体
  // 对于需要精确控制流的场景，推荐使用如上所示的 axios 手动代理方式
  console.warn(
    `通用流式代理 (${routePath}) 配置较为复杂，请谨慎使用或采用手动代理。`
  );
  // 示例性代码，可能无法直接工作
  // app.use(routePath, createProxyMiddleware({ target: targetUrl, changeOrigin: true, ws: true }));
};
