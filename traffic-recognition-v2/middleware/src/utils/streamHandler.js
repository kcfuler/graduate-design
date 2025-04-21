/**
 * 流处理工具
 * 用于处理大文件上传和流式传输
 */
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");
const { formatBackendError } = require("./errorHandler");

/**
 * 设置流式代理中间件
 * 用于处理大文件上传，如视频文件
 * @param {string} backendUrl - 后端服务URL
 * @returns {Function} Express中间件函数
 */
const setupStreamProxy = (backendUrl) => {
  return async (req, res) => {
    // 如果没有上传文件，返回错误
    if (!req.file) {
      return res.status(400).json({
        error: "请求错误",
        message: "没有上传文件",
        code: "NO_FILE_UPLOADED",
      });
    }

    try {
      const filePath = req.file.path;
      const fileStats = fs.statSync(filePath);

      // 日志记录
      console.log(
        `处理文件上传: ${req.file.originalname}, 大小: ${fileStats.size} bytes`
      );

      // 创建表单数据
      const formData = new FormData();

      // 添加视频文件
      formData.append("video", fs.createReadStream(filePath), {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });

      // 添加其他表单字段
      if (req.body) {
        Object.keys(req.body).forEach((key) => {
          formData.append(key, req.body[key]);
        });
      }

      // 发送请求到后端
      const response = await axios.post(`${backendUrl}/infer/video`, formData, {
        headers: {
          ...formData.getHeaders(),
          "Content-Length": formData.getLengthSync(),
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
      });

      // 上传完成后删除临时文件
      fs.unlinkSync(filePath);

      // 返回后端响应
      return res.status(response.status).json(response.data);
    } catch (error) {
      // 删除临时文件
      if (req.file && req.file.path) {
        try {
          fs.unlinkSync(req.file.path);
        } catch (unlinkError) {
          console.error("删除临时文件失败:", unlinkError);
        }
      }

      // 处理错误响应
      console.error("视频处理错误:", error);

      // 如果是后端返回的错误
      if (error.response) {
        const statusCode = error.response.status;
        const errorData = formatBackendError(error.response.data);
        return res.status(statusCode).json(errorData);
      }

      // 其他错误
      return res.status(500).json({
        error: "视频处理错误",
        message: error.message || "处理视频文件时发生错误",
        code: "VIDEO_PROCESSING_ERROR",
      });
    }
  };
};

/**
 * 流式下载文件
 * 用于从后端下载大文件并流式传输给客户端
 * @param {string} backendUrl - 后端服务URL
 * @param {string} endpoint - API端点
 * @returns {Function} Express中间件函数
 */
const streamDownload = (backendUrl, endpoint) => {
  return async (req, res) => {
    try {
      // 获取查询参数
      const queryParams = new URLSearchParams(req.query).toString();
      const url = `${backendUrl}/${endpoint}${
        queryParams ? `?${queryParams}` : ""
      }`;

      // 发送请求到后端，使用流式响应
      const response = await axios({
        method: "get",
        url,
        responseType: "stream",
        headers: {
          // 转发认证头等
          ...(req.headers.authorization && {
            Authorization: req.headers.authorization,
          }),
        },
      });

      // 设置响应头
      if (response.headers["content-type"]) {
        res.setHeader("Content-Type", response.headers["content-type"]);
      }

      if (response.headers["content-disposition"]) {
        res.setHeader(
          "Content-Disposition",
          response.headers["content-disposition"]
        );
      }

      if (response.headers["content-length"]) {
        res.setHeader("Content-Length", response.headers["content-length"]);
      }

      // 管道流
      response.data.pipe(res);
    } catch (error) {
      console.error("流下载错误:", error);

      // 如果是后端返回的错误
      if (error.response) {
        const statusCode = error.response.status;
        const errorData = formatBackendError(error.response.data);
        return res.status(statusCode).json(errorData);
      }

      // 其他错误
      return res.status(500).json({
        error: "下载错误",
        message: error.message || "从后端下载文件时发生错误",
        code: "DOWNLOAD_ERROR",
      });
    }
  };
};

module.exports = {
  setupStreamProxy,
  streamDownload,
};
