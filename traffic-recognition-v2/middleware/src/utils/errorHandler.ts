import { Request, Response, NextFunction } from "express";

/**
 * API错误响应接口
 */
interface ApiErrorResponse {
  error: string;
  message: string;
  code: string;
  details?: any; // 可选的详细信息
}

/**
 * 处理API代理或服务器内部过程中的错误
 * @param {Error | any} err - 错误对象
 * @param {Request} req - Express请求对象
 * @param {Response} res - Express响应对象
 * @param {NextFunction} next - Express next中间件函数
 */
export const handleApiError = (
  err: any,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  console.error(`API 错误发生在 ${req.method} ${req.originalUrl}:`, err);

  let statusCode: number = 500;
  let responseBody: ApiErrorResponse = {
    error: "内部服务器错误",
    message: "处理请求时发生意外错误。",
    code: "INTERNAL_ERROR",
  };

  // 检查是否是 http-proxy-middleware 的错误
  if (err.code) {
    switch (err.code) {
      case "ECONNREFUSED":
        statusCode = 503; // 服务不可用
        responseBody = {
          error: "服务不可用",
          message: "无法连接到目标后端服务。",
          code: "BACKEND_UNAVAILABLE",
        };
        break;
      case "ETIMEDOUT":
      case "ECONNRESET": // 连接被重置也可能是超时或网络问题
        statusCode = 504; // 网关超时
        responseBody = {
          error: "网关超时",
          message: "后端服务响应超时或连接中断。",
          code: "BACKEND_TIMEOUT",
        };
        break;
      // 可以根据需要添加更多特定的错误代码处理
      default:
        // 对于其他已知代码但未特殊处理的错误，保留默认500，但使用错误代码
        responseBody.message = err.message || "代理过程中发生未知错误。";
        responseBody.code = err.code;
        break;
    }
  } else if (err instanceof Error) {
    // 处理标准 Error 对象
    responseBody.message = err.message;
    // 如果是特定类型的自定义错误，可以在这里添加处理逻辑
    // 例如: if (err instanceof ValidationError) { statusCode = 400; ... }
  } else {
    // 处理非Error类型的异常（例如直接throw一个字符串）
    responseBody.message = "发生了未知类型的错误。";
    if (typeof err === "string") {
      responseBody.details = err;
    }
  }

  // 防止在响应已发送后再次发送头
  if (!res.headersSent) {
    res.status(statusCode).json(responseBody);
  }
};

/**
 * 格式化后端可能返回的错误响应 (如果需要)
 * @param {any} error - 后端返回的原始错误信息
 * @returns {ApiErrorResponse} 标准化后的错误对象
 */
export const formatBackendError = (error: any): ApiErrorResponse => {
  // 尝试解析后端错误结构
  if (error && typeof error === "object") {
    // FastAPI validation errors (HTTP 422)
    if (error.detail && Array.isArray(error.detail)) {
      return {
        error: "验证错误",
        message: "请求数据验证失败。",
        code: "VALIDATION_ERROR",
        details: error.detail.map((e: any) => ({
          field: e.loc?.join("."),
          msg: e.msg,
        })),
      };
    }
    // 其他可能的后端错误结构
    if (error.error && error.message && error.code) {
      return error as ApiErrorResponse;
    }
    if (error.message) {
      return {
        error: "后端错误",
        message: error.message,
        code: error.code || "BACKEND_ERROR",
        details: error.details,
      };
    }
  }

  // 如果无法解析，返回通用后端错误
  return {
    error: "后端错误",
    message: "从后端接收到无法解析的错误响应。",
    code: "UNKNOWN_BACKEND_ERROR",
    details: error, // 包含原始错误信息以供调试
  };
};
