/**
 * API错误处理工具
 * 用于统一处理代理过程中的错误
 */

/**
 * 处理API代理过程中的错误
 * @param {Error} err - 错误对象
 * @param {Object} req - 请求对象
 * @param {Object} res - 响应对象
 */
const handleApiError = (err, req, res) => {
  console.error("API代理错误:", err);

  // 根据错误类型返回不同的状态码和消息
  if (err.code === "ECONNREFUSED") {
    return res.status(503).json({
      error: "服务不可用",
      message: "无法连接到后端服务",
      code: "BACKEND_UNAVAILABLE",
    });
  }

  if (err.code === "ETIMEDOUT") {
    return res.status(504).json({
      error: "网关超时",
      message: "后端服务响应超时",
      code: "BACKEND_TIMEOUT",
    });
  }

  // 默认错误响应
  return res.status(500).json({
    error: "内部服务器错误",
    message: "处理请求时发生错误",
    code: "INTERNAL_ERROR",
  });
};

/**
 * 格式化后端错误响应
 * 用于标准化后端返回的错误格式
 * @param {Object} error - 后端返回的错误对象
 * @returns {Object} 格式化后的错误对象
 */
const formatBackendError = (error) => {
  // 如果错误已经是标准格式，直接返回
  if (error && error.error && error.message) {
    return error;
  }

  // 如果是字符串，转换为标准格式
  if (typeof error === "string") {
    return {
      error: "后端错误",
      message: error,
      code: "BACKEND_ERROR",
    };
  }

  // 如果是FastAPI的错误格式
  if (error && error.detail) {
    return {
      error: "请求错误",
      message: error.detail,
      code: "VALIDATION_ERROR",
    };
  }

  // 默认错误格式
  return {
    error: "未知错误",
    message: "发生了未定义的错误",
    code: "UNKNOWN_ERROR",
  };
};

module.exports = {
  handleApiError,
  formatBackendError,
};
