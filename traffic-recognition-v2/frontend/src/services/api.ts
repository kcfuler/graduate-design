import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from "axios";

// 创建axios实例
const api = axios.create({
  baseURL: "/api", // 通过中间层代理
  timeout: 30000, // 30秒超时
  headers: {
    "Content-Type": "application/json",
  },
});

// 定义错误响应类型
export interface ApiErrorResponse {
  error: string;
  message: string;
  code?: string;
}

// 默认错误信息
const DEFAULT_ERROR_MESSAGE = "请求失败，请稍后重试";

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 这里可以添加token等认证信息
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error: AxiosError<ApiErrorResponse>) => {
    let errorMessage = DEFAULT_ERROR_MESSAGE;

    if (error.response?.data) {
      errorMessage =
        error.response.data.message ||
        error.response.data.error ||
        DEFAULT_ERROR_MESSAGE;
    } else if (error.message) {
      errorMessage = error.message;
    }

    console.error("API请求错误:", errorMessage);

    return Promise.reject({
      status: error.response?.status || 500,
      message: errorMessage,
      code: error.response?.data?.code || "UNKNOWN_ERROR",
      original: error,
    });
  }
);

// API服务类
export class ApiService {
  // 模型管理
  static async getModels() {
    return api.get("/models");
  }

  static async getActiveModel() {
    return api.get("/models/active");
  }

  static async setActiveModel(modelId: string) {
    return api.post("/models/active", { model_id: modelId });
  }

  // 模型指标
  static async getModelMetrics(modelId?: string) {
    return modelId ? api.get(`/metrics/${modelId}`) : api.get("/metrics");
  }

  static async evaluateModel(modelId: string) {
    return api.post(`/metrics/${modelId}/evaluate`);
  }

  // 推理服务
  static async inferWithImage(
    modelId: string,
    imageFile: File,
    drawBoxes: boolean = true
  ) {
    const formData = new FormData();
    formData.append("model_id", modelId);
    formData.append("image", imageFile);
    formData.append("draw_boxes", String(drawBoxes));

    return api.post("/infer", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  static async inferWithImageUrl(
    modelId: string,
    imageUrl: string,
    drawBoxes: boolean = true
  ) {
    const formData = new FormData();
    formData.append("model_id", modelId);
    formData.append("image_url", imageUrl);
    formData.append("draw_boxes", String(drawBoxes));

    return api.post("/infer", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  static async inferWithBase64Image(
    modelId: string,
    imageBase64: string,
    drawBoxes: boolean = true
  ) {
    return api.post("/infer/base64", {
      model_id: modelId,
      image_base64: imageBase64,
      draw_boxes: drawBoxes,
    });
  }

  static async inferWithVideo(
    modelId: string,
    videoFile: File,
    drawBoxes: boolean = true
  ) {
    const formData = new FormData();
    formData.append("model_id", modelId);
    formData.append("video", videoFile);
    formData.append("draw_boxes", String(drawBoxes));

    return api.post("/infer/video", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      // 设置较长的超时时间，因为视频处理可能需要更长时间
      timeout: 120000, // 2分钟
    });
  }

  // 健康检查
  static async checkHealth() {
    return api.get("/");
  }
}

export default api;
