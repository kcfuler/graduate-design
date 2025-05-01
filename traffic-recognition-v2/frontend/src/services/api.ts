import axios, { AxiosRequestConfig } from 'axios';
import { 
  ApiResponse, ModelsResponse, ActiveModelResponse, 
  AllMetricsResponse, ModelMetricsResponse, InferenceResult,
  VideoInferenceResult
} from '../types';

const API_BASE_URL = '/api';

// 创建axios实例
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json',
  },
});

// 响应拦截器统一处理错误
api.interceptors.response.use(
  (response) => response,
  (error) => {
    let errorMessage = '请求失败';
    let errorCode = 'UNKNOWN_ERROR';

    if (error.response) {
      // 服务器返回了错误响应
      const { data } = error.response;
      if (data && data.error) {
        errorMessage = data.message || '服务器错误';
        errorCode = data.code || 'SERVER_ERROR';
      } else {
        errorMessage = `服务器返回错误: ${error.response.status}`;
        errorCode = `HTTP_${error.response.status}`;
      }
    } else if (error.request) {
      // 请求已发送但未收到响应
      errorMessage = '无法连接到服务器，请检查网络连接';
      errorCode = 'NETWORK_ERROR';
    }

    return Promise.reject({
      error: {
        error: errorCode,
        message: errorMessage,
        code: errorCode,
      }
    });
  }
);

// 模型相关API
export const modelApi = {
  // 获取所有模型
  getAllModels: async (): Promise<ApiResponse<ModelsResponse>> => {
    try {
      const { data } = await api.get<ModelsResponse>('/models');
      return { data };
    } catch (error) {
      return error as ApiResponse<ModelsResponse>;
    }
  },

  // 获取当前活动模型
  getActiveModel: async (): Promise<ApiResponse<ActiveModelResponse>> => {
    try {
      const { data } = await api.get<ActiveModelResponse>('/models/active');
      return { data };
    } catch (error) {
      return error as ApiResponse<ActiveModelResponse>;
    }
  },

  // 设置当前活动模型
  setActiveModel: async (modelId: string): Promise<ApiResponse<{ message: string }>> => {
    try {
      const { data } = await api.post<{ message: string }>('/models/active', { model_id: modelId });
      return { data };
    } catch (error) {
      return error as ApiResponse<{ message: string }>;
    }
  },
};

// 指标相关API
export const metricsApi = {
  // 获取所有模型指标
  getAllMetrics: async (): Promise<ApiResponse<AllMetricsResponse>> => {
    try {
      const { data } = await api.get<AllMetricsResponse>('/metrics');
      return { data };
    } catch (error) {
      return error as ApiResponse<AllMetricsResponse>;
    }
  },

  // 获取特定模型的指标
  getModelMetrics: async (modelId: string): Promise<ApiResponse<ModelMetricsResponse>> => {
    try {
      const { data } = await api.get<ModelMetricsResponse>(`/metrics/${modelId}`);
      return { data };
    } catch (error) {
      return error as ApiResponse<ModelMetricsResponse>;
    }
  },

  // 触发模型评估
  evaluateModel: async (modelId: string): Promise<ApiResponse<{ status: string; message: string }>> => {
    try {
      const { data } = await api.post<{ status: string; message: string }>(`/metrics/${modelId}/evaluate`);
      return { data };
    } catch (error) {
      return error as ApiResponse<{ status: string; message: string }>;
    }
  },
};

// 推理相关API
export const inferenceApi = {
  // 使用图像文件进行推理
  inferWithImage: async (
    modelId: string, 
    imageFile: File, 
    drawBoxes = true
  ): Promise<ApiResponse<InferenceResult>> => {
    try {
      const formData = new FormData();
      formData.append('model_id', modelId);
      formData.append('image', imageFile);
      formData.append('draw_boxes', drawBoxes.toString());
      
      const config: AxiosRequestConfig = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };
      
      const { data } = await api.post<InferenceResult>('/infer', formData, config);
      return { data };
    } catch (error) {
      return error as ApiResponse<InferenceResult>;
    }
  },

  // 使用图像URL进行推理
  inferWithImageUrl: async (
    modelId: string, 
    imageUrl: string, 
    drawBoxes = true
  ): Promise<ApiResponse<InferenceResult>> => {
    try {
      const formData = new FormData();
      formData.append('model_id', modelId);
      formData.append('image_url', imageUrl);
      formData.append('draw_boxes', drawBoxes.toString());
      
      const config: AxiosRequestConfig = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };
      
      const { data } = await api.post<InferenceResult>('/infer', formData, config);
      return { data };
    } catch (error) {
      return error as ApiResponse<InferenceResult>;
    }
  },

  // 使用Base64编码图像进行推理
  inferWithBase64: async (
    modelId: string, 
    imageBase64: string, 
    drawBoxes = true
  ): Promise<ApiResponse<InferenceResult>> => {
    try {
      const { data } = await api.post<InferenceResult>('/infer/base64', {
        model_id: modelId,
        image_base64: imageBase64,
        draw_boxes: drawBoxes,
      });
      return { data };
    } catch (error) {
      return error as ApiResponse<InferenceResult>;
    }
  },

  // 使用视频文件进行推理
  inferWithVideo: async (
    modelId: string, 
    videoFile: File, 
    drawBoxes = true
  ): Promise<ApiResponse<VideoInferenceResult>> => {
    try {
      const formData = new FormData();
      formData.append('model_id', modelId);
      formData.append('video', videoFile);
      formData.append('draw_boxes', drawBoxes.toString());
      
      const config: AxiosRequestConfig = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };
      
      const { data } = await api.post<VideoInferenceResult>('/infer/video', formData, config);
      return { data };
    } catch (error) {
      return error as ApiResponse<VideoInferenceResult>;
    }
  },
};

// 健康检查API
export const healthApi = {
  // 检查中间层状态
  checkHealth: async (): Promise<ApiResponse<{ status: string; message: string }>> => {
    try {
      const { data } = await api.get<{ status: string; message: string }>('/health');
      return { data };
    } catch (error) {
      return error as ApiResponse<{ status: string; message: string }>;
    }
  },
};

export default {
  model: modelApi,
  metrics: metricsApi,
  inference: inferenceApi,
  health: healthApi,
}; 