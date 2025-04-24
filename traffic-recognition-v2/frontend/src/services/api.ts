import axios from 'axios';
import type { ModelInfo, InferenceRequest, InferenceResult, ApiError } from '../types';

// 从 Vite 环境变量获取基础 URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'; // 提供备用 URL

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 错误处理辅助函数
const handleError = (error: unknown): ApiError => {
  if (axios.isAxiosError(error) && error.response?.data?.detail) {
    // 尝试提取 FastAPI 错误详情
    return { message: error.response.data.detail };
  } else if (axios.isAxiosError(error)) {
    return { message: error.message };
  } else {
    return { message: '发生未知错误' };
  }
};

// 获取可用模型列表
export const getModels = async (): Promise<ModelInfo[] | ApiError> => {
  try {
    const response = await apiClient.get<ModelInfo[]>('/models');
    return response.data;
  } catch (error) {
    return handleError(error);
  }
};

// 执行推理
export const performInference = async (
  data: InferenceRequest
): Promise<InferenceResult | ApiError> => {
  try {
    // 注意：图片上传通常使用 multipart/form-data
    // 但如果后端期望 base64 字符串，则 application/json 也可以
    // 这里假设后端 API /inference/predict 接受 JSON 包含 base64 图片
    const response = await apiClient.post<InferenceResult>('/inference/predict', data);
    return response.data;
  } catch (error) {
    return handleError(error);
  }
};

// 获取模型指标 (示例)
export const getMetrics = async (modelId: string): Promise<Record<string, unknown> | ApiError> => {
  try {
    const response = await apiClient.get(`/metrics/${modelId}`);
    return response.data;
  } catch (error) {
    return handleError(error);
  }
};

// 触发模型训练 (示例 - 可能需要不同的数据格式和端点)
export const startTraining = async (params: unknown): Promise<{ message: string } | ApiError> => {
  try {
    const response = await apiClient.post('/training/start', params);
    return response.data;
  } catch (error) {
    return handleError(error);
  }
}; 