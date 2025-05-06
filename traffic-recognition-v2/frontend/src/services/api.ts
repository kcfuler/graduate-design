import { 
  Model, 
  ApiResponse, 
  InferenceRequest, 
  InferenceResponse,
  DetectionResult,
  PerformanceMetrics
} from '../types';

const API_BASE_URL = '/api';

// 通用请求函数
async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || '请求失败');
  }

  return response.json();
}

// 获取所有可用模型
export async function getModels(): Promise<Model[]> {
  const response = await request<ApiResponse<Model[]>>('/models');
  return response.data;
}

// 获取当前激活模型
export async function getActiveModel(): Promise<Model> {
  const response = await request<ApiResponse<Model>>('/models/active');
  return response.data;
}

// 设置当前激活模型
export async function setActiveModel(modelId: string): Promise<Model> {
  const response = await request<ApiResponse<Model>>('/models/active', {
    method: 'POST',
    body: JSON.stringify({ modelId }),
  });
  return response.data;
}

// 使用图像文件进行推理
export async function inferWithFile(
  modelId: string, 
  file: File
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('modelId', modelId);
  
  const response = await fetch(`${API_BASE_URL}/infer`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || '推理请求失败');
  }
  
  const result = await response.json();
  return result.data;
}

// 使用图像URL进行推理
export async function inferWithUrl(
  modelId: string, 
  imageUrl: string
): Promise<InferenceResponse> {
  const response = await request<ApiResponse<InferenceResponse>>('/infer', {
    method: 'POST',
    body: JSON.stringify({ modelId, url: imageUrl }),
  });
  return response.data;
}

// 使用视频文件进行推理
export async function inferWithVideo(
  modelId: string, 
  file: File
): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('modelId', modelId);
  
  const response = await fetch(`${API_BASE_URL}/infer/video`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || '视频推理请求失败');
  }
  
  const result = await response.json();
  return result.data;
} 