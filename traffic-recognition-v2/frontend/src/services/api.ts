import { 
  Model, 
  ApiResponse, 
  InferenceRequest, 
  InferenceResponse,
  DetectionResult,
  PerformanceMetrics
} from '../types';

// API基础URL, 根据环境使用不同的地址
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api' 
  : 'http://localhost:8000';  // 开发环境指向本地后端

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

/**
 * 模型服务相关API
 */
export const modelService = {
  // 获取所有可用模型
  async getModels(): Promise<Model[]> {
    // 后端返回的数据格式是 { models: Model[] }
    const response = await request<{ models: any[] }>('/models');
    
    // 将后端模型数据映射到前端模型格式
    return response.models.map(model => ({
      id: model.id,
      name: model.name,
      description: model.description || '',
      // 添加假数据，因为后端没有提供这些字段
      accuracy: model.accuracy || 0.85,
      mAP: model.mAP || 0.75,
      fps: model.fps || 25,
      isActive: false  // 默认都不是激活状态，稍后会通过getActiveModel API更新
    }));
  },

  // 获取当前激活模型
  async getActiveModel(): Promise<Model | null> {
    const response = await request<{ active_model: any }>('/models/active');
    
    if (!response.active_model) {
      return null;
    }
    
    return {
      id: response.active_model.id,
      name: response.active_model.name,
      description: response.active_model.description || '',
      accuracy: response.active_model.accuracy || 0.85,
      mAP: response.active_model.mAP || 0.75,
      fps: response.active_model.fps || 25,
      isActive: true
    };
  },

  // 设置当前激活模型
  async setActiveModel(modelId: string): Promise<Model> {
    const response = await request<ApiResponse<Model>>('/models/active', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId }),
    });
    return response.data;
  }
};

/**
 * 推理服务相关API
 */
export const inferenceService = {
  // 使用图像文件进行推理
  async inferWithFile(modelId: string, file: File): Promise<InferenceResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);
    
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
  },

  // 使用图像URL进行推理
  async inferWithUrl(modelId: string, imageUrl: string): Promise<InferenceResponse> {
    const response = await request<ApiResponse<InferenceResponse>>('/infer', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, url: imageUrl }),
    });
    return response.data;
  },

  // 使用视频文件进行推理
  async inferWithVideo(modelId: string, file: File): Promise<InferenceResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);
    
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
};

// 为了向后兼容，导出单独的函数
export const getModels = modelService.getModels;
export const getActiveModel = modelService.getActiveModel;
export const setActiveModel = modelService.setActiveModel;
export const inferWithFile = inferenceService.inferWithFile;
export const inferWithUrl = inferenceService.inferWithUrl;
export const inferWithVideo = inferenceService.inferWithVideo; 