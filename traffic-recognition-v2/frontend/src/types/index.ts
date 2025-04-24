// 基础类型定义

export interface ApiError {
  message: string;
}

// 模型相关类型 (示例)
export interface ModelInfo {
  id: string;
  name: string;
  description: string;
}

// 推理相关类型 (示例)
export interface InferenceRequest {
  model_id: string;
  image_data: string; // Base64 encoded image
}

export interface InferenceResult {
  predictions: Array<{ label: string; score: number; box?: [number, number, number, number] }>;
}

// 可以根据需要添加更多类型定义 