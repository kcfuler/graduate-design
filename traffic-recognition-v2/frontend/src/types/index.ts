export * from './model';
export * from './inference';
export * from './response';

// 交通标志识别系统的类型定义

// API响应的基本结构
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  code?: string;
}

// API错误响应结构
export interface ApiError {
  error: string;
  message: string;
  code: string;
}

// 识别结果类型
export interface DetectionResult {
  label: string;
  confidence: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

// 模型信息类型
export interface Model {
  id: string;
  name: string;
  description: string;
  accuracy: number;
  mAP: number;
  fps: number;
  isActive?: boolean;
}

// 性能指标类型
export interface PerformanceMetrics {
  fps: number;
  mAP: number;
  inferenceTime: number; // 毫秒
}

// 推理请求参数
export interface InferenceRequest {
  modelId: string;
  file?: File;
  url?: string;
}

// 推理响应
export interface InferenceResponse {
  detections: DetectionResult[];
  metrics: PerformanceMetrics;
  imageUrl?: string; // 处理后的图像URL
} 