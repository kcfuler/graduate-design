// 通用API响应接口
export interface ApiResponse<T> {
  status?: string;
  message?: string;
  data?: T;
  error?: {
    error: string;
    message: string;
    code: string;
  };
}

// 模型列表响应
export interface ModelsResponse {
  models: import('./model').Model[];
}

// 当前活动模型响应
export interface ActiveModelResponse {
  active_model: import('./model').Model;
}

// 模型指标响应
export interface AllMetricsResponse {
  models_metrics: import('./model').ModelWithMetrics[];
}

// 单个模型指标响应
export interface ModelMetricsResponse {
  model_id: string;
  model_name: string;
  metrics: import('./model').ModelMetrics;
}

// 评估结果响应
export interface EvaluateResponse {
  status: string;
  message: string;
}

// 健康检查响应
export interface HealthResponse {
  status: string;
  message: string;
} 