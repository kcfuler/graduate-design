export interface Model {
  id: string;
  name: string;
  description: string;
  type: 'classification' | 'detection';
  version: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  inference_time: number;
  last_updated: string;
  mAP50?: number; // 仅检测模型有此字段
}

export interface ModelWithMetrics {
  model_id: string;
  model_name: string;
  metrics: ModelMetrics;
} 