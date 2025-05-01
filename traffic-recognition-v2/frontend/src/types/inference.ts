export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Prediction {
  label: string;
  confidence: number;
  bounding_box?: BoundingBox; // 仅检测模型的结果有此字段
}

export interface InferenceResult {
  model_id: string;
  predictions: Prediction[];
  inference_time: number;
  image_result?: string; // 包含边界框的Base64图像，仅当draw_boxes=true时返回
}

export interface VideoInferenceResult extends InferenceResult {
  frames_processed: number;
  total_detections: number;
  processing_time: number;
  video_result_url: string;
  detections: {
    frame: number;
    predictions: Prediction[];
  }[];
}

export type MediaType = 'image' | 'video' | 'url' | 'base64'; 