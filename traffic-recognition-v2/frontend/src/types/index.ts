// /Users/bytedance/Documents/毕设/graduate-design/traffic-recognition-v2/frontend/src/types/index.ts

/**
 * API 响应的基础结构
 */
export interface ApiResponse<T = any> {
  code: number;
  message: string;
  data: T;
}

/**
 * 识别结果类型
 */
export interface RecognitionResult {
  label: string; // 识别出的标签
  confidence: number; // 置信度
  box: [number, number, number, number]; // 边界框 [xmin, ymin, xmax, ymax]
}

/**
 * 可选的模型类型
 */
export type ModelType = "yolo" | "mobilenet";
