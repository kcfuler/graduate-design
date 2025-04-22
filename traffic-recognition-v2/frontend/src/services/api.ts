// /Users/bytedance/Documents/毕设/graduate-design/traffic-recognition-v2/frontend/src/services/api.ts
import axios from "axios";
import { ApiResponse, RecognitionResult, ModelType } from "@/types";

// 从环境变量读取后端 API 地址，如果未设置则使用默认值
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

// 创建 Axios 实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * 上传图片并进行识别
 * @param imageFile 图片文件
 * @param model 使用的模型类型
 * @returns Promise<ApiResponse<RecognitionResult[]>>
 */
export const uploadImageForRecognition = async (
  imageFile: File,
  model: ModelType
): Promise<ApiResponse<RecognitionResult[]>> => {
  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    const response = await apiClient.post<ApiResponse<RecognitionResult[]>>(
      `/predict/${model}`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  } catch (error) {
    // 处理 Axios 错误
    if (axios.isAxiosError(error) && error.response) {
      // 后端返回了错误响应
      return {
        code: error.response.status,
        message: error.response.data?.message || "请求失败",
        data: [],
      };
    } else {
      // 网络错误或其他未知错误
      console.error("API call failed:", error);
      return {
        code: 500,
        message: "网络错误或服务器内部错误",
        data: [],
      };
    }
  }
};

// 可以根据需要添加其他 API 调用函数
// export const getModels = async (): Promise<ApiResponse<string[]>> => { ... };
