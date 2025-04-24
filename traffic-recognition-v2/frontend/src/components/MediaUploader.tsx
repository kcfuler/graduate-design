import React, { useState, useEffect, ChangeEvent } from 'react';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { performInference } from '../services/api';
import type { InferenceResult, ApiError } from '../types';

interface MediaUploaderProps {
  selectedModelId: string | undefined;
  onResult: (result: InferenceResult | ApiError) => void;
  onImagePreview: (url: string | null) => void;
}

// 辅助函数：将文件读取为 Base64 字符串 (移除 data URL 前缀)
const readFileAsBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        const base64String = reader.result.split(',')[1]; // 获取逗号后的部分
        resolve(base64String);
      } else {
        reject(new Error('无法读取文件为 Base64 字符串'));
      }
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
};

const MediaUploader: React.FC<MediaUploaderProps> = ({ selectedModelId, onResult, onImagePreview }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isInferencing, setIsInferencing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setError(null); // 清除旧错误
    const file = event.target.files?.[0];

    let newPreviewUrl: string | null = null; // 暂存新的 URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (file) {
      setSelectedFile(file);
      newPreviewUrl = URL.createObjectURL(file);
      setPreviewUrl(newPreviewUrl);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
    onImagePreview(newPreviewUrl);
  };

  // 组件卸载时清理预览 URL
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        onImagePreview(null);
      }
    };
  }, [previewUrl, onImagePreview]);

  const handleInferenceClick = async () => {
    if (!selectedModelId) {
      setError('请先选择一个模型');
      return;
    }
    if (!selectedFile) {
      setError('请先选择一个图片文件');
      return;
    }

    setIsInferencing(true);
    setError(null);

    try {
      const base64Data = await readFileAsBase64(selectedFile);
      const result = await performInference({ model_id: selectedModelId, image_data: base64Data });
      onResult(result); // 将成功或 API 错误传递给父组件
    } catch (err) {
      const message = err instanceof Error ? err.message : '推理过程中发生未知错误';
      setError(message);
      onResult({ message }); // 传递错误给父组件
    } finally {
      setIsInferencing(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="file-upload" className="sr-only">
          选择图片
        </label>
        <Input
          id="file-upload"
          type="file"
          accept="image/*" // 暂时只接受图片
          onChange={handleFileChange}
          disabled={isInferencing}
        />
      </div>

      {previewUrl && (
        <div className="mt-4 border rounded-md p-2 max-w-sm mx-auto">
          <img src={previewUrl} alt="文件预览" className="max-w-full h-auto block" />
        </div>
      )}

      {error && (
        <div className="mt-2 text-red-500 text-sm">
          错误: {error}
        </div>
      )}

      <Button
        onClick={handleInferenceClick}
        disabled={!selectedModelId || !selectedFile || isInferencing}
        className="mt-4 w-full"
      >
        {isInferencing ? '处理中...' : '开始推理'}
      </Button>
    </div>
  );
};

export default MediaUploader; 