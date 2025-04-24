import React, { useState, useEffect } from 'react';
import { getModels } from '../services/api';
import type { ModelInfo } from '../types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface ModelSelectorProps {
  onModelChange: (modelId: string | undefined) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ onModelChange }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      setIsLoading(true);
      setError(null);
      const result = await getModels();
      if (Array.isArray(result)) { // 检查结果是否为 ModelInfo[]
        setModels(result);
      } else { // 处理 ApiError
        setError(result.message);
      }
      setIsLoading(false);
    };

    fetchModels();
  }, []); // 空依赖数组确保只在挂载时运行

  const handleValueChange = (value: string) => {
    const selectedId = value === "none" ? undefined : value; // 处理可能代表"无选择"的值
    setSelectedModel(selectedId);
    onModelChange(selectedId);
  };

  if (isLoading) {
    return <div>加载模型中...</div>;
  }

  if (error) {
    return <div className="text-red-500">加载模型失败: {error}</div>;
  }

  return (
    <Select onValueChange={handleValueChange} value={selectedModel}>
      <SelectTrigger className="w-[280px]">
        <SelectValue placeholder="选择一个识别模型" />
      </SelectTrigger>
      <SelectContent>
        {/* 可以添加一个"无"或"默认"选项 */}
        {/* <SelectItem value="none">-- 无 --</SelectItem> */}
        {models.map((model) => (
          <SelectItem key={model.id} value={model.id}>
            {model.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default ModelSelector; 