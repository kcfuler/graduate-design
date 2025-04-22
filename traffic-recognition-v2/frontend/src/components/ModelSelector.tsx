// /Users/bytedance/Documents/毕设/graduate-design/traffic-recognition-v2/frontend/src/components/ModelSelector.tsx
import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ModelType } from "@/types";

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
  disabled?: boolean;
}

const availableModels: ModelType[] = ["yolo", "mobilenet"]; // 可用的模型列表

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  disabled,
}) => {
  return (
    <div className="space-y-2">
      <Label htmlFor="model-select">选择识别模型</Label>
      <Select
        value={selectedModel}
        onValueChange={(value) => onModelChange(value as ModelType)}
        disabled={disabled}
      >
        <SelectTrigger id="model-select" className="w-full">
          <SelectValue placeholder="选择一个模型" />
        </SelectTrigger>
        <SelectContent>
          {availableModels.map((model) => (
            <SelectItem key={model} value={model}>
              {model.toUpperCase()} {/* 显示大写模型名称 */}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};
