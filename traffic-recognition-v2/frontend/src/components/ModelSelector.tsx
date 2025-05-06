import React, { useState, useEffect } from 'react';
import { Select, Card, Typography, Badge } from 'antd';
import { ExperimentOutlined } from '@ant-design/icons';

const { Option } = Select;
const { Title, Text } = Typography;

export interface Model {
  id: string;
  name: string;
  description: string;
  accuracy: number;
  mAP: number;
  fps: number;
  isActive?: boolean;
}

interface ModelSelectorProps {
  models: Model[];
  onModelSelect: (modelId: string) => void;
  selectedModel?: string;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ 
  models, 
  onModelSelect, 
  selectedModel 
}) => {
  const [expanded, setExpanded] = useState(false);
  
  const handleModelChange = (modelId: string) => {
    onModelSelect(modelId);
  };

  const selectedModelData = models.find(model => model.id === selectedModel);

  return (
    <div className="w-full bg-white rounded-lg shadow p-4">
      <div className="flex items-center mb-4">
        <ExperimentOutlined className="text-lg mr-2 text-blue-500" />
        <Title level={5} className="m-0">选择识别模型</Title>
      </div>
      
      <Select
        className="w-full mb-4"
        placeholder="选择模型"
        value={selectedModel}
        onChange={handleModelChange}
      >
        {models.map(model => (
          <Option key={model.id} value={model.id}>
            <div className="flex items-center justify-between">
              <span>{model.name}</span>
              {model.isActive && <Badge status="processing" text="当前激活" />}
            </div>
          </Option>
        ))}
      </Select>
      
      {selectedModelData && (
        <Card size="small" className="mt-2">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Text type="secondary">准确率</Text>
              <div className="font-semibold">{(selectedModelData.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div>
              <Text type="secondary">mAP@0.5</Text>
              <div className="font-semibold">{selectedModelData.mAP.toFixed(2)}</div>
            </div>
            <div>
              <Text type="secondary">FPS</Text>
              <div className="font-semibold">{selectedModelData.fps.toFixed(1)}</div>
            </div>
            <div>
              <Text type="secondary">模型ID</Text>
              <div className="font-semibold text-xs text-gray-500 truncate">{selectedModelData.id}</div>
            </div>
          </div>
          <div className="mt-2">
            <Text type="secondary">描述</Text>
            <div className="text-sm">{selectedModelData.description}</div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ModelSelector; 