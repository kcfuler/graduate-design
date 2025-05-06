import React from 'react';
import { Select, Spin, Alert } from 'antd';
import Card from 'antd/lib/card';
import Typography from 'antd/lib/typography';
import { ExperimentOutlined } from '@ant-design/icons';

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
  isLoading?: boolean;
  error?: string | null;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ 
  models, 
  onModelSelect, 
  selectedModel,
  isLoading = false,
  error = null
}) => {
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
      
      {isLoading ? (
        <div className="flex justify-center items-center py-8">
          <Spin />
          <span className="ml-2">加载模型列表中...</span>
        </div>
      ) : error ? (
        <Alert message={error} type="error" showIcon />
      ) : (
        <>
          <div style={{ marginBottom: '1rem' }}>
            <Select
              placeholder="选择模型"
              value={selectedModel}
              onChange={handleModelChange}
              disabled={models.length === 0}
              style={{ width: '100%' }}
            >
              {models.map(model => (
                <Select.Option key={model.id} value={model.id}>
                  {model.name}{model.isActive ? ' (当前激活)' : ''}
                </Select.Option>
              ))}
            </Select>
          </div>
          
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
        </>
      )}
    </div>
  );
};

export default ModelSelector; 