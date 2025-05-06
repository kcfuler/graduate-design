import React, { useState } from 'react';
import ResultDisplay, { DetectionResult } from './ResultDisplay';
import MediaUploader from './MediaUploader';
import RunButton from './RunButton';
import MetricsDisplay, { PerformanceMetrics } from './MetricsDisplay';
import ModelSelector, { Model } from './ModelSelector';

// 模拟数据
const mockModels: Model[] = [
  {
    id: 'yolov8n',
    name: 'YOLOv8-Nano',
    description: '轻量级模型，适合边缘设备',
    accuracy: 0.83,
    mAP: 0.65,
    fps: 35,
    isActive: true
  },
  {
    id: 'yolov8s',
    name: 'YOLOv8-Small',
    description: '平衡性能和速度的模型',
    accuracy: 0.87,
    mAP: 0.75,
    fps: 28
  },
  {
    id: 'yolov8m',
    name: 'YOLOv8-Medium',
    description: '较高精度模型，适合通用场景',
    accuracy: 0.91,
    mAP: 0.82,
    fps: 22
  }
];

const InferenceLayout: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('yolov8n');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [detections, setDetections] = useState<DetectionResult[] | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  
  // 处理文件上传
  const handleFileChange = (file: File) => {
    setFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    // 重置推理结果
    setDetections(null);
    setMetrics(null);
  };
  
  // 处理模型选择
  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
  };
  
  // 运行推理
  const handleRunInference = async () => {
    if (!file || !selectedModel) return;
    
    setIsProcessing(true);
    
    try {
      // 这里应该调用后端API进行推理
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 模拟返回数据
      const mockData = {
        detections: [
          {
            label: '限速30',
            confidence: 0.95,
            box: { x: 100, y: 100, width: 150, height: 150 }
          },
          {
            label: '停车让行',
            confidence: 0.88,
            box: { x: 400, y: 200, width: 120, height: 120 }
          }
        ],
        metrics: {
          fps: 28.5,
          mAP: 0.79,
          inferenceTime: 35  // 毫秒
        }
      };
      
      setDetections(mockData.detections);
      setMetrics(mockData.metrics);
    } catch (error) {
      console.error('推理失败:', error);
      // 处理错误
    } finally {
      setIsProcessing(false);
    }
  };
  
  return (
    <div className="grid grid-cols-2 gap-4 p-4 h-[calc(100vh-60px)]">
      {/* 推理结果区域 */}
      <div className="grid-area-infer-result">
        <ResultDisplay 
          imageUrl={previewUrl} 
          detections={detections} 
          isLoading={isProcessing}
        />
      </div>
      
      {/* 性能指标区域 */}
      <div className="grid-area-perf-metrics">
        <div className="h-1/2 mb-4">
          <ModelSelector 
            models={mockModels} 
            onModelSelect={handleModelSelect} 
            selectedModel={selectedModel}
          />
        </div>
        <div className="h-1/2">
          <MetricsDisplay metrics={metrics} />
        </div>
      </div>
      
      {/* 上传区域 */}
      <div className="grid-area-upload">
        <MediaUploader onFileChange={handleFileChange} />
      </div>
      
      {/* 运行按钮区域 */}
      <div className="grid-area-run-button">
        <RunButton 
          onRun={handleRunInference} 
          isDisabled={!file} 
          isLoading={isProcessing} 
        />
      </div>
    </div>
  );
};

export default InferenceLayout; 