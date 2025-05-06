import React, { useState } from 'react';
import { useRequest } from 'ahooks';
import ResultDisplay, { DetectionResult } from './ResultDisplay';
import MediaUploader from './MediaUploader';
import RunButton from './RunButton';
import MetricsDisplay, { PerformanceMetrics } from './MetricsDisplay';
import ModelSelector from './ModelSelector';
import useModelData from '../hooks/useModelData';
import { InferenceResponse } from '../types';

// 移除模拟数据
// const mockModels: Model[] = [ ... ];

const InferenceLayout: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detections, setDetections] = useState<DetectionResult[] | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);

  // 使用自定义Hook处理模型数据
  const { 
    models,
    selectedModel,
    isLoadingModels,
    isInferring,
    modelError,
    handleModelSelect,
    runInference
  } = useModelData();

  // 处理文件上传
  const handleFileChange = (file: File) => {
    setFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    // 重置推理结果
    setDetections(null);
    setMetrics(null);
  };

  // 处理推理请求
  const handleRunInference = async () => {
    if (!file) return;
    
    const result: InferenceResponse | null = await runInference(file);
    if (!result) return;
    
    setDetections(result.detections);
    setMetrics(result.metrics);
  };
  
  return (
    <div className="grid grid-cols-2 gap-4 p-4 h-[calc(100vh-60px)]">
      {/* 推理结果区域 */}
      <div className="grid-area-infer-result">
        <ResultDisplay 
          imageUrl={previewUrl} 
          detections={detections} 
          isLoading={isInferring}
        />
      </div>
      
      {/* 性能指标区域 */}
      <div className="grid-area-perf-metrics">
        <div className="h-1/2 mb-4">
          <ModelSelector 
            models={models} 
            onModelSelect={handleModelSelect} 
            selectedModel={selectedModel}
            isLoading={isLoadingModels}
            error={modelError ? modelError.message : null}
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
          isDisabled={!file || !selectedModel || models.length === 0} 
          isLoading={isInferring} 
        />
      </div>
    </div>
  );
};

export default InferenceLayout; 