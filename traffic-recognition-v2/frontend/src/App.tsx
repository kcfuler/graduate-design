import React, { useState } from 'react';
import Header from './components/Header';
import ModelSelector from './components/ModelSelector';
import MediaUploader from './components/MediaUploader';
import ResultDisplay from './components/ResultDisplay';
import MetricsDisplay from './components/MetricsDisplay';
import TrainingPanel from './components/TrainingPanel';
import type { InferenceResult, ApiError } from './types'; // 导入类型
// import './App.css'; // 如果 index.css 已导入 Tailwind，这个可能不再需要

function App() {
  // 状态管理
  const [selectedModelId, setSelectedModelId] = useState<string | undefined>(undefined);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | ApiError | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  // 处理函数
  const handleModelChange = (modelId: string | undefined) => {
    setSelectedModelId(modelId);
    // 重置结果和指标当模型改变时
    setInferenceResult(null);
    // MetricsDisplay 会自动根据 selectedModelId 更新，无需手动重置
    // setUploadedImageUrl(null); // 可能不需要重置图片？看需求
  };

  const handleInferenceResult = (result: InferenceResult | ApiError) => {
    setInferenceResult(result);
  };

  const handleImagePreview = (url: string | null) => {
    setUploadedImageUrl(url);
    // 当选择新图片时，清除旧的推理结果
    if (url) {
      setInferenceResult(null);
    }
  };

  return (
    <div className="container mx-auto p-4 flex flex-col min-h-screen">
      <Header />
      <main className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-8 flex-grow">
        <section className="space-y-6">
          <div>
            <h2 className="text-xl font-semibold mb-3">1. 选择模型</h2>
            <ModelSelector onModelChange={handleModelChange} />
          </div>
          <div className="mt-6">
             <h2 className="text-xl font-semibold mb-3">2. 上传图片并推理</h2>
            <MediaUploader
              selectedModelId={selectedModelId}
              onResult={handleInferenceResult}
              onImagePreview={handleImagePreview}
            />
          </div>
           <div className="mt-6">
             <h2 className="text-xl font-semibold mb-3">模型指标</h2>
             <MetricsDisplay selectedModelId={selectedModelId} />
          </div>
          <div className="mt-6">
             <h2 className="text-xl font-semibold mb-3">模型训练 (占位符)</h2>
            <TrainingPanel selectedModelId={selectedModelId} />
          </div>
        </section>

        <section>
          <h2 className="text-xl font-semibold mb-3">3. 推理结果</h2>
          <ResultDisplay result={inferenceResult} imageUrl={uploadedImageUrl} />
        </section>

      </main>
      <footer className="mt-8 text-center text-gray-500 text-sm">
        交通标志识别系统 &copy; 2024
      </footer>
    </div>
  );
}

export default App;
