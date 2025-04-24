import React, { useState } from 'react';
import { startTraining } from '../services/api';
// import type { ApiError } from '../types'; // ApiError is unused
import { Button } from "@/components/ui/button";

interface TrainingPanelProps {
  selectedModelId: string | undefined;
}

const TrainingPanel: React.FC<TrainingPanelProps> = ({ selectedModelId }) => {
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleStartTraining = async () => {
    if (!selectedModelId) {
      setError('请先选择一个模型以开始训练。');
      setStatusMessage(null);
      return;
    }

    setIsTraining(true);
    setStatusMessage(null);
    setError(null);

    try {
      // 传递 model_id 作为参数，符合后端预期（如果后端需要）
      const result = await startTraining({ model_id: selectedModelId });

      if (result && typeof result === 'object' && 'message' in result) {
        setStatusMessage(result.message);
      } else {
        setError('收到未知的响应格式。');
      }

    } catch (err) {
      // 处理 startTraining 内部可能抛出的未捕获错误（理论上 api.ts 会处理）
      const message = err instanceof Error ? err.message : '启动训练时发生未知网络错误。';
      setError(message);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="mt-6 p-4 border rounded-md space-y-3">
      <h3 className="text-lg font-semibold">模型训练控制台</h3>
      <Button
        onClick={handleStartTraining}
        disabled={!selectedModelId || isTraining}
        className="w-full md:w-auto"
      >
        {isTraining ? '训练任务启动中...' : '启动模型训练'}
      </Button>
      {statusMessage && (
        <p className="text-green-600 mt-2 text-sm">状态: {statusMessage}</p>
      )}
      {error && (
        <p className="text-red-500 mt-2 text-sm">错误: {error}</p>
      )}
    </div>
  );
};

export default TrainingPanel; 