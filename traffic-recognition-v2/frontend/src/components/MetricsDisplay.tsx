import React, { useState, useEffect } from 'react';
import { getMetrics } from '../services/api';
import type { ApiError } from '../types';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MetricsDisplayProps {
  selectedModelId: string | undefined;
}

// 辅助函数检查是否为 ApiError (复用或重新定义)
function isApiError(data: Record<string, unknown> | ApiError | null): data is ApiError {
  return data !== null && typeof data === 'object' && 'message' in data;
}

// 辅助函数格式化指标值
const formatMetricValue = (value: unknown): string => {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (value === null || value === undefined) {
    return 'N/A';
  }
  // 对于复杂类型，简单地字符串化
  try {
    return JSON.stringify(value);
  } catch {
    return '[无法显示的值]';
  }
};

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ selectedModelId }) => {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedModelId) {
      setMetrics(null);
      setError(null);
      setIsLoading(false);
      return;
    }

    const fetchMetrics = async () => {
      setIsLoading(true);
      setMetrics(null);
      setError(null);
      const result = await getMetrics(selectedModelId);
      if (isApiError(result)) {
        setError(result.message);
      } else {
        setMetrics(result); // result is Record<string, unknown>
      }
      setIsLoading(false);
    };

    fetchMetrics();
  }, [selectedModelId]); // 当 selectedModelId 变化时重新获取

  const renderContent = () => {
    if (!selectedModelId) {
      return <p className="text-gray-500">请选择一个模型以查看指标。</p>;
    }
    if (isLoading) {
      return <p>加载指标中...</p>;
    }
    if (error) {
      return <p className="text-red-500">加载指标错误: {error}</p>;
    }
    if (metrics) {
      const entries = Object.entries(metrics);
      if (entries.length === 0) {
        return <p>此模型无可用指标数据。</p>;
      }
      return (
        <Card>
          <CardHeader>
            <CardTitle>模型性能指标</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {entries.map(([key, value]) => (
              <div key={key} className="flex justify-between text-sm">
                <span className="font-medium capitalize">{key.replace(/_/g, ' ')}:</span>
                <span>{formatMetricValue(value)}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      );
    }
    // 如果没加载、没错误、也没数据（理论上不应发生除非 selectedModelId 为空）
    return null;
  };

  return <div className="mt-6">{renderContent()}</div>;
};

export default MetricsDisplay; 