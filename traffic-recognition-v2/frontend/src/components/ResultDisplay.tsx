import React, { useState, useRef, useEffect } from 'react';
import type { InferenceResult, ApiError } from '../types';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ResultDisplayProps {
  result: InferenceResult | ApiError | null;
  imageUrl: string | null; // 需要原始图片 URL 来显示和绘制边界框
}

// 辅助函数检查是否为 ApiError
function isApiError(data: InferenceResult | ApiError | null): data is ApiError {
  return data !== null && typeof data === 'object' && 'message' in data && !('predictions' in data);
}

// 辅助函数检查是否为有效的 InferenceResult
function isValidResult(data: InferenceResult | ApiError | null): data is InferenceResult {
  return data !== null && typeof data === 'object' && 'predictions' in data && Array.isArray(data.predictions);
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, imageUrl }) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  // 当图片 URL 变化时重置图片尺寸
  useEffect(() => {
    setImageSize(null);
  }, [imageUrl]);

  const handleImageLoad = () => {
    if (imgRef.current) {
      // 使用 offsetWidth/Height 获取渲染后的尺寸，这对于 CSS 缩放更准确
      setImageSize({
        width: imgRef.current.offsetWidth,
        height: imgRef.current.offsetHeight,
      });
    }
  };

  const renderContent = () => {
    if (result === null) {
      return <p className="text-gray-500">等待推理结果...</p>;
    }

    if (isApiError(result)) {
      return <p className="text-red-500">推理错误: {result.message}</p>;
    }

    if (isValidResult(result)) {
      if (result.predictions.length === 0) {
        return <p>未检测到交通标志。</p>;
      }

      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* 图片与边界框 */}
          <div className="relative w-full max-w-md mx-auto">
            {imageUrl && (
              <img
                ref={imgRef}
                src={imageUrl}
                alt="推理图片"
                onLoad={handleImageLoad}
                className="block w-full h-auto rounded-md border"
              />
            )}
            {imageUrl && imageSize && result.predictions.map((pred, index) => {
              if (!pred.box) return null;
              const [xMin, yMin, xMax, yMax] = pred.box; // 假设是归一化坐标 [0, 1]
              const boxStyle: React.CSSProperties = {
                position: 'absolute',
                left: `${xMin * 100}%`,
                top: `${yMin * 100}%`,
                width: `${(xMax - xMin) * 100}%`,
                height: `${(yMax - yMin) * 100}%`,
                border: '2px solid #FF0000', // 红色边框
                pointerEvents: 'none', // 允许下方图片交互
              };
              const labelStyle: React.CSSProperties = {
                position: 'absolute',
                left: `${xMin * 100}%`,
                top: `calc(${yMin * 100}% - 1.2em)`, // 显示在框的上方
                backgroundColor: '#FF0000',
                color: 'white',
                padding: '1px 3px',
                fontSize: '0.75rem',
                whiteSpace: 'nowrap',
              };

              return (
                <React.Fragment key={index}>
                  <div style={boxStyle} />
                  <span style={labelStyle}>
                    {pred.label} ({(pred.score * 100).toFixed(1)}%)
                  </span>
                </React.Fragment>
              );
            })}
          </div>

          {/* 预测结果列表 */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold">识别结果:</h3>
            {result.predictions.map((pred, index) => (
              <Card key={index}>
                <CardHeader className="p-4">
                  <CardTitle className="text-base">{pred.label}</CardTitle>
                </CardHeader>
                <CardContent className="p-4 pt-0">
                  <p>置信度: {(pred.score * 100).toFixed(1)}%</p>
                  {pred.box && (
                    <p className="text-xs text-gray-500 mt-1">
                      边界框: [{pred.box.map(b => b.toFixed(2)).join(', ' )}]
                    </p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      );
    }

    // 如果 result 既不是 null，也不是 ApiError，也不是有效 InferenceResult (理论上不应发生)
    return <p className="text-orange-500">收到无效的响应格式。</p>;
  };

  return <div className="mt-6">{renderContent()}</div>;
};

export default ResultDisplay; 