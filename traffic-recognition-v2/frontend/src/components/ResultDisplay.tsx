import React, { useEffect, useRef } from 'react';
import { Spin, Empty } from 'antd';
import { FileImageOutlined } from '@ant-design/icons';

export interface DetectionResult {
  label: string;
  confidence: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface ResultDisplayProps {
  imageUrl: string | null;
  detections: DetectionResult[] | null;
  isLoading: boolean;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ 
  imageUrl, 
  detections, 
  isLoading 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // 绘制检测框和标签
  useEffect(() => {
    if (!imageUrl || !detections || detections.length === 0 || !canvasRef.current) {
      return;
    }
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const image = new Image();
    image.onload = () => {
      // 确定画布尺寸，保持宽高比
      const containerWidth = canvas.parentElement?.clientWidth || 600;
      const containerHeight = canvas.parentElement?.clientHeight || 400;
      const scale = Math.min(
        containerWidth / image.width,
        containerHeight / image.height
      );
      
      const width = image.width * scale;
      const height = image.height * scale;
      
      canvas.width = width;
      canvas.height = height;
      
      // 清空画布
      ctx.clearRect(0, 0, width, height);
      
      // 绘制图像
      ctx.drawImage(image, 0, 0, width, height);
      
      // 绘制检测框和标签
      detections.forEach(detection => {
        const { box, label, confidence } = detection;
        const { x, y, width: boxWidth, height: boxHeight } = box;
        
        // 计算实际坐标
        const drawX = x * scale;
        const drawY = y * scale;
        const drawWidth = boxWidth * scale;
        const drawHeight = boxHeight * scale;
        
        // 绘制边界框
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(drawX, drawY, drawWidth, drawHeight);
        
        // 绘制标签背景
        const text = `${label} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = '14px Arial';
        const textWidth = ctx.measureText(text).width;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.fillRect(drawX, drawY - 20, textWidth + 10, 20);
        
        // 绘制标签文本
        ctx.fillStyle = '#000';
        ctx.fillText(text, drawX + 5, drawY - 5);
      });
    };
    
    image.src = imageUrl;
  }, [imageUrl, detections]);
  
  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-white rounded-lg shadow">
        <Spin tip="正在处理图像..." />
      </div>
    );
  }
  
  if (!imageUrl) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-white rounded-lg shadow p-4">
        <Empty 
          image={<FileImageOutlined style={{ fontSize: 60 }} />}
          description="暂无识别结果" 
        />
      </div>
    );
  }
  
  return (
    <div className="w-full h-full flex items-center justify-center bg-white rounded-lg shadow overflow-hidden">
      <canvas 
        ref={canvasRef} 
        className="max-w-full max-h-full object-contain"
      />
    </div>
  );
};

export default ResultDisplay; 