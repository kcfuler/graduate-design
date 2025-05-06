import React from 'react';
import { Progress, Statistic, Row, Col } from 'antd';
import { DashboardOutlined } from '@ant-design/icons';

export interface PerformanceMetrics {
  fps: number;
  mAP: number;
  inferenceTime: number; // 毫秒
}

interface MetricsDisplayProps {
  metrics: PerformanceMetrics | null;
}

const MetricsDisplay: React.FC<MetricsDisplayProps> = ({ metrics }) => {
  // 将mAP值转换为百分比格式
  const mapPercent = metrics ? Math.round(metrics.mAP * 100) : 0;
  
  // 计算FPS进度条的颜色和显示标准
  const getFpsStatus = (fps: number) => {
    if (fps >= 20) return 'success';
    if (fps >= 10) return 'normal';
    return 'exception';
  };
  
  // 计算处理时间进度条的颜色和显示标准
  const getTimeStatus = (time: number) => {
    if (time <= 50) return 'success';
    if (time <= 150) return 'normal';
    return 'exception';
  };

  return (
    <div className="w-full h-full bg-white rounded-lg shadow p-4">
      <div className="flex items-center mb-4">
        <DashboardOutlined className="text-lg mr-2 text-blue-500" />
        <h3 className="text-lg font-medium m-0">性能指标</h3>
      </div>
      
      {!metrics ? (
        <div className="text-center py-8 text-gray-400">
          暂无性能数据
        </div>
      ) : (
        <div className="space-y-6">
          <div>
            <div className="flex justify-between mb-1">
              <span className="font-medium">FPS (每秒帧数)</span>
              <span className="font-bold">{metrics.fps.toFixed(1)}</span>
            </div>
            <Progress 
              percent={Math.min(100, (metrics.fps / 30) * 100)} 
              status={getFpsStatus(metrics.fps)} 
              showInfo={false}
            />
            <div className="text-xs text-gray-500 mt-1">目标: 30+ FPS</div>
          </div>
          
          <div>
            <div className="flex justify-between mb-1">
              <span className="font-medium">mAP@0.5 (平均精度)</span>
              <span className="font-bold">{mapPercent}%</span>
            </div>
            <Progress 
              percent={mapPercent} 
              status={mapPercent >= 80 ? 'success' : 'normal'} 
              showInfo={false}
            />
            <div className="text-xs text-gray-500 mt-1">基准: 80%+</div>
          </div>
          
          <div>
            <div className="flex justify-between mb-1">
              <span className="font-medium">推理时间</span>
              <span className="font-bold">{metrics.inferenceTime.toFixed(0)} ms</span>
            </div>
            <Progress 
              percent={Math.min(100, 100 - (metrics.inferenceTime / 200) * 100)} 
              status={getTimeStatus(metrics.inferenceTime)} 
              showInfo={false}
            />
            <div className="text-xs text-gray-500 mt-1">目标: &lt;50ms</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricsDisplay; 