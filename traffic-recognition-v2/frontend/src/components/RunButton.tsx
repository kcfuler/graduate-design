import React from 'react';
import { Button } from 'antd';
import { PlayCircleOutlined, LoadingOutlined } from '@ant-design/icons';

interface RunButtonProps {
  onRun: () => void;
  isDisabled: boolean;
  isLoading: boolean;
}

const RunButton: React.FC<RunButtonProps> = ({ 
  onRun, 
  isDisabled, 
  isLoading 
}) => {
  return (
    <div className="w-full h-full flex items-center justify-center bg-white rounded-lg shadow p-4">
      <Button
        type="primary"
        size="large"
        icon={isLoading ? <LoadingOutlined /> : <PlayCircleOutlined />}
        onClick={onRun}
        disabled={isDisabled || isLoading}
        className="h-16 w-full text-lg flex items-center justify-center"
      >
        {isLoading ? '正在识别...' : '运行推理'}
      </Button>
    </div>
  );
};

export default RunButton; 