import React, { useState, useRef } from 'react';
import { Upload, Button, message } from 'antd';
import { UploadOutlined, InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';

interface MediaUploaderProps {
  onFileChange: (file: File) => void;
}

const MediaUploader: React.FC<MediaUploaderProps> = ({ onFileChange }) => {
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        onFileChange(file);
      }
    }
  };
  
  const validateFile = (file: File): boolean => {
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'video/mp4', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
      message.error('不支持的文件格式，请上传图片(JPG/PNG/GIF)或视频(MP4/MOV)文件');
      return false;
    }
    
    // 限制文件大小为50MB
    const maxSize = 50 * 1024 * 1024; 
    if (file.size > maxSize) {
      message.error('文件大小超过限制(50MB)');
      return false;
    }
    
    return true;
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        onFileChange(file);
      }
    }
  };

  const uploadProps: UploadProps = {
    beforeUpload: (file) => {
      if (validateFile(file)) {
        onFileChange(file);
      }
      return false;
    },
    showUploadList: false,
  };

  return (
    <div 
      className="w-full h-full p-4 flex flex-col items-center justify-center bg-white rounded-lg shadow"
    >
      <Upload.Dragger {...uploadProps} className="w-full h-full">
        <p className="text-lg flex items-center justify-center">
          <InboxOutlined className="text-blue-500 mr-2 text-2xl" />
          <span>拖放文件到此处或</span>
        </p>
        <p className="mt-2">
          <Button type="primary" icon={<UploadOutlined />}>
            选择文件
          </Button>
        </p>
        <p className="text-gray-400 mt-2">
          支持图片(JPG/PNG/GIF)或视频(MP4/MOV)文件
        </p>
      </Upload.Dragger>
    </div>
  );
};

export default MediaUploader; 