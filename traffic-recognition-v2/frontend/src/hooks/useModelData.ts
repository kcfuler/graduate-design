import { useState } from 'react';
import { useRequest } from 'ahooks';
import { modelService, inferenceService } from '../services/api';
import type { Model, InferenceResponse } from '../types';

/**
 * 自定义Hook，处理模型数据和推理过程
 */
export const useModelData = () => {
  const [selectedModel, setSelectedModel] = useState<string>('');
  
  // 获取模型列表
  const { 
    data: models = [], 
    loading: isLoadingModels, 
    error: modelError,
    refresh: refreshModels
  } = useRequest(async () => {
    // 获取所有模型
    const modelsList = await modelService.getModels();
    
    // 获取当前活动模型
    const activeModel = await modelService.getActiveModel();
    
    // 如果有活动模型，设置isActive属性
    if (activeModel) {
      const updatedModels = modelsList.map(model => ({
        ...model,
        isActive: model.id === activeModel.id
      }));
      
      // 设置选中的模型
      setSelectedModel(activeModel.id);
      return updatedModels;
    } else {
      // 如果有模型，默认选择第一个
      if (modelsList.length > 0) {
        setSelectedModel(modelsList[0].id);
      }
      return modelsList;
    }
  }, {
    onError: (error) => {
      console.error('获取模型数据失败:', error);
    }
  });
  
  // 设置激活模型
  const { 
    loading: isSettingActive,
    run: setActiveModel
  } = useRequest(modelService.setActiveModel, {
    manual: true,
    onSuccess: () => {
      // 成功后刷新模型列表
      refreshModels();
    }
  });
  
  // 进行推理
  const { 
    loading: isInferring,
    run: runInference 
  } = useRequest(
    async (file: File): Promise<InferenceResponse | null> => {
      if (!file || !selectedModel) return null;
      
      // 暂时使用模拟数据，因为后端API可能还未实现
      // TODO: 当后端API可用时，替换为真实调用
      // return inferenceService.inferWithFile(selectedModel, file);
      
      // 模拟2秒延迟
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 模拟返回数据
      return {
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
    }, 
    {
      manual: true
    }
  );
  
  // 切换选中的模型
  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
  };
  
  return {
    // 数据
    models,
    selectedModel,
    
    // 加载状态
    isLoadingModels,
    isSettingActive,
    isInferring,
    
    // 错误状态
    modelError,
    
    // 操作函数
    handleModelSelect,
    setActiveModel,
    runInference,
    refreshModels
  };
};

export default useModelData; 