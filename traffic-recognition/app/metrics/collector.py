import time
import psutil
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class InferenceMetrics:
    """推理指标数据类"""
    inference_time: float  # 推理时间（秒）
    memory_usage: float    # 内存使用（MB）
    gpu_memory: Optional[float] = None  # GPU 内存使用（MB）
    batch_size: int = 1    # 批次大小
    image_size: Optional[tuple] = None  # 图像尺寸


class MetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        self._metrics_history: List[InferenceMetrics] = []
        self._start_time: Optional[float] = None
        self._process = psutil.Process()
    
    def start_timer(self) -> None:
        """开始计时"""
        self._start_time = time.time()
    
    def stop_timer(self) -> float:
        """
        停止计时并返回持续时间
        
        Returns:
            持续时间（秒）
        """
        if self._start_time is None:
            return 0.0
        duration = time.time() - self._start_time
        self._start_time = None
        return duration
    
    def get_memory_usage(self) -> float:
        """
        获取内存使用情况
        
        Returns:
            内存使用量（MB）
        """
        return self._process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """
        获取 GPU 内存使用情况
        
        Returns:
            GPU 内存使用量（MB），如果没有 GPU 则返回 None
        """
        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / 1024 / 1024
    
    def collect_metrics(self, batch_size: int = 1, image_size: Optional[tuple] = None) -> Dict[str, any]:
        """
        收集当前推理的指标
        
        Args:
            batch_size: 批次大小
            image_size: 图像尺寸
            
        Returns:
            包含推理指标的字典
        """
        inference_time = self.stop_timer()
        memory_usage = self.get_memory_usage()
        gpu_memory = self.get_gpu_memory_usage()
        
        # 创建一个普通字典而不是使用InferenceMetrics类
        metrics = {
            "inference_time": float(inference_time),
            "memory_usage": float(memory_usage),
            "gpu_memory": float(gpu_memory) if gpu_memory is not None else None,
            "batch_size": int(batch_size)
        }
        
        # 将图像尺寸转换为可序列化格式
        if image_size is not None:
            metrics["image_width"] = int(image_size[0])
            metrics["image_height"] = int(image_size[1])
        
        # 保存指标历史记录 (使用InferenceMetrics对象仅用于内部存储)
        self._metrics_history.append(InferenceMetrics(
            inference_time=inference_time,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            batch_size=batch_size,
            image_size=image_size
        ))
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        获取平均指标
        
        Returns:
            包含平均指标的字典
        """
        if not self._metrics_history:
            return {}
        
        total_time = sum(m.inference_time for m in self._metrics_history)
        total_memory = sum(m.memory_usage for m in self._metrics_history)
        total_gpu_memory = sum(m.gpu_memory for m in self._metrics_history if m.gpu_memory is not None)
        
        count = len(self._metrics_history)
        gpu_count = sum(1 for m in self._metrics_history if m.gpu_memory is not None)
        
        metrics = {
            "average_inference_time": total_time / count,
            "average_memory_usage": total_memory / count,
            "total_inferences": count
        }
        
        if gpu_count > 0:
            metrics["average_gpu_memory"] = total_gpu_memory / gpu_count
        
        return metrics
    
    def export_metrics(self, file_path: str) -> None:
        """
        导出指标到文件
        
        Args:
            file_path: 导出文件路径
        """
        import pandas as pd
        
        data = []
        for metrics in self._metrics_history:
            row = {
                "timestamp": datetime.now().isoformat(),
                "inference_time": metrics.inference_time,
                "memory_usage": metrics.memory_usage,
                "batch_size": metrics.batch_size
            }
            if metrics.gpu_memory is not None:
                row["gpu_memory"] = metrics.gpu_memory
            if metrics.image_size is not None:
                row["image_width"] = metrics.image_size[0]
                row["image_height"] = metrics.image_size[1]
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self._metrics_history.clear() 