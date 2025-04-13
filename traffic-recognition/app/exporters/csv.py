import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime


class CSVExporter:
    """CSV 数据导出器"""
    
    def __init__(self, output_dir: str = "exports"):
        """
        初始化导出器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _generate_filename(self, prefix: str = "export") -> str:
        """
        生成导出文件名
        
        Args:
            prefix: 文件名前缀
            
        Returns:
            完整的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.csv"
        return os.path.join(self.output_dir, filename)
    
    def export_inference_results(
        self,
        results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        导出推理结果
        
        Args:
            results: 推理结果列表
            filename: 自定义文件名（可选）
            
        Returns:
            导出的文件路径
        """
        if not results:
            raise ValueError("导出数据不能为空")
        
        # 准备数据
        data = []
        for result in results:
            row = {
                "timestamp": datetime.now().isoformat(),
                "class_id": result.get("class_id"),
                "class_name": result.get("class_name"),
                "confidence": result.get("confidence")
            }
            
            # 添加额外的元数据
            if "metadata" in result:
                row.update(result["metadata"])
            
            data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(data)
        
        # 确定输出文件路径
        output_path = filename if filename else self._generate_filename("inference")
        
        # 导出到 CSV
        df.to_csv(output_path, index=False)
        return output_path
    
    def export_metrics(
        self,
        metrics: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        导出性能指标
        
        Args:
            metrics: 性能指标字典
            filename: 自定义文件名（可选）
            
        Returns:
            导出的文件路径
        """
        if not metrics:
            raise ValueError("导出数据不能为空")
        
        # 准备数据
        data = [{
            "timestamp": datetime.now().isoformat(),
            **metrics
        }]
        
        # 创建 DataFrame
        df = pd.DataFrame(data)
        
        # 确定输出文件路径
        output_path = filename if filename else self._generate_filename("metrics")
        
        # 导出到 CSV
        df.to_csv(output_path, index=False)
        return output_path
    
    def export_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        导出批量处理结果
        
        Args:
            batch_results: 批量处理结果列表
            filename: 自定义文件名（可选）
            
        Returns:
            导出的文件路径
        """
        if not batch_results:
            raise ValueError("导出数据不能为空")
        
        # 准备数据
        data = []
        for batch in batch_results:
            row = {
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(batch.get("results", [])),
                "total_time": batch.get("total_time"),
                "average_time": batch.get("average_time")
            }
            
            # 添加额外的元数据
            if "metadata" in batch:
                row.update(batch["metadata"])
            
            data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(data)
        
        # 确定输出文件路径
        output_path = filename if filename else self._generate_filename("batch")
        
        # 导出到 CSV
        df.to_csv(output_path, index=False)
        return output_path 