import os
import gradio as gr
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

from app.models import ModelFactory
from app.processors.image import ImageProcessor
from app.metrics.collector import MetricsCollector
from app.exporters.csv import CSVExporter


class TrafficSignRecognitionApp:
    """交通标识识别应用"""
    
    def __init__(self):
        """初始化应用"""
        # 初始化组件
        self.model_factory = ModelFactory()
        self.image_processor = ImageProcessor()
        self.metrics_collector = MetricsCollector()
        self.csv_exporter = CSVExporter()
        
        # 获取可用模型列表
        self.available_models = ModelFactory.get_available_models()
        
        # 初始化当前模型
        self.current_model = None
        self.current_model_name = None
        if self.available_models:
            self.load_model(self.available_models[0])
    
    def load_model(self, model_name: str) -> None:
        """
        加载模型
        
        Args:
            model_name: 模型名称
        """
        if model_name != self.current_model_name:
            self.current_model = self.model_factory.create_model(model_name, model_path=model_name)
            self.current_model_name = model_name
    
    def process_image(
        self,
        image: np.ndarray,
        model_name: str
    ) -> Dict[str, any]:
        """
        处理单张图像
        
        Args:
            image: 输入图像
            model_name: 模型名称
            
        Returns:
            处理结果字典
        """
        # 检查图像是否为None
        if image is None:
            return {
                "error": "输入图像为空",
                "image": None,
                "predictions": [],
                "metrics": {}
            }
            
        # 加载模型
        self.load_model(model_name)
        
        # 开始收集指标
        self.metrics_collector.start_timer()
        
        try:
            # 预处理图像
            processed_image = self.image_processor.preprocess(image)
            
            # 模型推理
            predictions = self.current_model.predict(processed_image)
            
            # 收集指标
            metrics = self.metrics_collector.collect_metrics(
                batch_size=1,
                image_size=image.shape[:2]
            )
            
            # 准备结果
            result = {
                "predictions": predictions,
                "metrics": metrics,
                "image": image
            }
            
            return result
            
        except Exception as e:
            # 停止计时，确保不会影响下一次请求
            self.metrics_collector.stop_timer()
            
            # 返回标准化的错误信息
            return {
                "error": str(e),
                "image": image,
                "predictions": [],
                "metrics": {}
            }
    
    def process_batch(
        self,
        images: List[np.ndarray],
        model_name: str
    ) -> Dict[str, any]:
        """
        批量处理图像
        
        Args:
            images: 输入图像列表
            model_name: 模型名称
            
        Returns:
            处理结果字典
        """
        # 检查输入图像列表是否为空
        if not images:
            return {
                "error": "输入图像列表为空",
                "results": [],
                "total_time": 0.0,
                "average_time": 0.0
            }
            
        # 加载模型
        self.load_model(model_name)
        
        # 开始收集指标
        self.metrics_collector.start_timer()
        
        try:
            results = []
            total_time = 0.0
            
            for image in images:
                # 检查单个图像是否为None
                if image is None:
                    results.append({
                        "error": "输入图像为空",
                        "predictions": [],
                        "metrics": {},
                        "image": None
                    })
                    continue
                
                # 预处理图像
                processed_image = self.image_processor.preprocess(image)
                
                # 模型推理
                predictions = self.current_model.predict(processed_image)
                
                # 收集指标
                metrics = self.metrics_collector.collect_metrics(
                    batch_size=1,
                    image_size=image.shape[:2]
                )
                
                results.append({
                    "predictions": predictions,
                    "metrics": metrics,
                    "image": image
                })
                
                total_time += metrics.get("inference_time", 0.0)
            
            # 计算平均时间，避免除以零的情况
            average_time = total_time / len(images) if len(images) > 0 else 0.0
            
            return {
                "results": results,
                "total_time": float(total_time),
                "average_time": float(average_time)
            }
            
        except Exception as e:
            # 停止计时，确保不会影响下一次请求
            self.metrics_collector.stop_timer()
            
            # 返回标准化的错误信息
            return {
                "error": str(e),
                "results": [],
                "total_time": 0.0,
                "average_time": 0.0
            }
    
    def export_results(
        self,
        results: Dict[str, any],
        export_type: str
    ) -> str:
        """
        导出结果
        
        Args:
            results: 处理结果
            export_type: 导出类型
            
        Returns:
            导出文件路径
        """
        try:
            if export_type == "inference":
                return self.csv_exporter.export_inference_results(
                    results["predictions"]
                )
            elif export_type == "metrics":
                return self.csv_exporter.export_metrics(
                    results["metrics"]
                )
            elif export_type == "batch":
                return self.csv_exporter.export_batch_results(
                    [results]
                )
            else:
                raise ValueError(f"不支持的导出类型: {export_type}")
                
        except Exception as e:
            return f"导出失败: {str(e)}"


def create_interface():
    """创建 Gradio 界面"""
    app = TrafficSignRecognitionApp()
    
    with gr.Blocks(title="交通标识识别系统") as interface:
        gr.Markdown("# 交通标识识别系统")
        
        with gr.Row():
            with gr.Column():
                # 模型选择
                model_dropdown = gr.Dropdown(
                    choices=app.available_models,
                    label="选择模型",
                    value=app.available_models[0] if app.available_models else None
                )
                
                # 图像输入
                image_input = gr.Image(
                    label="输入图像",
                    type="numpy"
                )
                
                # 处理按钮
                process_btn = gr.Button("处理图像")
                
                # 批量处理
                batch_input = gr.File(
                    label="批量处理",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                batch_process_btn = gr.Button("批量处理")
            
            with gr.Column():
                # 结果显示
                result_output = gr.JSON(
                    label="处理结果"
                )
                
                # 性能指标
                metrics_output = gr.JSON(
                    label="性能指标"
                )
                
                # 导出选项
                export_type = gr.Radio(
                    choices=["inference", "metrics", "batch"],
                    label="导出类型",
                    value="inference"
                )
                
                export_btn = gr.Button("导出结果")
                
                # 导出结果
                export_output = gr.Text(
                    label="导出结果"
                )
        
        # 单张图像处理
        process_btn.click(
            fn=app.process_image,
            inputs=[image_input, model_dropdown],
            outputs=[result_output, metrics_output]
        )
        
        # 批量处理
        batch_process_btn.click(
            fn=app.process_batch,
            inputs=[batch_input, model_dropdown],
            outputs=[result_output, metrics_output]
        )
        
        # 导出结果
        export_btn.click(
            fn=app.export_results,
            inputs=[result_output, export_type],
            outputs=export_output
        )
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) 