import unittest
import numpy as np
from app.main import TrafficSignRecognitionApp


class TestTrafficSignRecognitionApp(unittest.TestCase):
    """交通标识识别应用测试类"""
    
    def setUp(self):
        """测试准备"""
        self.app = TrafficSignRecognitionApp()
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """测试初始化"""
        # 检查组件初始化
        self.assertIsNotNone(self.app.model_factory)
        self.assertIsNotNone(self.app.image_processor)
        self.assertIsNotNone(self.app.metrics_collector)
        self.assertIsNotNone(self.app.csv_exporter)
        
        # 检查模型列表
        self.assertIsInstance(self.app.available_models, list)
        self.assertIn("mobilenet", self.app.available_models)
    
    def test_load_model(self):
        """测试模型加载"""
        # 测试加载 MobileNet
        self.app.load_model("mobilenet")
        self.assertIsNotNone(self.app.current_model)
        self.assertEqual(self.app.current_model_name, "mobilenet")
        
        # 测试重复加载
        original_model = self.app.current_model
        self.app.load_model("mobilenet")
        self.assertEqual(self.app.current_model, original_model)
    
    def test_process_image(self):
        """测试图像处理"""
        # 加载模型
        self.app.load_model("mobilenet")
        
        # 处理图像
        result = self.app.process_image(self.test_image, "mobilenet")
        
        # 检查结果
        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
        self.assertIn("metrics", result)
        self.assertIn("image", result)
        
        # 检查预测结果
        predictions = result["predictions"]
        self.assertIsInstance(predictions, list)
        self.assertTrue(all(isinstance(p, dict) for p in predictions))
        
        # 检查性能指标
        metrics = result["metrics"]
        self.assertIsInstance(metrics, dict)
        self.assertIn("inference_time", metrics)
        self.assertIn("memory_usage", metrics)
    
    def test_process_batch(self):
        """测试批量处理"""
        # 准备批量图像
        batch_images = [self.test_image] * 3
        
        # 处理批量图像
        result = self.app.process_batch(batch_images, "mobilenet")
        
        # 检查结果
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)
        self.assertIn("total_time", result)
        self.assertIn("average_time", result)
        
        # 检查批量结果
        results = result["results"]
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, dict) for r in results))
    
    def test_export_results(self):
        """测试结果导出"""
        # 处理图像
        result = self.app.process_image(self.test_image, "mobilenet")
        
        # 测试导出推理结果
        export_path = self.app.export_results(result, "inference")
        self.assertTrue(export_path.endswith(".csv"))
        
        # 测试导出性能指标
        export_path = self.app.export_results(result, "metrics")
        self.assertTrue(export_path.endswith(".csv"))
        
        # 测试导出批量结果
        batch_result = self.app.process_batch([self.test_image], "mobilenet")
        export_path = self.app.export_results(batch_result, "batch")
        self.assertTrue(export_path.endswith(".csv"))
        
        # 测试无效导出类型
        with self.assertRaises(ValueError):
            self.app.export_results(result, "invalid")


if __name__ == "__main__":
    unittest.main() 