import unittest
import os
import pandas as pd
from app.exporters.csv import CSVExporter


class TestCSVExporter(unittest.TestCase):
    """CSV 导出器测试类"""
    
    def setUp(self):
        """测试准备"""
        self.exporter = CSVExporter("test_exports")
        self.test_results = [{
            "class_id": 1,
            "class_name": "stop",
            "confidence": 0.95,
            "metadata": {
                "image_size": (224, 224),
                "processing_time": 0.1
            }
        }]
        
        self.test_metrics = {
            "inference_time": 0.1,
            "memory_usage": 100.0,
            "gpu_memory": 200.0
        }
        
        self.test_batch = [{
            "results": self.test_results,
            "total_time": 0.3,
            "average_time": 0.1
        }]
    
    def tearDown(self):
        """测试清理"""
        # 清理测试文件
        if os.path.exists("test_exports"):
            for file in os.listdir("test_exports"):
                os.remove(os.path.join("test_exports", file))
            os.rmdir("test_exports")
    
    def test_export_inference_results(self):
        """测试推理结果导出"""
        # 导出结果
        file_path = self.exporter.export_inference_results(self.test_results)
        
        # 检查导出文件
        self.assertTrue(os.path.exists(file_path))
        df = pd.read_csv(file_path)
        
        # 检查数据
        self.assertEqual(len(df), 1)
        self.assertIn("class_id", df.columns)
        self.assertIn("class_name", df.columns)
        self.assertIn("confidence", df.columns)
    
    def test_export_metrics(self):
        """测试性能指标导出"""
        # 导出指标
        file_path = self.exporter.export_metrics(self.test_metrics)
        
        # 检查导出文件
        self.assertTrue(os.path.exists(file_path))
        df = pd.read_csv(file_path)
        
        # 检查数据
        self.assertEqual(len(df), 1)
        self.assertIn("inference_time", df.columns)
        self.assertIn("memory_usage", df.columns)
        self.assertIn("gpu_memory", df.columns)
    
    def test_export_batch_results(self):
        """测试批量结果导出"""
        # 导出结果
        file_path = self.exporter.export_batch_results(self.test_batch)
        
        # 检查导出文件
        self.assertTrue(os.path.exists(file_path))
        df = pd.read_csv(file_path)
        
        # 检查数据
        self.assertEqual(len(df), 1)
        self.assertIn("batch_size", df.columns)
        self.assertIn("total_time", df.columns)
        self.assertIn("average_time", df.columns)
    
    def test_custom_filename(self):
        """测试自定义文件名"""
        # 测试推理结果导出
        custom_path = "custom_inference.csv"
        file_path = self.exporter.export_inference_results(
            self.test_results,
            custom_path
        )
        self.assertEqual(file_path, custom_path)
        
        # 测试性能指标导出
        custom_path = "custom_metrics.csv"
        file_path = self.exporter.export_metrics(
            self.test_metrics,
            custom_path
        )
        self.assertEqual(file_path, custom_path)
        
        # 测试批量结果导出
        custom_path = "custom_batch.csv"
        file_path = self.exporter.export_batch_results(
            self.test_batch,
            custom_path
        )
        self.assertEqual(file_path, custom_path)
    
    def test_empty_data(self):
        """测试空数据导出"""
        # 测试空推理结果
        with self.assertRaises(ValueError):
            self.exporter.export_inference_results([])
        
        # 测试空性能指标
        with self.assertRaises(ValueError):
            self.exporter.export_metrics({})
        
        # 测试空批量结果
        with self.assertRaises(ValueError):
            self.exporter.export_batch_results([])


if __name__ == "__main__":
    unittest.main() 