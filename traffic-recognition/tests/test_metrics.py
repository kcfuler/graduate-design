import unittest
import time
from app.metrics.collector import MetricsCollector, InferenceMetrics


class TestMetricsCollector(unittest.TestCase):
    """性能指标收集器测试类"""
    
    def setUp(self):
        """测试准备"""
        self.collector = MetricsCollector()
    
    def test_timer(self):
        """测试计时功能"""
        # 测试开始计时
        self.collector.start_timer()
        self.assertIsNotNone(self.collector._start_time)
        
        # 测试停止计时
        time.sleep(0.1)  # 等待一段时间
        duration = self.collector.stop_timer()
        self.assertIsNone(self.collector._start_time)
        self.assertGreater(duration, 0)
    
    def test_memory_usage(self):
        """测试内存使用统计"""
        # 测试获取内存使用
        memory = self.collector.get_memory_usage()
        self.assertIsInstance(memory, float)
        self.assertGreater(memory, 0)
    
    def test_collect_metrics(self):
        """测试指标收集"""
        # 开始计时
        self.collector.start_timer()
        time.sleep(0.1)  # 等待一段时间
        
        # 收集指标
        metrics = self.collector.collect_metrics(
            batch_size=2,
            image_size=(224, 224)
        )
        
        # 检查指标
        self.assertIsInstance(metrics, InferenceMetrics)
        self.assertGreater(metrics.inference_time, 0)
        self.assertGreater(metrics.memory_usage, 0)
        self.assertEqual(metrics.batch_size, 2)
        self.assertEqual(metrics.image_size, (224, 224))
    
    def test_get_average_metrics(self):
        """测试平均指标计算"""
        # 收集多个指标
        for _ in range(3):
            self.collector.start_timer()
            time.sleep(0.1)
            self.collector.collect_metrics()
        
        # 获取平均指标
        averages = self.collector.get_average_metrics()
        
        # 检查平均指标
        self.assertIsInstance(averages, dict)
        self.assertIn("average_inference_time", averages)
        self.assertIn("average_memory_usage", averages)
        self.assertIn("total_inferences", averages)
        self.assertEqual(averages["total_inferences"], 3)
    
    def test_export_metrics(self):
        """测试指标导出"""
        import os
        import pandas as pd
        
        # 收集一些指标
        self.collector.start_timer()
        time.sleep(0.1)
        self.collector.collect_metrics()
        
        # 导出指标
        file_path = "test_metrics.csv"
        self.collector.export_metrics(file_path)
        
        # 检查导出文件
        self.assertTrue(os.path.exists(file_path))
        df = pd.read_csv(file_path)
        self.assertFalse(df.empty)
        
        # 清理测试文件
        os.remove(file_path)
    
    def test_clear_history(self):
        """测试历史记录清理"""
        # 收集一些指标
        self.collector.start_timer()
        time.sleep(0.1)
        self.collector.collect_metrics()
        
        # 清理历史记录
        self.collector.clear_history()
        
        # 检查历史记录
        self.assertEqual(len(self.collector._metrics_history), 0)


if __name__ == "__main__":
    unittest.main() 