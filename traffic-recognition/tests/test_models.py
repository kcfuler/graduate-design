import unittest
import numpy as np
import torch
from app.models import ModelFactory
from app.models.mobilenet import MobileNetModel


class TestMobileNetModel(unittest.TestCase):
    """MobileNet 模型测试类"""
    
    def setUp(self):
        """测试准备"""
        self.model = MobileNetModel()
        self.test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_load_model(self):
        """测试模型加载"""
        # 测试默认加载
        self.model.load_model()
        self.assertIsNotNone(self.model.model)
        
        # 测试自定义路径加载
        self.model.load_model("path/to/model.pth")
        self.assertIsNotNone(self.model.model)
    
    def test_preprocess(self):
        """测试图像预处理"""
        # 测试单张图像
        processed = self.model.preprocess(self.test_image)
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        
        # 测试批量图像
        batch_images = [self.test_image] * 2
        processed = self.model.preprocess(batch_images)
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (2, 3, 224, 224))
    
    def test_postprocess(self):
        """测试后处理"""
        # 创建模拟输出
        mock_output = torch.randn(1, len(self.model.class_names))
        
        # 测试后处理
        results = self.model.postprocess(mock_output)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, dict) for r in results))
        self.assertTrue(all("class_id" in r for r in results))
        self.assertTrue(all("class_name" in r for r in results))
        self.assertTrue(all("confidence" in r for r in results))


class TestModelFactory(unittest.TestCase):
    """模型工厂测试类"""
    
    def setUp(self):
        """测试准备"""
        self.factory = ModelFactory()
    
    def test_get_available_models(self):
        """测试获取可用模型列表"""
        models = self.factory.get_available_models()
        self.assertIsInstance(models, list)
        self.assertIn("mobilenet", models)
    
    def test_create_model(self):
        """测试创建模型"""
        # 测试创建 MobileNet
        model = self.factory.create_model("mobilenet")
        self.assertIsInstance(model, MobileNetModel)
        
        # 测试创建不存在的模型
        with self.assertRaises(ValueError):
            self.factory.create_model("nonexistent")


if __name__ == "__main__":
    unittest.main() 