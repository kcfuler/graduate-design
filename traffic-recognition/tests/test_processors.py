import unittest
import numpy as np
from app.processors.image import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """图像处理器测试类"""
    
    def setUp(self):
        """测试准备"""
        self.processor = ImageProcessor()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_is_supported_format(self):
        """测试格式检查"""
        # 测试支持的格式
        self.assertTrue(self.processor.is_supported_format("test.jpg"))
        self.assertTrue(self.processor.is_supported_format("test.png"))
        self.assertTrue(self.processor.is_supported_format("test.jpeg"))
        
        # 测试不支持的格式
        self.assertFalse(self.processor.is_supported_format("test.txt"))
        self.assertFalse(self.processor.is_supported_format("test.pdf"))
    
    def test_preprocess(self):
        """测试图像预处理"""
        # 测试默认尺寸
        processed = self.processor.preprocess(self.test_image)
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, (224, 224, 3))
        
        # 测试自定义尺寸
        processed = self.processor.preprocess(self.test_image, target_size=(128, 128))
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape, (128, 128, 3))
        
        # 测试数值范围
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))
    
    def test_batch_preprocess(self):
        """测试批量预处理"""
        # 准备批量图像
        batch_images = [self.test_image] * 3
        
        # 测试批量处理
        processed = self.processor.batch_preprocess(batch_images)
        self.assertIsInstance(processed, list)
        self.assertEqual(len(processed), 3)
        
        # 检查每张图像
        for img in processed:
            self.assertIsInstance(img, np.ndarray)
            self.assertEqual(img.shape, (224, 224, 3))
    
    def test_draw_detection(self):
        """测试检测结果绘制"""
        # 准备检测结果
        detections = [{
            "class_name": "stop",
            "confidence": 0.95,
            "bbox": [100, 100, 200, 200]
        }]
        
        # 测试绘制
        result = self.processor.draw_detection(self.test_image, detections)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.test_image.shape)


if __name__ == "__main__":
    unittest.main() 