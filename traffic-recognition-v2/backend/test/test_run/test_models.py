#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试工具

测试交通标志识别模型的加载和推理功能
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 添加项目根目录到sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

# 导入模型类
try:
    from app.models.mobilenet_v3.model import MobileNetV3Model
    from app.models.yolov11.model import YOLOv11Model
except ImportError as e:
    print(f"导入模型类时出错: {e}")
    print("请确保您在项目根目录下运行此脚本")
    sys.exit(1)

# 配置文件路径
CONFIG_PATH = current_dir / "config.json"

def load_config():
    """加载模型配置"""
    if not CONFIG_PATH.exists():
        print(f"错误: 配置文件不存在 - {CONFIG_PATH}")
        return None
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def get_test_images(num_images=5):
    """
    获取测试图像路径列表
    
    Args:
        num_images: 要获取的图像数量
        
    Returns:
        测试图像路径列表
    """
    # 检查测试图像目录
    test_images_dir = root_dir / "test_images"
    if not test_images_dir.exists():
        # 如果测试图像目录不存在，则创建
        test_images_dir.mkdir(exist_ok=True)
        print(f"创建测试图像目录: {test_images_dir}")
        
        # 如果没有测试图像，使用样例图像或者生成空白图像
        print("未找到测试图像，将使用生成的空白图像")
        # 生成空白图像
        for i in range(num_images):
            img = np.ones((640, 640, 3), dtype=np.uint8) * 255
            # 添加一些简单形状模拟交通标志
            if i % 3 == 0:
                # 红色圆形 - 模拟禁止标志
                cv2.circle(img, (320, 320), 150, (0, 0, 255), 10)
                cv2.line(img, (220, 320), (420, 320), (0, 0, 255), 10)
            elif i % 3 == 1:
                # 蓝色矩形 - 模拟指示标志
                cv2.rectangle(img, (170, 170), (470, 470), (255, 0, 0), 10)
                cv2.putText(img, "P", (270, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
            else:
                # 黄色三角形 - 模拟警告标志
                pts = np.array([[320, 170], [170, 470], [470, 470]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 255), 10)
                cv2.putText(img, "!", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 10)
            
            # 保存图像
            img_path = test_images_dir / f"test_image_{i+1}.jpg"
            cv2.imwrite(str(img_path), img)
            
    # 获取所有图像文件
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    # 如果仍然没有图像，尝试在其他目录查找
    if not image_files:
        # 查找项目中所有可能的图像
        for ext in ["jpg", "jpeg", "png"]:
            image_files.extend(list(root_dir.glob(f"**/*.{ext}")))
        
        # 限制数量
        image_files = image_files[:num_images]
    
    if not image_files:
        print("未找到任何测试图像")
        return []
    
    return image_files[:num_images]

def test_mobilenet_v3():
    """测试MobileNetV3模型"""
    config = load_config()
    if not config or "models" not in config or "mobilenet_v3" not in config["models"]:
        print("找不到MobileNetV3模型配置")
        return False
    
    model_config = config["models"]["mobilenet_v3"]
    model_file = model_config.get("file_name", "mobilenet_v3_small_224_1.0_float_no_top.h5")
    model_path = current_dir / "mobilenet_v3" / model_file
    
    # 检查模型文件是否存在
    if not model_path.exists():
        print(f"MobileNetV3模型文件不存在: {model_path}")
        print("请先运行 download.py 下载模型")
        return False
    
    print(f"加载MobileNetV3模型: {model_path}")
    
    # 加载模型
    try:
        model = MobileNetV3Model(model_path)
        print("MobileNetV3模型加载成功")
    except Exception as e:
        print(f"加载MobileNetV3模型失败: {e}")
        return False
    
    # 获取测试图像
    test_images = get_test_images(3)
    if not test_images:
        print("没有找到测试图像")
        return False
    
    # 测试结果输出目录
    output_dir = current_dir / "test_results" / "mobilenet_v3"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 对每个测试图像进行预测
    for i, image_path in enumerate(test_images):
        try:
            print(f"测试图像 {i+1}/{len(test_images)}: {image_path}")
            
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            # 记录推理时间
            start_time = time.time()
            
            # 进行预测
            predictions = model.predict(image)
            
            # 计算推理时间
            inference_time = time.time() - start_time
            print(f"推理完成，耗时: {inference_time:.4f} 秒")
            
            # 显示预测结果
            print("预测结果:")
            for j, pred in enumerate(predictions):
                print(f"  {j+1}. 类别: {pred['label']}, 置信度: {pred['confidence']:.4f}")
            
            # 在图像上绘制预测结果
            output_image = image.copy()
            
            # 将BGR转换为RGB用于PIL
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(output_image)
            draw = ImageDraw.Draw(pil_image)
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                # 根据系统获取适合的字体
                if os.name == 'nt':  # Windows
                    font_path = "C:\\Windows\\Fonts\\Arial.ttf"
                else:  # Linux/Mac
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 20)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            # 添加标题
            title = f"MobileNetV3 - Inference time: {inference_time:.4f}s"
            draw.text((10, 10), title, fill="red", font=font)
            
            # 添加预测结果
            y_pos = 40
            for j, pred in enumerate(predictions[:3]):  # 只显示前3个预测结果
                label = pred["label"]
                conf = pred["confidence"]
                text = f"{j+1}. {label}: {conf:.4f}"
                draw.text((10, y_pos), text, fill="red", font=font)
                y_pos += 30
            
            # 保存结果图像
            output_path = output_dir / f"result_{i+1}_{Path(image_path).name}"
            pil_image.save(str(output_path))
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
    
    return True

def test_yolov11():
    """测试YOLOv11模型"""
    config = load_config()
    if not config or "models" not in config or "yolov11" not in config["models"]:
        print("找不到YOLOv11模型配置")
        return False
    
    model_config = config["models"]["yolov11"]
    model_file = model_config.get("file_name", "yolov11n.pt")
    model_path = current_dir / "yolov11" / model_file
    
    # 检查模型文件是否存在
    if not model_path.exists():
        print(f"YOLOv11模型文件不存在: {model_path}")
        print("请先运行 download.py 下载模型")
        return False
    
    print(f"加载YOLOv11模型: {model_path}")
    
    # 加载模型
    try:
        model = YOLOv11Model(model_path)
        print("YOLOv11模型加载成功")
    except Exception as e:
        print(f"加载YOLOv11模型失败: {e}")
        return False
    
    # 获取测试图像
    test_images = get_test_images(3)
    if not test_images:
        print("没有找到测试图像")
        return False
    
    # 测试结果输出目录
    output_dir = current_dir / "test_results" / "yolov11"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 对每个测试图像进行预测
    for i, image_path in enumerate(test_images):
        try:
            print(f"测试图像 {i+1}/{len(test_images)}: {image_path}")
            
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            # 记录推理时间
            start_time = time.time()
            
            # 进行预测
            predictions = model.predict(image)
            
            # 计算推理时间
            inference_time = time.time() - start_time
            print(f"推理完成，耗时: {inference_time:.4f} 秒")
            
            # 显示预测结果
            print("预测结果:")
            for j, pred in enumerate(predictions):
                print(f"  {j+1}. 类别: {pred['label']}, 置信度: {pred['confidence']:.4f}, 边界框: {pred['box']}")
            
            # 在图像上绘制预测结果
            output_image = image.copy()
            
            # 为每个类别分配不同的颜色
            colors = {}
            
            # 绘制边界框和标签
            for pred in predictions:
                # 获取预测信息
                box = pred["box"]
                label = pred["label"]
                conf = pred["confidence"]
                
                # 确保边界框坐标是整数
                x1, y1, x2, y2 = map(int, box)
                
                # 为类别分配一个固定颜色
                if label not in colors:
                    # 随机生成颜色，但避免接近黑色或白色
                    color = tuple(map(int, np.random.randint(50, 200, 3)))
                    colors[label] = color
                else:
                    color = colors[label]
                
                # 绘制边界框
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # 准备标签文本
                text = f"{label}: {conf:.2f}"
                
                # 获取文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # 绘制标签背景
                cv2.rectangle(
                    output_image,
                    (x1, y1 - text_height - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # 绘制标签文本
                cv2.putText(
                    output_image,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
            
            # 添加标题
            cv2.putText(
                output_image,
                f"YOLOv11 - Inference time: {inference_time:.4f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            # 保存结果图像
            output_path = output_dir / f"result_{i+1}_{Path(image_path).name}"
            cv2.imwrite(str(output_path), output_image)
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试交通标志识别模型")
    parser.add_argument(
        '--model', '-m', type=str, default='all',
        choices=['all', 'mobilenet_v3', 'yolov11'],
        help='指定要测试的模型 (默认: all)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("交通标志识别模型测试")
    print("=" * 80)
    
    # 先检查模型文件是否存在
    config = load_config()
    if not config:
        print("无法加载配置文件，请先运行 download.py 下载模型")
        return 1
    
    success = True
    
    if args.model in ['all', 'mobilenet_v3']:
        print("\n" + "=" * 50)
        print("测试 MobileNetV3 模型")
        print("=" * 50)
        if not test_mobilenet_v3():
            print("MobileNetV3模型测试失败")
            success = False
    
    if args.model in ['all', 'yolov11']:
        print("\n" + "=" * 50)
        print("测试 YOLOv11 模型")
        print("=" * 50)
        if not test_yolov11():
            print("YOLOv11模型测试失败")
            success = False
    
    print("\n" + "=" * 80)
    if success:
        print("所有模型测试完成")
        # 显示测试结果目录
        results_dir = current_dir / "test_results"
        print(f"查看测试结果目录: {results_dir}")
        return 0
    else:
        print("模型测试过程中发生错误")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 