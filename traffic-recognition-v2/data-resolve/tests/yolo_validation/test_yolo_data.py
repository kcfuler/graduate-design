#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import argparse
import random
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证YOLO格式数据')
    parser.add_argument('--data_dir', type=str, default='./processed_data/yolo',
                        help='YOLO格式数据目录')
    parser.add_argument('--samples', type=int, default=5,
                        help='每个集合中抽样的图像数量')
    parser.add_argument('--output_dir', type=str, default='./tests/yolo_validation/output',
                        help='可视化输出目录')
    return parser.parse_args()

def visualize_yolo_annotations(img_path, label_path, class_names, output_path):
    """可视化YOLO格式标注"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return False
    
    height, width = img.shape[:2]
    
    # 如果标签文件存在，读取标签
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 绘制边界框和标签
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            
            # 计算边界框坐标
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # 随机颜色
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 获取类别名称
            class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
            
            # 绘制标签
            label = f"{class_name}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 保存可视化结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return True

def check_yolo_format(data_dir, samples=5, output_dir='./tests/yolo_validation/output'):
    """检查YOLO格式数据的正确性"""
    # 检查目录结构
    required_dirs = [
        os.path.join(data_dir, 'train', 'images'),
        os.path.join(data_dir, 'train', 'labels'),
        os.path.join(data_dir, 'val', 'images'),
        os.path.join(data_dir, 'val', 'labels'),
        os.path.join(data_dir, 'test', 'images'),
        os.path.join(data_dir, 'test', 'labels')
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"错误: 目录 {directory} 不存在")
            return False
    
    # 检查类别文件
    classes_file = os.path.join(data_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        print(f"错误: 类别文件 {classes_file} 不存在")
        return False
    
    # 读取类别名称
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print(f"找到 {len(class_names)} 个类别")
    
    # 检查YAML配置文件
    yaml_file = os.path.join(data_dir, 'tt100k.yaml')
    if not os.path.exists(yaml_file):
        print(f"警告: YAML配置文件 {yaml_file} 不存在")
    
    # 检查每个分割集
    results = {}
    for split in ['train', 'val', 'test']:
        split_results = {}
        images_dir = os.path.join(data_dir, split, 'images')
        labels_dir = os.path.join(data_dir, split, 'labels')
        
        # 获取图像列表
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"{split}集: 找到 {len(image_files)} 张图像")
        split_results['total_images'] = len(image_files)
        
        # 检查标签匹配
        matched_labels = 0
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            if os.path.exists(os.path.join(labels_dir, label_file)):
                matched_labels += 1
        
        print(f"{split}集: {matched_labels}/{len(image_files)} 个图像有对应的标签文件")
        split_results['matched_labels'] = matched_labels
        
        # 随机抽样并可视化
        visualized_samples = []
        if samples > 0 and len(image_files) > 0:
            sample_size = min(samples, len(image_files))
            sampled_images = random.sample(image_files, sample_size)
            
            for img_file in tqdm(sampled_images, desc=f"可视化{split}集样本"):
                img_path = os.path.join(images_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                label_path = os.path.join(labels_dir, f"{base_name}.txt")
                output_path = os.path.join(output_dir, split, img_file)
                
                success = visualize_yolo_annotations(img_path, label_path, class_names, output_path)
                if success:
                    visualized_samples.append(output_path)
        
        split_results['visualized_samples'] = visualized_samples
        results[split] = split_results
    
    # 保存测试结果元数据
    metadata = {
        'data_dir': os.path.abspath(data_dir),
        'num_classes': len(class_names),
        'class_names': class_names,
        'results': results
    }
    
    metadata_path = os.path.join(output_dir, 'validation_results.json')
    with open(metadata_path, 'w') as f:
        # 转换路径为字符串以便JSON序列化
        import json
        class PathEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (list, tuple)):
                    return [self.default(item) for item in obj]
                return str(obj) if hasattr(obj, '__fspath__') else obj

        json.dump(metadata, f, indent=2, cls=PathEncoder)
    
    print(f"可视化结果和验证元数据已保存到 {output_dir}")
    return True

def main():
    args = parse_args()
    
    # 检查YOLO格式数据
    if check_yolo_format(args.data_dir, args.samples, args.output_dir):
        print("YOLO格式数据验证成功！")
    else:
        print("YOLO格式数据验证失败！")

if __name__ == '__main__':
    main() 