#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from pathlib import Path
import yaml

def create_dataset_yaml(data_dir, weather_condition, output_file=None):
    """为特定天气条件创建YOLO数据集配置文件"""
    data_dir = Path(data_dir)
    weather_dir = data_dir / weather_condition
    
    if not weather_dir.exists() or not (weather_dir / 'images').exists():
        print(f"错误: 目录不存在 {weather_dir}")
        return None
    
    # 读取类别文件
    classes_file = data_dir.parent / 'yolo' / 'classes.txt'
    if not classes_file.exists():
        print(f"错误: 类别文件不存在 {classes_file}")
        return None
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 默认输出文件名
    if output_file is None:
        output_file = f"weather_{weather_condition}.yaml"
    
    # 创建YAML配置
    config = {
        'path': str(weather_dir.absolute()),
        'train': 'images',  # 在天气条件目录下，所有图像都在images目录中
        'val': 'images',    # 因为样本量可能有限，暂时使用相同数据
        'nc': len(classes),
        'names': classes
    }
    
    # 保存配置文件
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"天气条件 '{weather_condition}' 的数据集配置文件已创建: {output_file}")
    return output_file

def filter_by_class(weather_dir, target_classes, output_dir):
    """
    从特定天气条件目录中筛选出包含特定类别的图像
    
    Args:
        weather_dir: 天气条件目录
        target_classes: 目标类别列表或字典
        output_dir: 输出目录
    """
    weather_dir = Path(weather_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 读取类别映射
    classes_file = weather_dir.parent.parent / 'yolo' / 'classes.txt'
    if not classes_file.exists():
        print(f"错误: 类别文件不存在 {classes_file}")
        return
    
    with open(classes_file, 'r') as f:
        all_classes = [line.strip() for line in f.readlines()]
    
    # 将目标类别转换为类别ID
    if isinstance(target_classes, list):
        target_class_ids = []
        for cls_name in target_classes:
            if cls_name in all_classes:
                target_class_ids.append(all_classes.index(cls_name))
            else:
                print(f"警告: 未找到类别 '{cls_name}'")
    elif isinstance(target_classes, dict):
        # 支持类别ID到名称的映射
        target_class_ids = []
        for cls_id, cls_name in target_classes.items():
            target_class_ids.append(int(cls_id))
    
    # 检查标签并复制文件
    labels_path = weather_dir / 'labels'
    images_path = weather_dir / 'images'
    
    count = 0
    print(f"在 '{weather_dir}' 中筛选包含类别 {target_classes} 的图像...")
    
    for label_file in labels_path.glob('*.txt'):
        # 检查标签文件是否包含目标类别
        found = False
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                if class_id in target_class_ids:
                    found = True
                    break
        
        if found:
            # 复制标签和对应的图像
            image_file = images_path / f"{label_file.stem}.jpg"
            if not image_file.exists():
                image_file = images_path / f"{label_file.stem}.png"
            
            if image_file.exists():
                shutil.copy(label_file, labels_dir / label_file.name)
                shutil.copy(image_file, images_dir / image_file.name)
                count += 1
    
    print(f"已复制 {count} 张包含目标类别的图像到 '{output_dir}'")
    
    # 创建配置文件
    yaml_path = output_dir / "dataset.yaml"
    config = {
        'path': str(output_dir.absolute()),
        'train': 'images',
        'val': 'images',
        'nc': len(all_classes),
        'names': all_classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"配置文件已保存到: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='提取特定天气条件和类别的交通标志数据')
    parser.add_argument('--data_dir', type=str, default='./processed_data/weather_conditions',
                        help='按天气条件分类的数据目录')
    parser.add_argument('--weather', type=str, default='rainy',
                        choices=['rainy', 'foggy', 'night', 'snow', 'normal'],
                        help='要提取的天气条件')
    parser.add_argument('--classes', type=str, nargs='+', default=[],
                        help='要提取的类别名称列表，例如 "p5 p10 p23"')
    parser.add_argument('--output_dir', type=str, default='./processed_data/custom_dataset',
                        help='输出目录')
    parser.add_argument('--create_yaml', action='store_true',
                        help='是否创建YOLO配置文件')
    
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    weather_dir = data_dir / args.weather
    output_dir = Path(args.output_dir) / f"{args.weather}_{'_'.join(args.classes)}" if args.classes else Path(args.output_dir) / args.weather
    
    # 检查目录
    if not weather_dir.exists():
        print(f"错误: 天气条件目录不存在 {weather_dir}")
        return
    
    # 如果指定了类别过滤
    if args.classes:
        filter_by_class(weather_dir, args.classes, output_dir)
    else:
        # 如果不需要类别过滤，只创建配置文件
        if args.create_yaml:
            create_dataset_yaml(data_dir, args.weather, str(output_dir / "dataset.yaml"))
        else:
            print("未指定类别过滤，也未请求创建配置文件，无操作执行")

if __name__ == "__main__":
    main() 