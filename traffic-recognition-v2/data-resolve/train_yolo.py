#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import yaml
from pathlib import Path

def check_yolo_installed():
    """检查是否安装了YOLO命令行工具"""
    try:
        subprocess.run(['yolo', '--help'], capture_output=True)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def create_dataset_yaml(data_dir, output_file='dataset.yaml'):
    """创建YOLO数据集配置文件"""
    data_dir = Path(data_dir)
    
    # 读取类别文件
    classes_file = data_dir / 'classes.txt'
    if not classes_file.exists():
        print(f"错误: 类别文件不存在 {classes_file}")
        return None
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 创建YAML配置
    config = {
        'path': str(data_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    # 保存配置文件
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"数据集配置文件已创建: {output_file}")
    return output_file

def train_yolo(data_yaml, model='yolov11', epochs=100, img_size=640, batch_size=16, 
               weights='', device='0', project='runs/train', name='exp'):
    """训练YOLO模型"""
    if not check_yolo_installed():
        print("错误: YOLO命令行工具未安装，请先安装它")
        print("提示: 可以通过 'pip install ultralytics' 安装")
        return False
    
    # 构建训练命令
    cmd = [
        'yolo', 'train',
        f'model={model}',
        f'data={data_yaml}',
        f'epochs={epochs}',
        f'imgsz={img_size}',
        f'batch={batch_size}',
        f'device={device}',
        f'project={project}',
        f'name={name}'
    ]
    
    # 如果提供了预训练权重
    if weights:
        cmd.append(f'weights={weights}')
    
    # 执行训练命令
    print("开始训练YOLO模型...")
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
        return True
    except subprocess.SubprocessError as e:
        print(f"训练过程出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='训练YOLO模型')
    parser.add_argument('--data_dir', type=str, default='./processed_data/yolo',
                        help='处理后的YOLO格式数据目录')
    parser.add_argument('--model', type=str, default='yolov11',
                        help='YOLO模型版本，例如yolov11')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--weights', type=str, default='',
                        help='预训练权重路径，空字符串表示从头开始训练')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备（GPU ID），例如0或0,1,2,3')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='tt100k',
                        help='实验名称')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return
    
    # 创建数据集配置
    data_yaml = create_dataset_yaml(args.data_dir, f"{args.project}/dataset.yaml")
    if not data_yaml:
        return
    
    # 训练模型
    train_yolo(
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        weights=args.weights,
        device=args.device,
        project=args.project,
        name=args.name
    )

if __name__ == "__main__":
    main() 