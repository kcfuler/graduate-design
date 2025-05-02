#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics import YOLO
import pandas as pd
import cv2

def test_yolo_model(model_path, data_yaml, imgsz=640, batch=16, device='cpu', conf=0.25, iou=0.7, output_dir='./results'):
    """测试YOLO模型并输出评估指标"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    if not os.path.exists(data_yaml):
        print(f"错误: 数据配置文件不存在: {data_yaml}")
        return None
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始测试模型: {model_path}")
    print(f"使用数据集: {data_yaml}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 执行评估
    results = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device, 
                         conf=conf, iou=iou, save_json=True, save_txt=True, 
                         save_conf=True, project=str(output_dir), name='val')
    
    metrics = results.results_dict
    
    # 保存指标到JSON文件
    metrics_file = output_dir / 'val' / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"评估指标已保存到: {metrics_file}")
    return metrics

def plot_confusion_matrix(metrics, class_names, output_dir):
    """绘制混淆矩阵"""
    if 'confusion_matrix' not in metrics:
        print("警告: 混淆矩阵数据不可用")
        return None
    
    conf_matrix = np.array(metrics['confusion_matrix']['matrix'])
    
    # 确保类别名称和混淆矩阵的维度一致
    if len(class_names) > conf_matrix.shape[0]:
        class_names = class_names[:conf_matrix.shape[0]]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.tight_layout()
    
    output_file = output_dir / 'confusion_matrix.png'
    plt.savefig(output_file)
    plt.close()
    print(f"混淆矩阵已保存到: {output_file}")
    return output_file

def plot_pr_curve(metrics, output_dir):
    """绘制精确率-召回率曲线"""
    if 'pr_curve' not in metrics:
        print("警告: PR曲线数据不可用")
        return None
    
    pr_data = metrics['pr_curve']
    
    plt.figure(figsize=(12, 8))
    
    # 绘制所有类别的PR曲线
    for i, class_name in enumerate(pr_data['class_names']):
        if i < len(pr_data['precision']):
            precision = pr_data['precision'][i]
            recall = pr_data['recall'][i]
            plt.plot(recall, precision, label=f'{class_name} (AP={pr_data["ap"][i]:.3f})')
    
    # 绘制mAP曲线
    if 'map50' in metrics:
        plt.axhline(y=metrics['map50'], color='r', linestyle='--', 
                    label=f'mAP@0.5={metrics["map50"]:.3f}')
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), ncol=3)
    plt.grid(True)
    plt.tight_layout()
    
    output_file = output_dir / 'pr_curve.png'
    plt.savefig(output_file)
    plt.close()
    print(f"PR曲线已保存到: {output_file}")
    return output_file

def test_on_images(model_path, test_images_dir, output_dir, conf=0.25):
    """在测试图像上运行模型进行预测"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    if not os.path.exists(test_images_dir):
        print(f"错误: 测试图像目录不存在: {test_images_dir}")
        return None
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有测试图像
    test_images = [str(p) for p in Path(test_images_dir).glob('*.jpg')]
    test_images += [str(p) for p in Path(test_images_dir).glob('*.png')]
    
    if not test_images:
        print(f"错误: 在目录中没有找到图像: {test_images_dir}")
        return None
    
    # 随机选择10张图像（如果有那么多）
    if len(test_images) > 10:
        test_images = np.random.choice(test_images, 10, replace=False)
    
    # 在测试图像上运行模型
    print(f"在{len(test_images)}张测试图像上运行模型...")
    results = model.predict(source=test_images, conf=conf, save=True, project=str(output_dir), name='predictions')
    
    print(f"预测结果已保存到: {output_dir}/predictions")
    return len(test_images)

def generate_markdown_report(metrics, class_names, figure_paths, output_file):
    """生成Markdown格式的测试报告"""
    # 准备报告内容
    report = f"""# YOLO模型测试报告

## 评估指标概览

- **mAP@0.5**: {metrics.get('map50', 'N/A'):.4f}
- **mAP@0.5:0.95**: {metrics.get('map', 'N/A'):.4f}
- **精确率**: {metrics.get('precision', 'N/A'):.4f}
- **召回率**: {metrics.get('recall', 'N/A'):.4f}
- **F1值**: {metrics.get('f1', 'N/A'):.4f}

## 测试参数

- **置信度阈值**: {metrics.get('conf', 'N/A')}
- **IoU阈值**: {metrics.get('iou', 'N/A')}
- **图像尺寸**: {metrics.get('imgsz', 'N/A')}

## 类别性能

| 类别ID | 类别名称 | 精确率 | 召回率 | AP@0.5 | AP@0.5:0.95 |
|--------|----------|--------|--------|--------|-------------|
"""
    
    # 添加各类别的性能
    if 'class_stats' in metrics:
        for cls_stat in metrics['class_stats']:
            report += f"| {cls_stat['class_id']} | {cls_stat['class_name']} | {cls_stat['precision']:.4f} | {cls_stat['recall']:.4f} | {cls_stat['ap50']:.4f} | {cls_stat['ap']:.4f} |\n"
    
    # 添加图表
    if figure_paths:
        report += "\n## 测试结果可视化\n\n"
        for name, path in figure_paths.items():
            if path:
                report += f"### {name}\n\n"
                report += f"![{name}]({Path(path).name})\n\n"
    
    # 写入报告文件
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"测试报告已生成: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='测试YOLO模型并生成报告')
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径，.pt文件')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径，.yaml文件')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cpu',
                        help='使用的设备，cpu或cuda设备ID')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU阈值')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='输出目录')
    parser.add_argument('--test_images', type=str, default='',
                        help='测试图像目录（可选）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载类别名称
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', [])
    
    # 测试模型
    metrics = test_yolo_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        output_dir=args.output_dir
    )
    
    if not metrics:
        return
    
    # 生成可视化图表
    figure_paths = {
        '混淆矩阵': plot_confusion_matrix(metrics, class_names, output_dir),
        'PR曲线': plot_pr_curve(metrics, output_dir)
    }
    
    # 如果提供了测试图像目录，则在图像上进行测试
    if args.test_images and os.path.exists(args.test_images):
        test_on_images(args.model, args.test_images, output_dir, args.conf)
    
    # 生成报告
    report_file = output_dir / 'test_report.md'
    generate_markdown_report(metrics, class_names, figure_paths, report_file)
    
    print(f"测试完成！报告已保存到：{report_file}")

if __name__ == "__main__":
    main() 