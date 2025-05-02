#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime

def load_results(results_dir):
    """加载训练和验证结果"""
    # 加载训练结果
    results_file = Path(results_dir) / 'results.csv'
    if not results_file.exists():
        print(f"错误: 结果文件不存在 {results_file}")
        return None
    
    # 加载配置文件
    hyp_file = Path(results_dir) / 'args.yaml'
    if hyp_file.exists():
        with open(hyp_file, 'r') as f:
            hyp = yaml.safe_load(f)
    else:
        hyp = {}
    
    # 加载验证结果
    val_results = {}
    for val_file in (Path(results_dir) / 'val').glob('*.json'):
        with open(val_file, 'r') as f:
            val_results[val_file.stem] = json.load(f)
    
    # 加载CSV结果
    results_df = pd.read_csv(results_file)
    
    return {
        'results_df': results_df,
        'hyp': hyp,
        'val_results': val_results
    }

def plot_results(results_df, output_dir):
    """绘制训练结果曲线"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 损失曲线
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['epoch'], results_df['train/box_loss'], label='train/box_loss')
    plt.plot(results_df['epoch'], results_df['train/cls_loss'], label='train/cls_loss')
    plt.plot(results_df['epoch'], results_df['train/dfl_loss'], label='train/dfl_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'train_loss.png')
    
    # mAP曲线
    plt.figure(figsize=(12, 8))
    for col in results_df.columns:
        if 'metrics/mAP' in col:
            plt.plot(results_df['epoch'], results_df[col], label=col)
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'val_map.png')
    
    # 精确率和召回率
    plt.figure(figsize=(12, 8))
    cols = [col for col in results_df.columns if 'metrics/precision' in col or 'metrics/recall' in col]
    for col in cols:
        plt.plot(results_df['epoch'], results_df[col], label=col)
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'precision_recall.png')
    
    return [output_dir / 'train_loss.png', output_dir / 'val_map.png', output_dir / 'precision_recall.png']

def generate_markdown_report(results, output_file, figure_paths=None):
    """生成Markdown格式的报告"""
    hyp = results['hyp']
    results_df = results['results_df']
    
    # 获取最后一个epoch的结果
    final_epoch = results_df.iloc[-1]
    
    # 准备报告内容
    report = f"""# YOLO模型训练测试报告

## 基本信息

- **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **训练模型**: {hyp.get('model', 'Unknown')}
- **数据集**: {hyp.get('data', 'Unknown')}
- **总训练轮次**: {int(final_epoch['epoch'])}
- **图像尺寸**: {hyp.get('imgsz', 'Unknown')}
- **批次大小**: {hyp.get('batch', 'Unknown')}

## 训练参数

```yaml
{yaml.dump(hyp, default_flow_style=False)}
```

## 训练性能

- **最终学习率**: {final_epoch.get('lr/pg0', 'N/A')}
- **训练时间**: {results_df['time'].sum() / 3600:.2f} 小时

## 训练损失

- **Box Loss**: {final_epoch.get('train/box_loss', 'N/A'):.4f}
- **Class Loss**: {final_epoch.get('train/cls_loss', 'N/A'):.4f}
- **DFL Loss**: {final_epoch.get('train/dfl_loss', 'N/A'):.4f}

## 验证指标

"""
    
    # 添加所有的mAP指标
    map_metrics = [col for col in final_epoch.index if 'metrics/mAP' in col]
    if map_metrics:
        report += "### 平均精度 (mAP)\n\n"
        for metric in map_metrics:
            metric_name = metric.replace('metrics/', '')
            report += f"- **{metric_name}**: {final_epoch.get(metric, 'N/A'):.4f}\n"
        report += "\n"
    
    # 添加精确率和召回率
    pr_metrics = [col for col in final_epoch.index if 'metrics/precision' in col or 'metrics/recall' in col]
    if pr_metrics:
        report += "### 精确率和召回率\n\n"
        for metric in pr_metrics:
            metric_name = metric.replace('metrics/', '')
            report += f"- **{metric_name}**: {final_epoch.get(metric, 'N/A'):.4f}\n"
        report += "\n"
    
    # 添加类别指标
    if 'val_results' in results and results['val_results']:
        # 获取最新的验证结果
        val_result = list(results['val_results'].values())[-1]
        if 'class_stats' in val_result:
            report += "## 各类别性能\n\n"
            report += "| 类别ID | 类别名称 | 精确率 | 召回率 | mAP@0.5 | mAP@0.5:0.95 |\n"
            report += "|--------|----------|--------|--------|---------|-------------|\n"
            
            for cls_stat in val_result['class_stats']:
                report += f"| {cls_stat['class_id']} | {cls_stat['class_name']} | {cls_stat['precision']:.4f} | {cls_stat['recall']:.4f} | {cls_stat['map50']:.4f} | {cls_stat['map']:.4f} |\n"
            
            report += "\n"
    
    # 添加混淆矩阵
    if figure_paths and len(figure_paths) > 0:
        report += "## 训练结果可视化\n\n"
        for i, fig_path in enumerate(figure_paths):
            caption = fig_path.stem.replace('_', ' ').title()
            report += f"### {caption}\n\n"
            report += f"![{caption}]({fig_path.name})\n\n"
    
    # 写入报告文件
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"报告已生成: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='生成YOLO训练测试报告')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='训练结果目录，包含results.csv文件')
    parser.add_argument('--output_dir', type=str, default='./reports',
                        help='报告输出目录')
    
    args = parser.parse_args()
    
    # 检查结果目录
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录不存在: {args.results_dir}")
        return
    
    # 加载结果
    results = load_results(args.results_dir)
    if not results:
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制结果图
    figure_paths = plot_results(results['results_df'], output_dir)
    
    # 生成报告
    model_name = Path(args.results_dir).name
    report_file = output_dir / f"{model_name}_report.md"
    generate_markdown_report(results, report_file, figure_paths)
    
    print(f"报告生成完成！")
    print(f"报告文件: {report_file}")

if __name__ == "__main__":
    main() 