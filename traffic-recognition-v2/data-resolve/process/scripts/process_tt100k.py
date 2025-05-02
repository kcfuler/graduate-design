#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from tqdm import tqdm

# 添加父目录到系统路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.processor import TT100KProcessor
from data.utils import split_dataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理TT100K数据集')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='TT100K数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='处理后数据的输出目录')
    parser.add_argument('--selected_types', type=str, default=None,
                        help='需要处理的交通标志类型，用逗号分隔，默认处理所有类型')
    parser.add_argument('--formats', type=str, default='all',
                        help='需要生成的数据格式，可选: all, yolo, mobilenet, weather')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 处理选择的类型
    selected_types = None
    if args.selected_types:
        selected_types = args.selected_types.split(',')
        print(f"将只处理以下类型的交通标志: {selected_types}")
    
    # 创建处理器实例
    processor = TT100KProcessor(args.data_dir, args.output_dir, selected_types)
    
    # 加载标注
    processor.load_annotations()
    
    # 按所选格式处理数据
    if args.formats == 'all':
        processor.process_all()
    else:
        formats = args.formats.split(',')
        if 'yolo' in formats:
            processor.process_to_yolo_format()
        if 'mobilenet' in formats:
            processor.process_to_mobilenet_format()
        if 'weather' in formats:
            processor.process_by_weather_conditions()
    
    print("数据处理完成！")

if __name__ == '__main__':
    main() 