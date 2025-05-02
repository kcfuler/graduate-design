#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from process.data.processor import TT100KProcessor

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试TT100K数据处理')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='TT100K数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='处理后数据的输出目录')
    parser.add_argument('--sample_count', type=int, default=100,
                        help='采样处理的图像数量')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建处理器实例
    processor = TT100KProcessor(args.data_dir, args.output_dir)
    
    # 加载标注
    annotations = processor.load_annotations()
    
    # 随机采样一部分图像ID进行处理
    image_ids = list(annotations['imgs'].keys())
    if args.sample_count < len(image_ids):
        random.seed(42)
        sampled_ids = random.sample(image_ids, args.sample_count)
    else:
        sampled_ids = image_ids
    
    print(f"采样 {len(sampled_ids)} 张图像进行测试")
    
    # 创建一个新的注释文件，只包含采样的图像
    sampled_annotations = {
        'imgs': {img_id: annotations['imgs'][img_id] for img_id in sampled_ids},
        'types': annotations['types']
    }
    
    # 保存采样的注释文件
    sampled_annotation_path = os.path.join(args.output_dir, 'sampled_annotations.json')
    with open(sampled_annotation_path, 'w') as f:
        json.dump(sampled_annotations, f)
    
    # 备份原始注释
    original_annotations = processor.annotations
    
    # 使用采样注释
    processor.annotations = sampled_annotations
    
    # 处理为YOLO格式
    processor.process_to_yolo_format()
    
    # 恢复原始注释
    processor.annotations = original_annotations
    
    print("测试完成！")

if __name__ == '__main__':
    main() 