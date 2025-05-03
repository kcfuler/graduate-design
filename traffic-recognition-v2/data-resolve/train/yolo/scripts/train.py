#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import re
from pathlib import Path

# 导入YOLO相关库
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("请先安装ultralytics库: pip install ultralytics")
    sys.exit(1)


def get_next_experiment_number(output_dir='../outputs'):
    """获取下一个可用的实验编号"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 1

    # 查找所有t-N格式的目录
    max_num = 0
    for item in output_path.iterdir():
        if item.is_dir() and re.match(r'^t-\d+$', item.name):
            try:
                num = int(item.name.split('-')[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue

    return max_num + 1


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用YOLOv11训练TT100K数据集')
    parser.add_argument('--config', type=str, default='../configs/tt100k.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='yolov11n',
                        help='模型类型: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练权重路径，默认使用COCO预训练权重')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=-1,
                        help='批次大小')
    parser.add_argument('--img-size', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--device', type=str, default='',
                        help='训练设备, 例如: cpu, 0, 0,1,2,3')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--name', type=str, default='t-1',
                        help='实验名称，格式为t-N')
    parser.add_argument('--project', type=str, default='../outputs',
                        help='保存结果的项目目录')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置文件
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 获取项目目录的绝对路径
    project_path = Path(args.project).resolve()
    project_path.mkdir(exist_ok=True, parents=True)

    # 如果使用默认实验名称t-1，自动获取下一个可用编号
    if args.name == 't-1':
        next_num = get_next_experiment_number(args.project)
        args.name = f't-{next_num}'
        print(f"自动设置实验名称为: {args.name}")

    # 创建模型
    model = YOLO(
        f"{args.model}.pt" if args.pretrained is None else args.pretrained)

    # 训练模型
    results = model.train(
        data=str(config_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        name=args.name,
        device=args.device,
        workers=args.workers,
        project=str(project_path)  # 指定保存目录
    )

    print(f"模型已保存到: {project_path / args.name}")
    print(f"最佳模型权重: {project_path / args.name / 'weights' / 'best.pt'}")


if __name__ == '__main__':
    main()
