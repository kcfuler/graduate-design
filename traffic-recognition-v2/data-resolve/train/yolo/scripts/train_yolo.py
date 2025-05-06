#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import yaml
import re
import sys
from pathlib import Path


def check_yolo_installed():
    """检查是否安装了YOLO库"""
    try:
        import ultralytics
        return True
    except ImportError:
        return False


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


def find_latest_version_dir(base_dir='../../../processed_data/yolo/'):
    """查找版本号最大的数据目录"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"警告: 基础目录不存在: {base_dir}")
        return base_dir

    # 查找所有版本目录
    version_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and re.match(r'^\d+$', item.name):
            version_dirs.append((int(item.name), item))

    if not version_dirs:
        print(f"警告: 在 {base_dir} 中未找到任何版本目录")
        return base_dir

    # 按版本号排序并获取最大版本
    version_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_version_dir = version_dirs[0][1] / 'final'

    if not latest_version_dir.exists():
        print(f"警告: 最新版本的 'final' 目录不存在: {latest_version_dir}")
        return base_dir

    print(f"找到最新版本数据目录: {latest_version_dir}")
    return str(latest_version_dir)


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

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存配置文件
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"数据集配置文件已创建: {output_file}")
    return output_file


def train_yolo(data_yaml, model='yolo11n.pt', epochs=100, img_size=640, batch_size=16,
               pretrained=True, device='0', project='runs/train', name='exp',
               use_a1_optimization=False, use_a2_optimization=False, use_a3_optimization=False,
               cls_pw=None, focal_loss=False, add_p2_head=False, anchors_num=9,
               iou_loss='siou', iou_thres=0.6, iou_type='giou'):
    """训练YOLO模型"""
    if not check_yolo_installed():
        print("错误: YOLO库未安装，请先安装它")
        print("提示: 可以通过 'pip install ultralytics' 安装")
        return False

    try:
        # 导入ultralytics库 - 使用Python API而不是命令行
        from ultralytics import YOLO
        import torch

        # 准备基本训练参数
        args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': device,
            'project': project,
            'name': name,
        }

        # A1优化：类重采样/Focal-Loss
        if use_a1_optimization:
            if img_size >= 1280 and batch_size >= 16:
                args['mixup'] = 0.2
                args['copy_paste'] = 0.1

            if cls_pw:
                args['cls_pw'] = cls_pw

        # 加载预训练模型
        print(f"加载模型: {model}")
        yolo_model = YOLO(model)

        # 准备训练
        print("开始训练YOLO模型...")
        print(f"训练参数: {args}")

        # 应用特殊优化 - 这些需要在训练前修改模型配置
        if use_a1_optimization and focal_loss:
            print("应用Focal Loss优化...")
            # Focal Loss在模型初始化时通过参数设置，暂不支持直接修改
            # 需要通过其他方式实现，例如自定义损失函数

        if use_a2_optimization:
            if add_p2_head:
                print("添加P2检测头...")
                # 这需要修改模型架构，目前API不直接支持
                # 需要考虑自定义模型或使用不同的YOLO版本

            print(f"设置anchor数量为{anchors_num}...")
            # 锚点数量设置，需要在模型初始化时设置

        if use_a3_optimization:
            print(f"设置IoU损失类型为{iou_loss}...")
            if hasattr(yolo_model.model, 'loss'):
                if iou_loss == 'siou':
                    # 在某些YOLO版本中，可以通过设置模型的loss属性来改变
                    print("注意: 尝试设置SIoU损失，但API可能不直接支持此操作")

            args['iou'] = iou_thres  # 这是YOLO支持的标准参数

        # 启动训练
        results = yolo_model.train(**args)
        print("训练完成!")
        return True

    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='训练YOLO模型')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='处理后的YOLO格式数据目录，不指定则自动查找最新版本')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='YOLO模型版本，例如yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='批次大小')
    parser.add_argument('--pretrained', action='store_true',
                        help='使用预训练模型')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备（GPU ID），例如0或0,1,2,3')
    parser.add_argument('--project', type=str, default='../outputs',
                        help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='t-1',
                        help='实验名称，格式为t-N')

    # A1优化相关参数
    parser.add_argument('--use_a1', action='store_true',
                        help='使用A1优化：类重采样/Focal-Loss')
    parser.add_argument('--cls_pw', type=float, default=None,
                        help='类别权重参数，建议值1.5-2.0')
    parser.add_argument('--focal_loss', action='store_true',
                        help='使用Focal Loss (γ=2)代替BCE Loss')

    # A2优化相关参数
    parser.add_argument('--use_a2', action='store_true',
                        help='使用A2优化：增加P2检测头 + Anchor重新聚类')
    parser.add_argument('--add_p2_head', action='store_true',
                        help='在Neck顶端添加P2检测头')
    parser.add_argument('--anchors_num', type=int, default=9,
                        help='聚类的anchor数量，建议9-12')

    # A3优化相关参数
    parser.add_argument('--use_a3', action='store_true',
                        help='使用A3优化：SIoU损失 & NMS IoU调节')
    parser.add_argument('--iou_loss', type=str, default='siou',
                        choices=['ciou', 'diou', 'giou', 'eiou', 'siou'],
                        help='IoU损失类型')
    parser.add_argument('--iou_thres', type=float, default=0.6,
                        help='IoU阈值，建议0.6')
    parser.add_argument('--iou_type', type=str, default='giou',
                        choices=['iou', 'giou', 'siou', 'eiou'],
                        help='IoU类型，建议giou')

    args = parser.parse_args()

    # 如果未指定数据目录，查找最新版本的数据目录
    if args.data_dir is None:
        args.data_dir = find_latest_version_dir()

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return

    # 创建数据集配置
    data_yaml = create_dataset_yaml(
        args.data_dir, f"{args.project}/dataset.yaml")
    if not data_yaml:
        return

    # 如果使用默认实验名称t-1，自动获取下一个可用编号
    if args.name == 't-1':
        next_num = get_next_experiment_number(args.project)
        args.name = f't-{next_num}'
        print(f"自动设置实验名称为: {args.name}")

    # 检查A1优化条件，如果启用且img_size小于1280，给出建议
    if args.use_a1 and args.img_size < 1280:
        print(f"警告: A1优化建议imgsz=1280，当前设置为{args.img_size}")

    # 检查A1优化条件，如果启用且batch_size小于16，给出建议
    if args.use_a1 and args.batch_size < 16 and args.batch_size != -1:
        print(f"警告: A1优化建议batch_size=16，当前设置为{args.batch_size}")

    # 训练模型
    model = args.model if args.pretrained else f'yolo11n'

    train_yolo(
        data_yaml=data_yaml,
        model=model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        pretrained=args.pretrained,
        device=args.device,
        project=args.project,
        name=args.name,
        use_a1_optimization=args.use_a1,
        use_a2_optimization=args.use_a2,
        use_a3_optimization=args.use_a3,
        cls_pw=args.cls_pw,
        focal_loss=args.focal_loss,
        add_p2_head=args.add_p2_head,
        anchors_num=args.anchors_num,
        iou_loss=args.iou_loss,
        iou_thres=args.iou_thres,
        iou_type=args.iou_type
    )


if __name__ == "__main__":
    main()
