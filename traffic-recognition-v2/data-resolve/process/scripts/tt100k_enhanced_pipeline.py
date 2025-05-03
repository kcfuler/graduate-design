#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TT100K增强数据处理流水线')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='TT100K原始数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='处理后数据的输出目录')
    parser.add_argument('--model', type=str, default='yolo',
                        help='模型名称，用于目录结构分类')
    parser.add_argument('--min_freq_high', type=int, default=50,
                        help='高频类别的最小频次阈值（A类）')
    parser.add_argument('--min_freq_mid', type=int, default=10,
                        help='中频类别的最小频次阈值（B类）')
    parser.add_argument('--num_clusters', type=int, default=9,
                        help='anchor box聚类数量')
    parser.add_argument('--balance_factor', type=int, default=3,
                        help='中频类别过采样的倍数')
    parser.add_argument('--mosaic_count', type=int, default=1000,
                        help='马赛克增强样本数量')
    parser.add_argument('--mixup_count', type=int, default=500,
                        help='mixup增强样本数量')
    parser.add_argument('--selected_types', type=str, default=None,
                        help='需要处理的交通标志类型，用逗号分隔，默认处理所有类型')
    parser.add_argument('--skip_step', type=int, nargs='+', default=[],
                        help='跳过的处理步骤编号，可指定多个')
    parser.add_argument('--only_step', type=int, nargs='+', default=None,
                        help='只执行指定的处理步骤编号，可指定多个')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--frequency_level', type=str, default='all',
                        help='输出数据集包含的频率层级，可选值：\'all\'、\'high\'、\'mid\'、\'low\'或\'high,mid\'等组合，用逗号分隔')

    return parser.parse_args()


def run_process(cmd, desc="执行命令"):
    """运行处理命令并打印状态"""
    print(f">>> {desc}")
    print(f"$ {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ 命令成功完成，返回码: {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 命令执行失败，返回码: {e.returncode}")
        return False


def get_next_train_id(base_dir, model_name):
    """获取下一个训练ID（训练次数）"""
    model_dir = os.path.join(base_dir, model_name)

    if not os.path.exists(model_dir):
        return 1

    existing_trains = [d for d in os.listdir(model_dir)
                       if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()]

    if not existing_trains:
        return 1

    return max(map(int, existing_trains)) + 1


def main():
    """主处理流水线"""
    args = parse_args()

    # 脚本目录
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # 处理步骤定义
    steps = [
        {
            'id': 1,
            'name': '分析类别分布并按频次分层',
            'script': 'advanced_tt100k_process.py',
            'desc': '根据类别频次对数据集进行分层处理，生成基础YOLO格式数据'
        },
        {
            'id': 2,
            'name': '应用数据增强',
            'script': 'augment_tt100k.py',
            'desc': '对处理后的数据应用马赛克和mixup等增强方法'
        }
    ]

    # 创建基础输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取下一个训练次数
    train_id = get_next_train_id(args.output_dir, args.model)

    # 构建版本化输出路径：processed_data/模型/训练次数
    model_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(model_dir, str(train_id))
    os.makedirs(train_dir, exist_ok=True)

    print(f"创建训练数据目录: {train_dir} (第{train_id}次训练)")

    # 确定要执行的步骤
    steps_to_run = []
    if args.only_step is not None:
        steps_to_run = [step for step in steps if step['id'] in args.only_step]
    else:
        steps_to_run = [step for step in steps if step['id']
                        not in args.skip_step]

    # 中间目录
    stratified_output_dir = os.path.join(train_dir, 'stratified')
    final_output_dir = os.path.join(train_dir, 'final')

    # 执行流水线
    for step in steps_to_run:
        print(f"\n{'='*80}")
        print(f"步骤 {step['id']}: {step['name']}")
        print(f"描述: {step['desc']}")
        print(f"{'-'*80}\n")

        script_path = os.path.join(scripts_dir, step['script'])

        if step['id'] == 1:
            # 第一步：分析类别分布并按频次分层
            cmd = f"python {script_path} --data_dir {args.data_dir} --output_dir {stratified_output_dir} --min_freq_high {args.min_freq_high} --min_freq_mid {args.min_freq_mid} --num_clusters {args.num_clusters} --balance_factor {args.balance_factor} --seed {args.seed} --frequency_level {args.frequency_level}"

            if args.selected_types:
                cmd += f" --selected_types {args.selected_types}"

            if not run_process(cmd, desc=step['desc']):
                print("分层处理失败，终止流水线")
                return

        elif step['id'] == 2:
            # 第二步：应用数据增强
            stratified_yolo_dir = os.path.join(
                stratified_output_dir, 'yolo_stratified')

            if not os.path.exists(stratified_yolo_dir):
                print(f"错误: 分层YOLO格式数据目录不存在: {stratified_yolo_dir}")
                print("请确保已完成步骤1，或检查输出路径")
                return

            cmd = f"python {script_path} --yolo_dir {stratified_yolo_dir} --output_dir {final_output_dir} --mosaic_count {args.mosaic_count} --mixup_count {args.mixup_count} --copy_orig --seed {args.seed} --frequency_level {args.frequency_level}"

            if not run_process(cmd, desc=step['desc']):
                print("数据增强失败，终止流水线")
                return

    # 创建最终数据集说明
    create_dataset_readme(args, final_output_dir, train_id)

    # 汇总处理结果
    print("\n" + "="*80)
    print("TT100K数据处理流水线已完成")
    print(f"最终数据集位置: {final_output_dir}")
    print(f"训练次数: 第{train_id}次")
    print("="*80 + "\n")


def create_dataset_readme(args, output_dir, train_id):
    """创建数据集说明文件"""
    readme_path = os.path.join(output_dir, "DATASET_INFO.md")

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# TT100K增强数据集\n\n")
        f.write(f"## 处理信息\n\n")
        f.write(f"- 模型: {args.model}\n")
        f.write(f"- 训练次数: {train_id}\n")
        f.write(f"- 原始数据集目录: `{args.data_dir}`\n")
        f.write(f"- 高频类别阈值(A类): ≥{args.min_freq_high}张\n")
        f.write(f"- 中频类别阈值(B类): {args.min_freq_mid}-{args.min_freq_high-1}张\n")
        f.write(f"- 低频类别(C类): <{args.min_freq_mid}张 (合并为unknown_rare类)\n")
        f.write(f"- 包含的频率层级: {args.frequency_level}\n")
        f.write(f"- Anchor boxes聚类数量: {args.num_clusters}\n")
        f.write(f"- 中频类别过采样倍数: {args.balance_factor}\n")
        f.write(f"- 马赛克增强样本数: {args.mosaic_count}\n")
        f.write(f"- Mixup增强样本数: {args.mixup_count}\n\n")

        if args.selected_types:
            f.write(f"- 选择处理的类别: {args.selected_types}\n\n")

        f.write("## 数据集结构\n\n")
        f.write("```\n")
        f.write("final/\n")
        f.write("├── classes.txt          # 类别名称文件\n")
        f.write("├── tt100k.yaml          # YOLO配置文件\n")
        f.write("├── train/               # 训练集\n")
        f.write("│   ├── images/          # 训练图像\n")
        f.write("│   └── labels/          # 训练标签\n")
        f.write("├── val/                 # 验证集\n")
        f.write("│   ├── images/          # 验证图像\n")
        f.write("│   └── labels/          # 验证标签\n")
        f.write("└── test/                # 测试集\n")
        f.write("    ├── images/          # 测试图像\n")
        f.write("    └── labels/          # 测试标签\n")
        f.write("```\n\n")

        f.write("## 使用说明\n\n")
        f.write("1. 训练YOLOv5/YOLOv8模型:\n")
        f.write("```bash\n")
        f.write(f"# YOLOv8\n")
        f.write(
            f"yolo detect train data={os.path.join(output_dir, 'tt100k.yaml')} model=yolov8n.pt epochs=200 batch=16\n\n")
        f.write(f"# YOLOv5\n")
        f.write(
            f"python train.py --data {os.path.join(output_dir, 'tt100k.yaml')} --weights yolov5s.pt --epochs 200 --batch-size 16\n")
        f.write("```\n\n")

        f.write("2. 推荐训练策略:\n")
        f.write("   - 前50轮冻结骨干网络训练\n")
        f.write("   - 接着解冻全部层训练150轮\n")
        f.write("   - 使用cosine学习率调度\n")
        f.write("   - 添加EMA (指数移动平均)\n\n")

        f.write("## 特别说明\n\n")
        f.write("本数据集通过以下步骤增强了TT100K数据集，以提高小目标检测性能:\n\n")
        f.write("1. 根据类别频次分三类处理，避免长尾问题\n")
        f.write("2. 对TT100K数据集特有的小目标重新聚类Anchor\n")
        f.write("3. 使用分层采样策略平衡类别分布\n")
        f.write("4. 应用马赛克和mixup数据增强\n\n")

        f.write("生成日期: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"数据集说明文件已生成: {readme_path}")


if __name__ == '__main__':
    main()
