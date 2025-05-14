#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import json
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import KMeans
import cv2

# 添加父目录到系统路径以导入模块
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TT100K简化数据处理流水线')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='TT100K原始数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='处理后数据的输出目录')
    parser.add_argument('--model', type=str, default='yolo',
                        help='模型名称，用于目录结构分类')
    parser.add_argument('--min_freq', type=int, default=100,
                        help='类别的最小频次阈值，小于此阈值的类别将被丢弃')
    parser.add_argument('--num_clusters', type=int, default=9,
                        help='anchor box聚类数量，设为0则不进行聚类')
    parser.add_argument('--selected_types', type=str, default=None,
                        help='需要处理的交通标志类型，用逗号分隔，默认处理所有符合频次阈值的类型')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    return parser.parse_args()


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


class TT100KSimpleProcessor:
    """TT100K数据集简化处理器，只根据类别频次筛选数据"""

    def __init__(self, data_dir, output_dir, min_freq=100, selected_types=None, seed=42):
        """
        初始化处理器

        Args:
            data_dir (str): 原始数据集目录
            output_dir (str): 输出目录
            min_freq (int): 类别的最小频次阈值，小于此阈值的类别将被丢弃
            selected_types (list): 需要处理的交通标志类型，None表示全部处理
            seed (int): 随机种子
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.min_freq = min_freq
        self.selected_types = selected_types
        self.seed = seed
        self.annotations = None
        self.class_frequency = None
        self.kept_classes = []  # 保留的类别
        self.class_mapping = {}  # 类别到索引的映射

        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)

    def load_annotations(self):
        """加载标注文件"""
        annotation_path = os.path.join(self.data_dir, 'annotations_all.json')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)

        print(f"加载了 {len(self.annotations['imgs'])} 张图片的标注信息")
        return self.annotations

    def analyze_class_distribution(self):
        """分析类别分布，按频次筛选"""
        if self.annotations is None:
            self.load_annotations()

        # 统计每个类别出现的频次
        class_counts = Counter()
        valid_count = 0
        total_annotations = 0

        for img_id, img_info in self.annotations['imgs'].items():
            if 'objects' not in img_info:
                continue

            for obj in img_info['objects']:
                category = obj['category']
                if self.selected_types is None or category in self.selected_types:
                    class_counts[category] += 1
                    total_annotations += 1
                    valid_count += 1

        # 保存类别频次统计
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'class_frequency.json'), 'w') as f:
            json.dump(dict(class_counts), f, indent=2)

        # 筛选频次大于阈值的类别
        self.class_frequency = class_counts
        for category, count in class_counts.items():
            if count >= self.min_freq:
                self.kept_classes.append(category)

        # 创建类别映射
        self.kept_classes.sort()  # 按字母顺序排序
        self.class_mapping = {category: idx for idx,
                              category in enumerate(self.kept_classes)}

        # 打印统计信息
        print(f"数据集共有 {len(class_counts)} 种交通标志类别, 共 {total_annotations} 个标注")
        print(f"筛选后保留 {len(self.kept_classes)} 种类别（出现频次 ≥ {self.min_freq}）")
        print(
            f"丢弃 {len(class_counts) - len(self.kept_classes)} 种类别（出现频次 < {self.min_freq}）")

        return class_counts

    def cluster_anchors(self, num_clusters=9):
        """使用K-means++对边界框进行聚类，生成更合适的anchor boxes"""
        if num_clusters <= 0:
            print("跳过anchor boxes聚类")
            return None

        if self.annotations is None:
            self.load_annotations()

        # 收集所有边界框的宽高
        bboxes = []
        for img_id, img_info in self.annotations['imgs'].items():
            if 'objects' not in img_info:
                continue

            # 获取图像尺寸
            if 'path' in img_info:
                img_path = os.path.join(self.data_dir, img_info['path'])
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                    else:
                        continue
                else:
                    continue
            else:
                continue

            for obj in img_info['objects']:
                category = obj['category']
                if category not in self.kept_classes:
                    continue

                bbox = obj['bbox']
                # 处理边界框
                if isinstance(bbox, dict):
                    x1, y1 = float(bbox.get('xmin', 0)), float(
                        bbox.get('ymin', 0))
                    x2, y2 = float(bbox.get('xmax', 0)), float(
                        bbox.get('ymax', 0))
                else:
                    try:
                        x1, y1, x2, y2 = float(bbox[0]), float(
                            bbox[1]), float(bbox[2]), float(bbox[3])
                    except:
                        continue

                # 计算相对宽高
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # 确保有效值
                if 0 < width < 1 and 0 < height < 1:
                    bboxes.append([width, height])

        if len(bboxes) == 0:
            print("未找到有效边界框，无法进行聚类")
            return None

        # 使用K-means++进行聚类
        bboxes = np.array(bboxes)
        kmeans = KMeans(n_clusters=num_clusters,
                        init='k-means++', random_state=self.seed)
        kmeans.fit(bboxes)

        # 获取聚类中心并排序
        anchors = kmeans.cluster_centers_
        anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # 按面积排序

        # 保存聚类结果
        anchor_file = os.path.join(self.output_dir, 'anchors.txt')
        with open(anchor_file, 'w') as f:
            for w, h in anchors:
                f.write(f"{w:.6f},{h:.6f} ")

        print(f"已生成 {num_clusters} 个anchor boxes并保存到 {anchor_file}")
        return anchors

    def process_dataset(self):
        """处理数据集，筛选类别并转换为YOLO格式"""
        if self.class_frequency is None:
            self.analyze_class_distribution()

        # 创建YOLO格式输出目录
        yolo_dir = os.path.join(self.output_dir, 'final')
        os.makedirs(yolo_dir, exist_ok=True)

        # 创建训练、验证、测试集目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(yolo_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(yolo_dir, split, 'labels'), exist_ok=True)

        # 保存类别映射
        with open(os.path.join(yolo_dir, 'classes.txt'), 'w') as f:
            for category in self.kept_classes:
                f.write(f"{category}\n")

        # 创建YOLO配置文件
        with open(os.path.join(yolo_dir, 'tt100k.yaml'), 'w') as f:
            f.write(f"path: {os.path.abspath(yolo_dir)}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n\n")
            f.write(f"nc: {len(self.kept_classes)}\n")
            f.write(f"names: {str(self.kept_classes)}\n")

        # 收集所有图像的路径和标注
        image_paths = []
        image_annotations = {}

        # 处理训练集和测试集图像
        for dataset_dir in ['train', 'test']:
            dir_path = os.path.join(self.data_dir, dataset_dir)
            if not os.path.exists(dir_path):
                continue

            for img_file in os.listdir(dir_path):
                if not img_file.endswith(('.jpg', '.png', '.jpeg')):
                    continue

                img_id = os.path.splitext(img_file)[0]
                if img_id not in self.annotations['imgs']:
                    continue

                img_path = os.path.join(dir_path, img_file)

                # 检查图像是否包含至少一个保留的类别
                if 'objects' in self.annotations['imgs'][img_id]:
                    has_kept_class = False
                    for obj in self.annotations['imgs'][img_id]['objects']:
                        if obj['category'] in self.kept_classes:
                            has_kept_class = True
                            break

                    if not has_kept_class:
                        continue  # 跳过没有保留类别的图像

                image_paths.append((img_path, dataset_dir))
                image_annotations[img_path] = self.annotations['imgs'][img_id]

        # 将图像分为训练集、验证集和测试集
        validation_ratio = 0.1  # 10%用于验证
        test_ratio = 0.1        # 10%用于测试

        # 洗牌以确保随机性
        random.shuffle(image_paths)

        # 分割数据集
        train_end = int(len(image_paths) * (1 - validation_ratio - test_ratio))
        val_end = int(len(image_paths) * (1 - test_ratio))

        train_paths = image_paths[:train_end]
        val_paths = image_paths[train_end:val_end]
        test_paths = image_paths[val_end:]

        # 处理并保存各数据集
        print(f"处理训练集 ({len(train_paths)} 张图像)...")
        for img_path, _ in tqdm(train_paths):
            self._process_and_save_image(
                img_path, 'train', image_annotations[img_path], yolo_dir)

        print(f"处理验证集 ({len(val_paths)} 张图像)...")
        for img_path, _ in tqdm(val_paths):
            self._process_and_save_image(
                img_path, 'val', image_annotations[img_path], yolo_dir)

        print(f"处理测试集 ({len(test_paths)} 张图像)...")
        for img_path, _ in tqdm(test_paths):
            self._process_and_save_image(
                img_path, 'test', image_annotations[img_path], yolo_dir)

        print(f"数据处理完成，数据已保存到 {yolo_dir}")
        return yolo_dir

    def _process_and_save_image(self, img_path, split, annotation, output_dir):
        """处理并保存单个图像及其标注"""
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                return

            img_height, img_width = img.shape[:2]

            # 创建目标文件路径
            img_filename = os.path.basename(img_path)
            img_name_base = os.path.splitext(img_filename)[0]

            target_img_path = os.path.join(
                output_dir, split, 'images', img_filename)
            target_label_path = os.path.join(
                output_dir, split, 'labels', f"{img_name_base}.txt")

            # 复制图像
            shutil.copy(img_path, target_img_path)

            # 处理标注
            if 'objects' in annotation:
                with open(target_label_path, 'w') as f:
                    for obj in annotation['objects']:
                        category = obj['category']
                        if category not in self.kept_classes:
                            continue  # 跳过不在保留类别中的标注

                        class_id = self.class_mapping[category]

                        # 处理边界框
                        bbox = obj['bbox']
                        if isinstance(bbox, dict):
                            x1, y1 = float(bbox.get('xmin', 0)), float(
                                bbox.get('ymin', 0))
                            x2, y2 = float(bbox.get('xmax', 0)), float(
                                bbox.get('ymax', 0))
                        else:
                            try:
                                x1, y1, x2, y2 = float(bbox[0]), float(
                                    bbox[1]), float(bbox[2]), float(bbox[3])
                            except:
                                continue

                        # 将边界框转换为YOLO格式(归一化的中心点坐标和宽高)
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        # 检查有效性
                        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width <= 0 or height <= 0:
                            continue

                        # 写入YOLO格式标注
                        f.write(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")


def create_dataset_readme(args, output_dir, train_id):
    """创建数据集说明文件"""
    readme_path = os.path.join(output_dir, "DATASET_INFO.md")

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# TT100K简化数据集\n\n")
        f.write(f"## 处理信息\n\n")
        f.write(f"- 模型: {args.model}\n")
        f.write(f"- 训练次数: {train_id}\n")
        f.write(f"- 原始数据集目录: `{args.data_dir}`\n")
        f.write(f"- 类别筛选阈值: ≥{args.min_freq}张\n")

        if args.selected_types:
            f.write(f"- 选择处理的类别: {args.selected_types}\n\n")

        if args.num_clusters > 0:
            f.write(f"- Anchor boxes聚类数量: {args.num_clusters}\n\n")

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
        f.write("本数据集通过以下步骤处理了TT100K数据集:\n\n")
        f.write(f"1. 仅保留出现频次≥{args.min_freq}的类别\n")
        f.write("2. 转换为YOLO格式以便训练\n")
        if args.num_clusters > 0:
            f.write(f"3. 生成{args.num_clusters}个适合小目标的anchor boxes\n\n")

        f.write("生成日期: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"数据集说明文件已生成: {readme_path}")


def main():
    """主处理流水线"""
    args = parse_args()

    # 处理选择的类型
    selected_types = None
    if args.selected_types:
        selected_types = args.selected_types.split(',')
        print(f"将只处理以下类型的交通标志: {selected_types}")

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

    # 创建处理器实例
    processor = TT100KSimpleProcessor(
        args.data_dir,
        train_dir,
        args.min_freq,
        selected_types,
        args.seed
    )

    # 分析类别分布
    processor.analyze_class_distribution()

    # 聚类生成anchor boxes（可选）
    if args.num_clusters > 0:
        processor.cluster_anchors(args.num_clusters)

    # 处理数据集
    final_output_dir = processor.process_dataset()

    # 创建最终数据集说明
    create_dataset_readme(args, final_output_dir, train_id)

    # 汇总处理结果
    print("\n" + "="*80)
    print("TT100K数据处理流水线已完成")
    print(f"最终数据集位置: {final_output_dir}")
    print(f"训练次数: 第{train_id}次")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
