#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.utils import split_dataset, convert_to_yolo_format
import os
import sys
import json
import argparse
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import random
from collections import Counter
from sklearn.cluster import KMeans

# 添加父目录到系统路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdvancedTT100KProcessor:
    """改进的TT100K数据集处理器，实现基于频次分层处理"""

    def __init__(self, data_dir, output_dir, selected_types=None, min_freq_high=50, min_freq_mid=10, frequency_level='all'):
        """
        初始化处理器

        Args:
            data_dir (str): 原始数据集目录
            output_dir (str): 输出目录
            selected_types (list): 需要处理的交通标志类型，None表示全部处理
            min_freq_high (int): 高频类别的最小频次阈值（A类）
            min_freq_mid (int): 中频类别的最小频次阈值（B类）
            frequency_level (str): 输出数据集包含的频率层级，可选值：'all'、'high'、'mid'、'low'或'high,mid'等组合，用逗号分隔
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.selected_types = selected_types
        self.min_freq_high = min_freq_high
        self.min_freq_mid = min_freq_mid
        self.frequency_level = frequency_level.lower()
        self.annotations = None
        self.class_frequency = None
        self.high_freq_classes = []  # A类：高频类别 (≥min_freq_high)
        self.mid_freq_classes = []   # B类：中频类别 (min_freq_mid ~ min_freq_high-1)
        self.low_freq_classes = []   # C类：低频类别 (<min_freq_mid)
        self.class_mapping = {}      # 类别到索引的映射
        self.merged_class_mapping = {}  # 合并稀疏类别后的映射

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
        """分析类别分布，按频次进行分层"""
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

        # 分层处理
        self.class_frequency = class_counts
        for category, count in class_counts.items():
            if count >= self.min_freq_high:
                self.high_freq_classes.append(category)
            elif count >= self.min_freq_mid:
                self.mid_freq_classes.append(category)
            else:
                self.low_freq_classes.append(category)

        # 创建类别映射（包含所有类别）
        all_classes = sorted(list(class_counts.keys()))
        self.class_mapping = {category: idx for idx,
                              category in enumerate(all_classes)}

        # 创建合并稀疏类别后的映射
        merged_classes = self.high_freq_classes + \
            self.mid_freq_classes + ['unknown_rare']
        self.merged_class_mapping = {
            category: idx for idx, category in enumerate(merged_classes)}

        # 为低频类添加映射到unknown_rare
        unknown_rare_idx = self.merged_class_mapping['unknown_rare']
        for category in self.low_freq_classes:
            self.merged_class_mapping[category] = unknown_rare_idx

        # 打印统计信息
        print(f"数据集共有 {len(all_classes)} 种交通标志类别, 共 {total_annotations} 个标注")
        print(
            f"A类高频类别 (≥{self.min_freq_high}): {len(self.high_freq_classes)} 种")
        print(
            f"B类中频类别 ({self.min_freq_mid}-{self.min_freq_high-1}): {len(self.mid_freq_classes)} 种")
        print(f"C类低频类别 (<{self.min_freq_mid}): {len(self.low_freq_classes)} 种")

        return class_counts

    def cluster_anchors(self, num_clusters=9):
        """使用K-means++对边界框进行聚类，生成更合适的anchor boxes"""
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
                if self.selected_types is not None and obj['category'] not in self.selected_types:
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
                        init='k-means++', random_state=42)
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

    def process_with_stratified_sampling(self, balance_factor=3):
        """使用分层采样策略处理数据集"""
        if self.class_frequency is None:
            self.analyze_class_distribution()

        # 创建YOLO格式输出目录
        output_dir = os.path.join(self.output_dir, 'yolo_stratified')
        os.makedirs(output_dir, exist_ok=True)

        # 创建训练、验证、测试集目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(
                output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(
                output_dir, split, 'labels'), exist_ok=True)

        # 根据frequency_level参数确定要包含的类别
        selected_freq_levels = self.frequency_level.split(',')
        include_classes = []

        if 'all' in selected_freq_levels:
            include_classes = self.high_freq_classes + \
                self.mid_freq_classes + ['unknown_rare']
        else:
            if 'high' in selected_freq_levels:
                include_classes.extend(self.high_freq_classes)
            if 'mid' in selected_freq_levels:
                include_classes.extend(self.mid_freq_classes)
            if 'low' in selected_freq_levels:
                include_classes.append('unknown_rare')  # 低频类合并为unknown_rare

        # 保存类别映射
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for category in include_classes:
                f.write(f"{category}\n")

        # 创建筛选后的类别到索引的映射
        filtered_class_mapping = {category: idx for idx,
                                  category in enumerate(include_classes)}

        # 对于低频类，特殊处理映射
        if 'low' in selected_freq_levels and 'unknown_rare' in filtered_class_mapping:
            unknown_rare_idx = filtered_class_mapping['unknown_rare']
            for category in self.low_freq_classes:
                filtered_class_mapping[category] = unknown_rare_idx

        # 创建YOLO配置文件
        with open(os.path.join(output_dir, 'tt100k.yaml'), 'w') as f:
            f.write(f"path: {os.path.abspath(output_dir)}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n\n")
            f.write(f"nc: {len(include_classes)}\n")
            f.write(f"names: {str(include_classes)}\n")

        # 收集所有图像的路径和标注
        image_paths = []
        image_annotations = {}
        images_with_mid_freq = []  # 记录包含中频类别的图像
        images_with_low_freq = []  # 记录包含低频类别的图像

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

                # 判断图像是否包含选定频率层级的类别
                if 'objects' in self.annotations['imgs'][img_id]:
                    has_selected_category = False
                    has_high_freq = False
                    has_mid_freq = False
                    has_low_freq = False

                    for obj in self.annotations['imgs'][img_id]['objects']:
                        category = obj['category']
                        if category in self.high_freq_classes:
                            has_high_freq = True
                        elif category in self.mid_freq_classes:
                            has_mid_freq = True
                        else:
                            has_low_freq = True

                    # 根据frequency_level判断是否包含所需类别
                    if 'all' in selected_freq_levels:
                        has_selected_category = True
                    else:
                        if 'high' in selected_freq_levels and has_high_freq:
                            has_selected_category = True
                        if 'mid' in selected_freq_levels and has_mid_freq:
                            has_selected_category = True
                        if 'low' in selected_freq_levels and has_low_freq:
                            has_selected_category = True

                    if not has_selected_category:
                        continue  # 跳过不包含所选频率层级类别的图像

                image_paths.append((img_path, dataset_dir))
                image_annotations[img_path] = self.annotations['imgs'][img_id]

                # 记录包含中频和低频类别的图像
                if has_mid_freq:
                    images_with_mid_freq.append(img_path)
                if has_low_freq:
                    images_with_low_freq.append(img_path)

        # 将图像分为训练集、验证集和测试集
        validation_ratio = 0.1  # 10%用于验证
        test_ratio = 0.1        # 10%用于测试

        # 洗牌以确保随机性
        random.seed(42)  # 使用固定种子以确保可重复性
        random.shuffle(image_paths)

        # 分割数据集
        train_end = int(len(image_paths) * (1 - validation_ratio - test_ratio))
        val_end = int(len(image_paths) * (1 - test_ratio))

        train_paths = image_paths[:train_end]
        val_paths = image_paths[train_end:val_end]
        test_paths = image_paths[val_end:]

        # 对中频类别进行过采样
        if 'mid' in selected_freq_levels and balance_factor > 1:
            extra_samples = []
            for img_path in images_with_mid_freq:
                if img_path in [p[0] for p in train_paths]:
                    for _ in range(balance_factor - 1):
                        extra_samples.append((img_path, 'train'))

            train_paths.extend(extra_samples)
            random.shuffle(train_paths)  # 再次洗牌

        # 处理并保存各数据集
        print(f"处理训练集 ({len(train_paths)} 张图像)...")
        for img_path, original_dir in tqdm(train_paths):
            self._process_and_save_image(img_path, 'train', image_annotations[img_path],
                                         output_dir, filtered_class_mapping)

        print(f"处理验证集 ({len(val_paths)} 张图像)...")
        for img_path, original_dir in tqdm(val_paths):
            self._process_and_save_image(img_path, 'val', image_annotations[img_path],
                                         output_dir, filtered_class_mapping)

        print(f"处理测试集 ({len(test_paths)} 张图像)...")
        for img_path, original_dir in tqdm(test_paths):
            self._process_and_save_image(img_path, 'test', image_annotations[img_path],
                                         output_dir, filtered_class_mapping)

        print(f"分层处理完成，数据已保存到 {output_dir}")
        return output_dir

    def _process_and_save_image(self, img_path, split, annotation, output_dir, class_mapping):
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
                        if category not in class_mapping and category not in self.low_freq_classes:
                            continue  # 跳过不在选定频率层级的类别

                        # 对于低频类别，如果包含在选定层级中，则映射到unknown_rare
                        if category in self.low_freq_classes and 'unknown_rare' in class_mapping:
                            class_id = class_mapping['unknown_rare']
                        elif category in class_mapping:
                            class_id = class_mapping[category]
                        else:
                            continue  # 跳过没有映射的类别

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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='TT100K数据集分层处理，基于分布频次')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='TT100K原始数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='处理后数据的输出目录')
    parser.add_argument('--selected_types', type=str, default=None,
                        help='需要处理的交通标志类型，用逗号分隔，默认处理所有类型')
    parser.add_argument('--min_freq_high', type=int, default=50,
                        help='高频类别的最小频次阈值（A类）')
    parser.add_argument('--min_freq_mid', type=int, default=10,
                        help='中频类别的最小频次阈值（B类）')
    parser.add_argument('--num_clusters', type=int, default=9,
                        help='anchor box聚类数量')
    parser.add_argument('--balance_factor', type=int, default=3,
                        help='中频类别过采样的倍数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--frequency_level', type=str, default='all',
                        help='输出数据集包含的频率层级，可选值：\'all\'、\'high\'、\'mid\'、\'low\'或\'high,mid\'等组合，用逗号分隔')

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
    processor = AdvancedTT100KProcessor(
        args.data_dir,
        args.output_dir,
        selected_types,
        args.min_freq_high,
        args.min_freq_mid,
        args.frequency_level
    )

    # 分析类别分布
    processor.analyze_class_distribution()

    # 聚类生成anchor boxes
    processor.cluster_anchors(args.num_clusters)

    # 使用分层采样策略处理数据
    output_dir = processor.process_with_stratified_sampling(
        args.balance_factor)

    print(f"数据处理完成！输出目录: {output_dir}")


if __name__ == '__main__':
    main()
