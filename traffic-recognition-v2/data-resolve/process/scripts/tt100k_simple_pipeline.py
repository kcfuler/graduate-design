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
    parser.add_argument('--max_images_per_class', type=int, default=300,
                        help='每个类别最多保留的图片数量')

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

    def __init__(self, data_dir, output_dir, min_freq=100, selected_types=None, seed=42, max_images_per_class=300):
        """
        初始化处理器

        Args:
            data_dir (str): 原始数据集目录
            output_dir (str): 输出目录
            min_freq (int): 类别的最小频次阈值，小于此阈值的类别将被丢弃
            selected_types (list): 需要处理的交通标志类型，None表示全部处理
            seed (int): 随机种子
            max_images_per_class (int): 每个类别最多保留的图片数量
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.min_freq = min_freq
        self.selected_types = selected_types
        self.seed = seed
        self.max_images_per_class = max_images_per_class
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
        """分析类别分布，按频次筛选，并限制每个类别的图片数量"""
        if self.annotations is None:
            self.load_annotations()

        # 统计每个类别出现的频次
        class_counts = Counter()
        # 记录包含每个类别的图片ID
        class_to_images = {cls: [] for cls in self.selected_types or []}

        for img_id, img_info in self.annotations['imgs'].items():
            if 'objects' not in img_info:
                continue

            # 记录该图片中出现的所有类别，用于后续筛选
            img_categories = set()
            for obj in img_info['objects']:
                category = obj['category']
                if self.selected_types is None or category in self.selected_types:
                    class_counts[category] += 1
                    img_categories.add(category)

            # 如果图片包含有效类别，则记录图片ID
            for category in img_categories:
                if category not in class_to_images:  # 处理 selected_types is None 的情况
                    class_to_images[category] = []
                class_to_images[category].append(img_id)

        # 保存类别频次统计
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'class_frequency.json'), 'w') as f:
            json.dump(dict(class_counts), f, indent=2)

        # 筛选频次大于阈值的类别
        self.class_frequency = class_counts
        self.kept_classes = []
        for category, count in class_counts.items():
            if count >= self.min_freq:
                self.kept_classes.append(category)

        # 创建类别映射
        self.kept_classes.sort()  # 按字母顺序排序
        self.class_mapping = {category: idx for idx,
                              category in enumerate(self.kept_classes)}

        # 打印筛选前统计信息
        print(f"数据集共有 {len(class_counts)} 种交通标志类别")
        print(
            f"筛选标准：频次 ≥ {self.min_freq}，每类最多图片数 ≤ {self.max_images_per_class}")

        # --- 新增逻辑：限制每个保留类别的图片数量 ---
        final_kept_image_ids = set()
        original_image_count = len(self.annotations['imgs'])

        for category in self.kept_classes:
            if category in class_to_images:
                images_for_category = class_to_images[category]
                # 去重，因为一张图片可能在上面被多次添加到同一个category列表
                unique_images_for_category = list(set(images_for_category))

                if len(unique_images_for_category) > self.max_images_per_class:
                    # 随机选择 N 张图片
                    selected_images = random.sample(
                        unique_images_for_category, self.max_images_per_class)
                    print(
                        f"类别 '{category}' 图片数 {len(unique_images_for_category)} > {self.max_images_per_class}，随机选取 {self.max_images_per_class} 张。")
                else:
                    selected_images = unique_images_for_category

                final_kept_image_ids.update(selected_images)

        # 根据 final_kept_image_ids 构建新的 annotations['imgs']
        new_imgs_annotation = {}
        for img_id in final_kept_image_ids:
            if img_id in self.annotations['imgs']:
                # 同时，需要确保图片中的标注对象至少有一个属于 kept_classes
                original_objects = self.annotations['imgs'][img_id].get(
                    'objects', [])
                filtered_objects = [
                    obj for obj in original_objects if obj['category'] in self.kept_classes]

                if filtered_objects:  # 只有当图片中还存在属于保留类别的标注时，才保留这张图片
                    new_img_info = self.annotations['imgs'][img_id].copy()
                    new_img_info['objects'] = filtered_objects
                    new_imgs_annotation[img_id] = new_img_info

        self.annotations['imgs'] = new_imgs_annotation

        # 更新 class_counts 和 kept_classes 基于筛选后的图片
        # 重新统计类别频次，因为某些图片可能因为其他类别的限制而被移除，导致某些最初满足 min_freq 的类别现在可能不再满足
        final_class_counts = Counter()
        for img_id, img_info in self.annotations['imgs'].items():
            if 'objects' not in img_info:
                continue
            for obj in img_info['objects']:
                category = obj['category']
                # 确保只统计在 self.kept_classes 中（即初步通过频次筛选）且在 selected_types（如果指定）中的类别
                if category in self.kept_classes and (self.selected_types is None or category in self.selected_types):
                    final_class_counts[category] += 1

        final_kept_classes = []
        for category, count in final_class_counts.items():
            if count >= self.min_freq:  # 再次检查频次，确保在图片数量限制后仍然满足
                final_kept_classes.append(category)

        self.kept_classes = sorted(final_kept_classes)
        self.class_mapping = {category: idx for idx,
                              category in enumerate(self.kept_classes)}
        self.class_frequency = final_class_counts

        # 打印统计信息
        print(
            f"经过图片数量限制和类别重新校验后，最终保留 {len(self.annotations['imgs'])} 张图片 (原为 {original_image_count} 张)")
        print(f"最终保留 {len(self.kept_classes)} 种类别。")
        if len(class_counts) - len(self.kept_classes) > 0:
            print(
                f"共丢弃 {len(class_counts) - len(self.kept_classes)} 种不符合最终标准的类别。")

        # 调试：打印最终保留的每个类别的对象数量
        # final_obj_counts_debug = Counter()
        # for img_id, img_info in self.annotations['imgs'].items():
        #     for obj in img_info.get('objects', []):
        #         if obj['category'] in self.kept_classes:
        #             final_obj_counts_debug[obj['category']] +=1
        # print(f"最终保留的对象数量（按类别）：{final_obj_counts_debug}")

        return self.class_frequency  # 返回最终的频次统计

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

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 处理selected_types参数
    selected_types_list = None
    if args.selected_types:
        selected_types_list = [s.strip()
                               for s in args.selected_types.split(',')]

    # 获取版本化的输出目录
    train_id = get_next_train_id(args.output_dir, args.model)
    versioned_output_dir = os.path.join(
        args.output_dir, args.model, str(train_id))
    os.makedirs(versioned_output_dir, exist_ok=True)

    print(f"TT100K数据集处理开始...")
    print(f"原始数据目录: {args.data_dir}")
    print(f"输出目录: {versioned_output_dir}")
    print(f"模型类型: {args.model}")
    print(f"类别最小频次: {args.min_freq}")
    print(f"每个类别最大图片数: {args.max_images_per_class}")  # 打印新参数
    if selected_types_list:
        print(f"指定处理类别: {selected_types_list}")
    print(
        f"Anchor聚类数量: {args.num_clusters if args.num_clusters > 0 else '不进行聚类'}")
    print(f"随机种子: {args.seed}")

    # 初始化处理器
    processor = TT100KSimpleProcessor(
        data_dir=args.data_dir,
        output_dir=versioned_output_dir,  # 使用版本化目录
        min_freq=args.min_freq,
        selected_types=selected_types_list,
        seed=args.seed,
        max_images_per_class=args.max_images_per_class  # 传递新参数
    )

    # 执行处理
    try:
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
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == '__main__':
    main()
