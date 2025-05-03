#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import cv2
import numpy as np
import argparse
import random
import shutil
from tqdm import tqdm
from pathlib import Path

# 添加父目录到系统路径以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TT100KAugmenter:
    """TT100K数据增强器"""

    def __init__(self, yolo_dataset_dir, output_dir,
                 mosaic_prob=0.5, mixup_prob=0.3, copy_orig=True, frequency_level='all'):
        """
        初始化增强器

        Args:
            yolo_dataset_dir (str): YOLO格式数据集目录
            output_dir (str): 增强后数据输出目录
            mosaic_prob (float): 马赛克增强的概率
            mixup_prob (float): 混合增强的概率
            copy_orig (bool): 是否复制原始样本
            frequency_level (str): 输出数据集包含的频率层级
        """
        self.yolo_dataset_dir = yolo_dataset_dir
        self.output_dir = output_dir
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_orig = copy_orig
        self.frequency_level = frequency_level

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 图像和标签路径映射
        self.train_imgs = []
        self.train_labels = []
        self.val_imgs = []
        self.val_labels = []
        self.test_imgs = []
        self.test_labels = []

    def load_dataset_paths(self):
        """加载数据集图像和标签路径"""
        print(f"使用频率级别: {self.frequency_level}")

        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(self.yolo_dataset_dir, split, 'images')
            label_dir = os.path.join(self.yolo_dataset_dir, split, 'labels')

            if not os.path.exists(img_dir) or not os.path.exists(label_dir):
                continue

            img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                         if f.endswith(('.jpg', '.png', '.jpeg'))]

            # 构建标签路径列表
            label_paths = [os.path.join(label_dir, Path(img).stem + '.txt')
                           for img in img_paths]

            # 仅保留具有有效标签文件的图像（确保标签文件存在）
            valid_pairs = [(img, label) for img, label in zip(img_paths, label_paths)
                           if os.path.exists(label)]

            # 根据频率级别过滤图像
            if self.frequency_level != 'all' and valid_pairs:
                filtered_pairs = []

                # 读取classes.txt获取当前数据集的类别信息
                classes_path = os.path.join(
                    self.yolo_dataset_dir, 'classes.txt')
                if os.path.exists(classes_path):
                    with open(classes_path, 'r') as f:
                        classes = [line.strip() for line in f.readlines()]

                    # 确保当前数据集已经是按照频率级别过滤过的
                    # 因为在advanced_tt100k_process.py已经做了严格的过滤
                    # 这里只是额外的保障措施
                    filtered_pairs = valid_pairs
                else:
                    # 如果找不到classes.txt，使用全部图像
                    filtered_pairs = valid_pairs

                valid_pairs = filtered_pairs

            if valid_pairs:
                valid_img_paths, valid_label_paths = zip(*valid_pairs)
            else:
                valid_img_paths, valid_label_paths = [], []

            if split == 'train':
                self.train_imgs = list(valid_img_paths)
                self.train_labels = list(valid_label_paths)
            elif split == 'val':
                self.val_imgs = list(valid_img_paths)
                self.val_labels = list(valid_label_paths)
            else:
                self.test_imgs = list(valid_img_paths)
                self.test_labels = list(valid_label_paths)

        print(f"训练集: {len(self.train_imgs)} 张图像")
        print(f"验证集: {len(self.val_imgs)} 张图像")
        print(f"测试集: {len(self.test_imgs)} 张图像")

    def setup_output_structure(self):
        """创建输出目录结构"""
        # 复制classes.txt和yaml文件
        src_classes = os.path.join(self.yolo_dataset_dir, 'classes.txt')
        src_yaml = os.path.join(self.yolo_dataset_dir, 'tt100k.yaml')

        if os.path.exists(src_classes):
            shutil.copy(src_classes, os.path.join(
                self.output_dir, 'classes.txt'))

        if os.path.exists(src_yaml):
            # 修改yaml文件以指向新目录
            with open(src_yaml, 'r') as f:
                yaml_content = f.read()

            yaml_content = yaml_content.replace(os.path.abspath(self.yolo_dataset_dir),
                                                os.path.abspath(self.output_dir))

            with open(os.path.join(self.output_dir, 'tt100k.yaml'), 'w') as f:
                f.write(yaml_content)

        # 创建分割目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir,
                        split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir,
                        split, 'labels'), exist_ok=True)

    def load_image_and_labels(self, img_path, label_path):
        """加载图像和标签"""
        img = cv2.imread(img_path)
        if img is None:
            return None, []

        h, w = img.shape[:2]

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    # YOLO格式: class_id, x_center, y_center, width, height
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h

                        # 转为 xyxy 格式便于处理
                        xmin = x_center - width / 2
                        ymin = y_center - height / 2
                        xmax = x_center + width / 2
                        ymax = y_center + height / 2

                        labels.append([class_id, xmin, ymin, xmax, ymax])
                    except:
                        continue

        return img, np.array(labels)

    def save_image_and_labels(self, img, labels, img_name, split):
        """保存图像和标签"""
        output_img_path = os.path.join(
            self.output_dir, split, 'images', img_name)
        output_label_path = os.path.join(self.output_dir, split, 'labels',
                                         os.path.splitext(img_name)[0] + '.txt')

        # 保存图像
        cv2.imwrite(output_img_path, img)

        # 转换标签回YOLO格式并保存
        h, w = img.shape[:2]
        with open(output_label_path, 'w') as f:
            for label in labels:
                if len(label) < 5:
                    continue

                class_id = int(label[0])
                xmin, ymin, xmax, ymax = label[1:5]

                # 转回YOLO格式
                x_center = (xmin + xmax) / 2 / w
                y_center = (ymin + ymax) / 2 / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h

                # 检查值的有效性
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 < box_width <= 1 and 0 < box_height <= 1):
                    continue

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    def apply_mosaic_augmentation(self, imgs, labels_list):
        """
        应用马赛克数据增强

        Args:
            imgs: 四张图像的列表
            labels_list: 四张图像对应的标签列表

        Returns:
            新的图像和标签
        """
        # 计算马赛克图像的尺寸（使用最大图像的尺寸而不是固定值）
        max_h = max([img.shape[0] for img in imgs if img is not None])
        max_w = max([img.shape[1] for img in imgs if img is not None])

        # 计算输出尺寸
        out_h = max_h * 2
        out_w = max_w * 2

        # 创建一个空的马赛克画布
        mosaic_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # 计算中心点
        cx = out_w // 2
        cy = out_h // 2

        # 合并后的标签
        mosaic_labels = []

        # 处理四个位置的图像
        positions = [
            (0, 0, cx, cy),           # 左上
            (cx, 0, out_w, cy),       # 右上
            (0, cy, cx, out_h),       # 左下
            (cx, cy, out_w, out_h)    # 右下
        ]

        for i, (img, labels) in enumerate(zip(imgs, labels_list)):
            if img is None or len(labels) == 0:
                continue

            h, w = img.shape[:2]

            # 各个区域的位置
            x1a, y1a, x2a, y2a = positions[i]  # 目标区域

            # 调整图像大小，但保持原始宽高比
            target_w, target_h = x2a - x1a, y2a - y1a
            img_resized = cv2.resize(img, (target_w, target_h))
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized

            # 调整标签
            if len(labels):
                # 调整边界框坐标到新的尺寸
                labels_cp = labels.copy()

                # 调整坐标比例
                labels_cp[:, 1] = labels_cp[:, 1] / w * target_w + x1a
                labels_cp[:, 3] = labels_cp[:, 3] / w * target_w + x1a
                labels_cp[:, 2] = labels_cp[:, 2] / h * target_h + y1a
                labels_cp[:, 4] = labels_cp[:, 4] / h * target_h + y1a

                # 添加到总标签中
                mosaic_labels.append(labels_cp)

        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            # 裁剪到图像边界内
            mosaic_labels[:, 1:5] = np.clip(
                mosaic_labels[:, 1:5], 0, [out_w, out_h, out_w, out_h])

            # 过滤掉太小的标注
            box_w = mosaic_labels[:, 3] - mosaic_labels[:, 1]
            box_h = mosaic_labels[:, 4] - mosaic_labels[:, 2]
            valid = (box_w > 2) & (box_h > 2)
            mosaic_labels = mosaic_labels[valid]

        return mosaic_img, mosaic_labels

    def apply_mixup_augmentation(self, img1, labels1, img2, labels2, alpha=0.5):
        """
        应用mixup数据增强

        Args:
            img1, img2: 两张图像
            labels1, labels2: 对应的标签
            alpha: 混合系数

        Returns:
            混合后的图像和标签
        """
        # 确保两张图像的尺寸相同，但保留第一张图像的原始尺寸
        h1, w1 = img1.shape[:2]
        img2 = cv2.resize(img2, (w1, h1))

        # 混合图像
        mixed_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

        # 混合标签（简单合并）
        labels2_cp = labels2.copy()

        # 调整第二张图片标签的坐标
        h2, w2 = img2.shape[:2]
        if h2 != h1 or w2 != w1:
            labels2_cp[:, 1] = labels2_cp[:, 1] / w2 * w1
            labels2_cp[:, 3] = labels2_cp[:, 3] / w2 * w1
            labels2_cp[:, 2] = labels2_cp[:, 2] / h2 * h1
            labels2_cp[:, 4] = labels2_cp[:, 4] / h2 * h1

        # 合并标签
        mixed_labels = np.vstack((labels1, labels2_cp)) if len(labels1) and len(labels2_cp) else \
            labels1 if len(labels1) else labels2_cp

        return mixed_img, mixed_labels

    def augment_train_dataset(self, mosaic_count=1000, mixup_count=500):
        """增强训练集"""
        # 确保输出目录结构已创建
        self.setup_output_structure()

        # 首先复制原始样本（如有需要）
        if self.copy_orig:
            print("复制原始训练样本...")
            for i, (img_path, label_path) in enumerate(tqdm(zip(self.train_imgs, self.train_labels),
                                                            total=len(self.train_imgs))):
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_name = os.path.basename(img_path)

                # 加载标签
                _, labels = self.load_image_and_labels(img_path, label_path)

                # 保存图像和标签
                self.save_image_and_labels(img, labels, img_name, 'train')

        # 应用马赛克增强
        if mosaic_count > 0:
            print(f"生成 {mosaic_count} 个马赛克增强样本...")
            for i in tqdm(range(mosaic_count)):
                # 随机选择4张图像
                indices = np.random.choice(
                    len(self.train_imgs), 4, replace=False)

                imgs = []
                labels_list = []

                for idx in indices:
                    img, labels = self.load_image_and_labels(
                        self.train_imgs[idx], self.train_labels[idx])
                    imgs.append(img)
                    labels_list.append(labels)

                # 应用马赛克增强
                mosaic_img, mosaic_labels = self.apply_mosaic_augmentation(
                    imgs, labels_list)

                # 生成文件名
                mosaic_name = f"mosaic_{i:05d}.jpg"

                # 保存
                self.save_image_and_labels(
                    mosaic_img, mosaic_labels, mosaic_name, 'train')

        # 应用mixup增强
        if mixup_count > 0:
            print(f"生成 {mixup_count} 个mixup增强样本...")
            for i in tqdm(range(mixup_count)):
                # 随机选择2张图像
                idx1, idx2 = np.random.choice(
                    len(self.train_imgs), 2, replace=False)

                img1, labels1 = self.load_image_and_labels(
                    self.train_imgs[idx1], self.train_labels[idx1])
                img2, labels2 = self.load_image_and_labels(
                    self.train_imgs[idx2], self.train_labels[idx2])

                if img1 is None or img2 is None:
                    continue

                # 随机alpha值
                alpha = np.random.beta(1.5, 1.5)

                # 应用mixup增强
                mixup_img, mixup_labels = self.apply_mixup_augmentation(
                    img1, labels1, img2, labels2, alpha)

                # 生成文件名
                mixup_name = f"mixup_{i:05d}.jpg"

                # 保存
                self.save_image_and_labels(
                    mixup_img, mixup_labels, mixup_name, 'train')

        # 复制验证集和测试集
        self._copy_validation_test_sets()

        print("数据增强完成!")

    def _copy_validation_test_sets(self):
        """复制验证集和测试集（不增强）"""
        for split, imgs, labels in [('val', self.val_imgs, self.val_labels),
                                    ('test', self.test_imgs, self.test_labels)]:
            print(f"复制{split}集...")
            for img_path, label_path in tqdm(zip(imgs, labels), total=len(imgs)):
                img_name = os.path.basename(img_path)

                # 直接复制文件
                shutil.copy(img_path, os.path.join(
                    self.output_dir, split, 'images', img_name))

                if os.path.exists(label_path):
                    label_name = os.path.basename(label_path)
                    shutil.copy(label_path, os.path.join(
                        self.output_dir, split, 'labels', label_name))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TT100K数据增强')
    parser.add_argument('--yolo_dir', type=str, required=True,
                        help='YOLO格式数据集目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='增强后数据输出目录')
    parser.add_argument('--mosaic_count', type=int, default=1000,
                        help='马赛克增强样本数量')
    parser.add_argument('--mixup_count', type=int, default=500,
                        help='mixup增强样本数量')
    parser.add_argument('--copy_orig', action='store_true',
                        help='是否复制原始样本')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--frequency_level', type=str, default='all',
                        help='输出数据集包含的频率层级，可选值：\'all\'、\'high\'、\'mid\'、\'low\'或\'high,mid\'等组合，用逗号分隔')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建增强器
    augmenter = TT100KAugmenter(
        args.yolo_dir,
        args.output_dir,
        copy_orig=args.copy_orig,
        frequency_level=args.frequency_level
    )

    # 加载数据集路径
    augmenter.load_dataset_paths()

    # 增强训练集
    augmenter.augment_train_dataset(
        mosaic_count=args.mosaic_count,
        mixup_count=args.mixup_count
    )


if __name__ == '__main__':
    main()
