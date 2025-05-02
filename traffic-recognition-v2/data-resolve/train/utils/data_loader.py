#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

def load_yolo_dataset(data_dir, img_size=640, augment=True):
    """
    加载YOLO格式的数据集
    
    Args:
        data_dir (str): 数据目录
        img_size (int): 图像尺寸
        augment (bool): 是否进行数据增强
    
    Returns:
        tuple: (images, labels)
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"图像或标签目录不存在: {images_dir} 或 {labels_dir}")
    
    # 获取所有图像文件
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"找到 {len(image_files)} 张图像")
    
    images = []
    labels = []
    
    for img_file in tqdm(image_files, desc="加载数据集"):
        # 读取图像并调整大小
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"无法读取图像: {img_file}")
            continue
            
        # 调整图像大小
        img = cv2.resize(img, (img_size, img_size))
        
        # 加载对应的标签文件
        label_file = labels_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            print(f"标签文件不存在: {label_file}")
            continue
            
        # 读取标签
        with open(label_file, 'r') as f:
            label_lines = f.read().splitlines()
            
        if not label_lines:
            continue
            
        # 解析标签
        label_data = []
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            label_data.append([cls_id, x_center, y_center, width, height])
        
        if not label_data:
            continue
            
        images.append(img)
        labels.append(np.array(label_data))
        
        # 数据增强
        if augment:
            # 水平翻转
            flipped_img = cv2.flip(img, 1)
            flipped_labels = []
            for label in label_data:
                cls_id, x_center, y_center, width, height = label
                # 翻转后的x坐标
                flipped_x_center = 1.0 - x_center
                flipped_labels.append([cls_id, flipped_x_center, y_center, width, height])
                
            images.append(flipped_img)
            labels.append(np.array(flipped_labels))
    
    return np.array(images), labels

def load_classification_dataset(data_dir, img_size=224):
    """
    加载分类模型的数据集
    
    Args:
        data_dir (str): 数据目录
        img_size (int): 图像尺寸
    
    Returns:
        tuple: (images, labels, class_names)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 获取所有类别
    class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
    class_names.sort()
    
    if not class_names:
        raise ValueError(f"在 {data_dir} 中未找到任何类别")
        
    print(f"找到 {len(class_names)} 个类别")
    
    images = []
    labels = []
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = data_dir / cls_name
        image_files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        
        print(f"类别 {cls_name}: {len(image_files)} 张图像")
        
        for img_file in tqdm(image_files, desc=f"加载 {cls_name}"):
            # 读取图像并调整大小
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            # 调整图像大小
            img = cv2.resize(img, (img_size, img_size))
            
            images.append(img)
            labels.append(cls_idx)
    
    return np.array(images), np.array(labels), class_names

def split_dataset(images, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    分割数据集
    
    Args:
        images (np.ndarray): 图像数组
        labels (np.ndarray): 标签数组
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子
    
    Returns:
        tuple: (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    np.random.seed(seed)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    
    val_images = images[val_indices]
    val_labels = labels[val_indices]
    
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels 