#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class TT100KProcessor:
    def __init__(self, data_dir, output_dir):
        """
        初始化处理器
        
        Args:
            data_dir: TT100K数据集根目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.annotation_file = self.data_dir / "annotations.json"
        
        # 加载标注数据
        print("加载标注文件...")
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 获取所有类别
        self.classes = self.annotations['types']
        self.class_to_id = {cls: i for i, cls in enumerate(self.classes)}
        
        # 创建目录
        self.yolo_dir = self.output_dir / "yolo"
        self.mobilenet_dir = self.output_dir / "mobilenet"
        self.weather_dir = self.output_dir / "weather_conditions"
        
        self._create_dirs()
    
    def _create_dirs(self):
        """创建必要的目录结构"""
        for dataset in ['train', 'test', 'val']:
            # YOLO目录结构
            os.makedirs(self.yolo_dir / dataset / "images", exist_ok=True)
            os.makedirs(self.yolo_dir / dataset / "labels", exist_ok=True)
            
            # MobileNet目录结构
            for cls in self.classes:
                os.makedirs(self.mobilenet_dir / dataset / cls, exist_ok=True)
        
        # 保存类别映射
        with open(self.yolo_dir / "classes.txt", 'w') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")
        
        # 创建天气条件目录
        for condition in ['rainy', 'foggy', 'night', 'snow', 'normal']:
            os.makedirs(self.weather_dir / condition / "images", exist_ok=True)
            os.makedirs(self.weather_dir / condition / "labels", exist_ok=True)
    
    def _convert_to_yolo_format(self, img_data, img_width, img_height):
        """
        将TT100K格式的标注转换为YOLO格式
        
        YOLO格式: <class_id> <x_center> <y_center> <width> <height>
        所有值都是相对于图像尺寸的归一化值
        """
        yolo_annotations = []
        
        for obj in img_data['objects']:
            category = obj['category']
            if category not in self.class_to_id:
                continue
                
            class_id = self.class_to_id[category]
            bbox = obj['bbox']
            
            # 计算归一化的中心点和宽高
            x_min, y_min = bbox['xmin'], bbox['ymin']
            x_max, y_max = bbox['xmax'], bbox['ymax']
            
            x_center = (x_min + x_max) / (2 * img_width)
            y_center = (y_min + y_max) / (2 * img_height)
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # 确保值在[0,1]范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    def classify_weather_condition(self, img_path):
        """
        简单的天气条件分类（示例实现，实际应使用更复杂的图像处理/机器学习方法）
        
        Returns:
            str: 'rainy', 'foggy', 'night', 'snow', 或 'normal'
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return 'normal'
        
        # 转换为HSV以便更好地分析
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 简单的启发式规则（实际项目中应使用更复杂的方法）
        # 亮度平均值和标准差
        brightness = np.mean(hsv[:,:,2])
        std_brightness = np.std(hsv[:,:,2])
        
        # 饱和度平均值
        saturation = np.mean(hsv[:,:,1])
        
        # 粗略的分类（仅作示例）
        if brightness < 70:  # 较暗
            return 'night'
        elif std_brightness > 60 and saturation < 50:  # 高对比度，低饱和度
            return 'rainy'
        elif brightness > 150 and saturation < 40:  # 高亮度，低饱和度
            return 'foggy'
        elif brightness > 120 and std_brightness < 40:  # 高亮度，低对比度
            return 'snow'
        else:
            return 'normal'
    
    def process_dataset(self, dataset_type='train', val_split=0.1):
        """
        处理指定类型的数据集
        
        Args:
            dataset_type: 'train'或'test'
            val_split: 从训练集中分离出的验证集比例
        """
        # 读取图像ID列表
        id_file = self.data_dir / dataset_type / "ids.txt"
        if not id_file.exists():
            print(f"警告: {id_file} 不存在，跳过处理 {dataset_type} 数据集")
            return
        
        with open(id_file, 'r') as f:
            img_ids = f.read().splitlines()
        
        # 如果是训练集，则分出一部分作为验证集
        if dataset_type == 'train' and val_split > 0:
            np.random.shuffle(img_ids)
            val_size = int(len(img_ids) * val_split)
            val_ids = img_ids[:val_size]
            train_ids = img_ids[val_size:]
            
            self._process_image_list(val_ids, 'val')
            self._process_image_list(train_ids, 'train')
        else:
            self._process_image_list(img_ids, dataset_type)
    
    def _process_image_list(self, img_ids, target_dataset):
        """处理图像列表"""
        print(f"处理 {target_dataset} 数据集，共 {len(img_ids)} 张图像...")
        
        for img_id in tqdm(img_ids):
            img_data = self.annotations['imgs'].get(img_id)
            if not img_data:
                continue
            
            # 确定图像路径
            img_path_rel = img_data['path']
            img_path = self.data_dir / img_path_rel
            
            if not img_path.exists():
                continue
            
            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # 1. 处理YOLO格式
            yolo_annotations = self._convert_to_yolo_format(img_data, img_width, img_height)
            
            # 保存图像和标注
            img_name = img_path.name
            yolo_img_path = self.yolo_dir / target_dataset / "images" / img_name
            yolo_label_path = self.yolo_dir / target_dataset / "labels" / (img_path.stem + ".txt")
            
            # 复制图像
            shutil.copy(img_path, yolo_img_path)
            
            # 保存标注
            with open(yolo_label_path, 'w') as f:
                for anno in yolo_annotations:
                    f.write(f"{anno}\n")
            
            # 2. 处理MobileNet格式（按类别组织）
            for obj in img_data['objects']:
                category = obj['category']
                if category not in self.class_to_id:
                    continue
                
                # 为每个目标裁剪一个区域
                bbox = obj['bbox']
                x_min, y_min = int(bbox['xmin']), int(bbox['ymin'])
                x_max, y_max = int(bbox['xmax']), int(bbox['ymax'])
                
                # 确保坐标在图像范围内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_max)
                y_max = min(img_height, y_max)
                
                # 裁剪区域太小则跳过
                if x_max - x_min < 10 or y_max - y_min < 10:
                    continue
                
                # 裁剪目标
                crop = img[y_min:y_max, x_min:x_max]
                
                # 保存裁剪的目标
                crop_filename = f"{img_path.stem}_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                crop_path = self.mobilenet_dir / target_dataset / category / crop_filename
                cv2.imwrite(str(crop_path), crop)
            
            # 3. 按天气条件分类
            weather = self.classify_weather_condition(img_path)
            weather_img_path = self.weather_dir / weather / "images" / img_name
            weather_label_path = self.weather_dir / weather / "labels" / (img_path.stem + ".txt")
            
            # 复制图像
            if not os.path.exists(weather_img_path):
                shutil.copy(img_path, weather_img_path)
            
            # 保存标注
            with open(weather_label_path, 'w') as f:
                for anno in yolo_annotations:
                    f.write(f"{anno}\n")
    
    def generate_yolo_config(self):
        """生成YOLO训练配置文件"""
        config_path = self.yolo_dir / "tt100k.yaml"
        
        config_content = f"""# YOLOv11 configuration for TT100K
path: {self.yolo_dir.absolute()}  # dataset root dir
train: train/images  # train images
val: val/images  # val images
test: test/images  # test images

# Classes
nc: {len(self.classes)}  # number of classes
names: {[cls for cls in self.classes]}  # class names
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"YOLO配置文件已保存到: {config_path}")
    
    def process(self):
        """执行完整的处理流程"""
        print("开始处理TT100K数据集...")
        
        # 处理训练集和测试集
        self.process_dataset('train')
        self.process_dataset('test')
        
        # 生成YOLO配置
        self.generate_yolo_config()
        
        print("处理完成！")
        print(f"YOLO格式数据: {self.yolo_dir}")
        print(f"MobileNet格式数据: {self.mobilenet_dir}")
        print(f"按天气条件分类的数据: {self.weather_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理TT100K数据集')
    parser.add_argument('--data_dir', type=str, default='./data', help='TT100K数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./processed_data', help='输出目录')
    args = parser.parse_args()
    
    processor = TT100KProcessor(args.data_dir, args.output_dir)
    processor.process() 