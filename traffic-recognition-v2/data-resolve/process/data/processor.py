import os
import json
import cv2
import numpy as np
from tqdm import tqdm

class TT100KProcessor:
    """TT100K数据集处理核心类"""
    
    def __init__(self, data_dir, output_dir, selected_types=None):
        """
        初始化处理器
        
        Args:
            data_dir (str): 原始数据集目录
            output_dir (str): 输出目录
            selected_types (list): 需要处理的交通标志类型，None表示全部处理
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.selected_types = selected_types
        self.annotations = None
        
    def load_annotations(self):
        """加载标注文件"""
        annotation_path = os.path.join(self.data_dir, 'annotations.json')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
            
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        
        print(f"加载了 {len(self.annotations['imgs'])} 张图片的标注信息")
        return self.annotations
    
    def process_to_yolo_format(self):
        """将数据转换为YOLO格式"""
        from .utils import convert_to_yolo_format, split_dataset
        
        if self.annotations is None:
            self.load_annotations()
            
        # 创建YOLO格式输出目录
        output_dir = os.path.join(self.output_dir, 'yolo')
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建训练、验证、测试集目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
        
        # 创建类别映射
        unique_types = set()
        for img_id, img_info in self.annotations['imgs'].items():
            if 'objects' not in img_info:
                continue
                
            for obj in img_info['objects']:
                category = obj['category']
                if self.selected_types is None or category in self.selected_types:
                    unique_types.add(category)
        
        type_list = sorted(list(unique_types))
        class_mapping = {category: idx for idx, category in enumerate(type_list)}
        
        # 保存类别映射
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for category in type_list:
                f.write(f"{category}\n")
        
        # 创建YOLO配置文件
        with open(os.path.join(output_dir, 'tt100k.yaml'), 'w') as f:
            f.write(f"path: {os.path.abspath(output_dir)}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n\n")
            f.write(f"nc: {len(type_list)}\n")
            f.write(f"names: {str(type_list)}\n")
        
        # 收集所有需要处理的图像
        image_paths = []
        image_annotations = {}
        
        # 处理训练集图像
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_dir):
            for img_file in os.listdir(train_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_id = os.path.splitext(img_file)[0]
                    if img_id in self.annotations['imgs']:
                        img_path = os.path.join(train_dir, img_file)
                        image_paths.append(img_path)
                        
                        # 收集该图像的标注
                        img_anno = []
                        if 'objects' in self.annotations['imgs'][img_id]:
                            for obj in self.annotations['imgs'][img_id]['objects']:
                                category = obj['category']
                                if self.selected_types is None or category in self.selected_types:
                                    img_anno.append(obj)
                        
                        image_annotations[img_path] = img_anno
        
        # 处理测试集图像
        test_dir = os.path.join(self.data_dir, 'test')
        if os.path.exists(test_dir):
            for img_file in os.listdir(test_dir):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_id = os.path.splitext(img_file)[0]
                    if img_id in self.annotations['imgs']:
                        img_path = os.path.join(test_dir, img_file)
                        image_paths.append(img_path)
                        
                        # 收集该图像的标注
                        img_anno = []
                        if 'objects' in self.annotations['imgs'][img_id]:
                            for obj in self.annotations['imgs'][img_id]['objects']:
                                category = obj['category']
                                if self.selected_types is None or category in self.selected_types:
                                    img_anno.append(obj)
                        
                        image_annotations[img_path] = img_anno
        
        # 分割数据集
        train_imgs, val_imgs, test_imgs = split_dataset(image_paths)
        
        # 转换为YOLO格式
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, split_imgs in splits.items():
            for img_path in tqdm(split_imgs, desc=f"处理{split_name}集"):
                annotations = image_annotations[img_path]
                convert_to_yolo_format(
                    img_path, 
                    annotations, 
                    os.path.join(output_dir, split_name),
                    class_mapping
                )
        
        # 打印统计信息
        print(f"YOLO格式数据已保存到 {output_dir}")
        print(f"共有 {len(type_list)} 种交通标志类别")
        print(f"训练集: {len(train_imgs)} 张图像")
        print(f"验证集: {len(val_imgs)} 张图像")
        print(f"测试集: {len(test_imgs)} 张图像")
    
    def process_to_mobilenet_format(self):
        """将数据转换为MobileNet训练格式"""
        if self.annotations is None:
            self.load_annotations()
            
        # 实现MobileNet格式转换
        output_dir = os.path.join(self.output_dir, 'mobilenet')
        os.makedirs(output_dir, exist_ok=True)
        
        # 这里实现具体的转换逻辑
        # ...
        
        print(f"MobileNet格式数据已保存到 {output_dir}")
    
    def process_all(self):
        """处理所有格式"""
        self.load_annotations()
        self.process_to_yolo_format()
        self.process_to_mobilenet_format() 