import os
import json
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def convert_to_yolo_format(image_path, annotations, output_dir, class_mapping):
    """
    将单个图像的标注转换为YOLO格式
    
    Args:
        image_path (str): 图像文件路径
        annotations (list): 图像的标注信息
        output_dir (str): 输出目录
        class_mapping (dict): 类别映射关系
    
    Returns:
        bool: 是否成功处理
    """
    try:
        # 读取图像获取宽高
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return False
            
        height, width = img.shape[:2]
        
        # 复制图像到输出目录
        img_filename = os.path.basename(image_path)
        output_img_path = os.path.join(output_dir, 'images', img_filename)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        shutil.copy(image_path, output_img_path)
        
        # 创建对应的标签文件
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_dir, 'labels', label_filename)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for anno in annotations:
                # 获取标注信息
                category = anno['category']
                if category not in class_mapping:
                    continue
                    
                class_id = class_mapping[category]
                
                # 处理边界框，注意bbox是一个对象而非数组
                bbox = anno['bbox']
                
                # 确保bbox是对象格式，如果是对象则提取坐标
                if isinstance(bbox, dict):
                    try:
                        x1 = float(bbox.get('xmin', 0))
                        y1 = float(bbox.get('ymin', 0))
                        x2 = float(bbox.get('xmax', 0))
                        y2 = float(bbox.get('ymax', 0))
                    except (ValueError, TypeError) as e:
                        print(f"解析bbox坐标出错: {e}, bbox = {bbox}")
                        continue
                else:
                    # 如果不是对象格式，尝试作为数组处理
                    try:
                        x1 = float(bbox[0])
                        y1 = float(bbox[1])
                        x2 = float(bbox[2])
                        y2 = float(bbox[3])
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"处理边界框 {bbox} 时出错: {e}")
                        continue
                
                # 转换为中心点坐标和宽高
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                # 确保值在合理范围内
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
                    print(f"边界框坐标超出范围: x_center={x_center}, y_center={y_center}, width={box_width}, height={box_height}")
                    continue
                
                # 写入YOLO格式
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                
        return True
    
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return False

def extract_crop_for_classification(image_path, annotations, output_dir, class_mapping=None):
    """
    从图像中裁剪出标注区域，用于分类模型训练
    
    Args:
        image_path (str): 图像文件路径
        annotations (list): 图像的标注信息
        output_dir (str): 输出目录
        class_mapping (dict): 类别映射关系，None表示使用原始类别名
    
    Returns:
        int: 成功裁剪的标注数量
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
            
        count = 0
        for i, anno in enumerate(annotations):
            category = anno['category']
            
            if class_mapping is not None and category not in class_mapping:
                continue
                
            # 创建类别目录
            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)
            
            # 处理边界框，注意bbox是一个对象而非数组
            bbox = anno['bbox']
            
            # 确保bbox是对象格式，如果是对象则提取坐标
            if isinstance(bbox, dict):
                try:
                    x1 = float(bbox.get('xmin', 0))
                    y1 = float(bbox.get('ymin', 0))
                    x2 = float(bbox.get('xmax', 0))
                    y2 = float(bbox.get('ymax', 0))
                except (ValueError, TypeError) as e:
                    print(f"解析bbox坐标出错: {e}, bbox = {bbox}")
                    continue
            else:
                # 如果不是对象格式，尝试作为数组处理
                try:
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = float(bbox[2])
                    y2 = float(bbox[3])
                except (ValueError, TypeError, IndexError) as e:
                    print(f"裁剪边界框 {bbox} 时出错: {e}")
                    continue
            
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            
            # 保存裁剪区域
            img_basename = os.path.splitext(os.path.basename(image_path))[0]
            crop_filename = f"{img_basename}_{i}.jpg"
            cv2.imwrite(os.path.join(class_dir, crop_filename), crop)
            count += 1
            
        return count
    
    except Exception as e:
        print(f"裁剪图像 {image_path} 时出错: {str(e)}")
        return 0

def split_dataset(file_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        file_list (list): 文件列表
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子
    
    Returns:
        tuple: (训练集文件, 验证集文件, 测试集文件)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
    np.random.seed(seed)
    np.random.shuffle(file_list)
    
    train_size = int(len(file_list) * train_ratio)
    val_size = int(len(file_list) * val_ratio)
    
    train_files = file_list[:train_size]
    val_files = file_list[train_size:train_size+val_size]
    test_files = file_list[train_size+val_size:]
    
    return train_files, val_files, test_files 