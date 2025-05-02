import os
import cv2
import numpy as np
from tqdm import tqdm

def classify_weather(image_path):
    """
    简单的天气条件分类函数
    
    Args:
        image_path (str): 图像文件路径
    
    Returns:
        str: 天气类型，'normal', 'rainy', 'foggy', 'night', 'snow' 之一
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 'unknown'
            
        # 转换为HSV空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 提取基本特征
        brightness = np.mean(hsv[:, :, 2])  # 亮度
        saturation = np.mean(hsv[:, :, 1])  # 饱和度
        
        # 灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算图像对比度
        contrast = np.std(gray)
        
        # 计算图像模糊程度
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        # 夜间检测：亮度低，对比度较高
        if brightness < 70:
            return 'night'
            
        # 雾天检测：低对比度，亮度中等到较高，饱和度低
        if contrast < 30 and brightness > 100 and saturation < 40 and blur_score < 100:
            return 'foggy'
            
        # 雨天检测：中等亮度，饱和度低
        if brightness > 70 and brightness < 120 and saturation < 50:
            # 检测雨线（简化）
            edges = cv2.Canny(gray, 50, 150)
            # 竖直线比例较高可能是雨
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=20, maxLineGap=5)
            if lines is not None:
                vertical_count = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < 10:  # 竖直线
                        vertical_count += 1
                
                if vertical_count > len(lines) * 0.4:
                    return 'rainy'
        
        # 雪天检测：高亮度，低饱和度，高对比度斑点
        if brightness > 120 and saturation < 40:
            # 二值化检测白色斑点
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(thresh == 255) / thresh.size
            if white_ratio > 0.3:
                return 'snow'
                
        # 默认为正常天气
        return 'normal'
        
    except Exception as e:
        print(f"分类图像 {image_path} 时出错: {str(e)}")
        return 'unknown'

def process_images_by_weather(image_paths, output_dir, copy_files=True):
    """
    按天气条件处理一组图像
    
    Args:
        image_paths (list): 图像路径列表
        output_dir (str): 输出目录
        copy_files (bool): 是否复制图像文件
    
    Returns:
        dict: 各天气类型的图像数量统计
    """
    # 创建输出目录
    weather_types = ['normal', 'rainy', 'foggy', 'night', 'snow', 'unknown']
    for weather in weather_types:
        os.makedirs(os.path.join(output_dir, weather), exist_ok=True)
    
    # 处理图像
    stats = {weather: 0 for weather in weather_types}
    
    for img_path in tqdm(image_paths, desc="按天气分类"):
        weather = classify_weather(img_path)
        stats[weather] += 1
        
        if copy_files:
            import shutil
            # 复制文件到对应目录
            dst_path = os.path.join(output_dir, weather, os.path.basename(img_path))
            shutil.copy(img_path, dst_path)
    
    return stats 