#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载工具

用于下载交通标志识别模型的预训练权重
"""

import os
import sys
import json
import argparse
import requests
import hashlib
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil
import time

# 获取当前文件的绝对路径
CURRENT_DIR = Path(__file__).resolve().parent
# 项目根目录
ROOT_DIR = CURRENT_DIR.parent
# 配置文件路径
CONFIG_PATH = CURRENT_DIR / "config.json"
# 下载缓存目录
CACHE_DIR = CURRENT_DIR / "cache"
# 确保缓存目录存在
CACHE_DIR.mkdir(exist_ok=True)

def load_config():
    """加载模型配置"""
    if not CONFIG_PATH.exists():
        print(f"错误: 配置文件不存在 - {CONFIG_PATH}")
        return None
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def download_file(url, save_path, desc=None):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载URL
        save_path: 保存路径
        desc: 进度条描述文字
    
    Returns:
        bool: 下载是否成功
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # 下载进度条
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
        
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        
        # 检查文件大小是否匹配
        if total_size != 0 and progress_bar.n != total_size:
            print(f"警告: 下载的文件大小不匹配 - {url}")
            return False
        
        return True
    except Exception as e:
        print(f"下载文件失败 - {url}: {e}")
        # 如果文件已经存在，尝试删除
        if save_path.exists():
            save_path.unlink()
        return False

def verify_file(file_path, expected_hash=None, hash_type="sha256"):
    """
    验证文件完整性
    
    Args:
        file_path: 文件路径
        expected_hash: 预期的哈希值
        hash_type: 哈希算法类型 (md5, sha1, sha256)
    
    Returns:
        bool: 验证是否通过
    """
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return False
    
    if expected_hash is None:
        # 如果没有提供哈希值，则假设验证通过
        return True
    
    # 计算文件哈希值
    hash_func = getattr(hashlib, hash_type)()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    file_hash = hash_func.hexdigest()
    
    # 比较哈希值
    if file_hash != expected_hash:
        print(f"哈希值不匹配: {file_path}")
        print(f"预期: {expected_hash}")
        print(f"实际: {file_hash}")
        return False
    
    return True

def extract_archive(archive_path, extract_dir):
    """
    解压缩文件
    
    Args:
        archive_path: 压缩文件路径
        extract_dir: 解压目录
    
    Returns:
        bool: 解压是否成功
    """
    try:
        if str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif str(archive_path).endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif str(archive_path).endswith(".tar"):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            print(f"不支持的压缩格式: {archive_path}")
            return False
        
        return True
    except Exception as e:
        print(f"解压文件失败 - {archive_path}: {e}")
        return False

def download_mobilenet_v3():
    """下载MobileNetV3模型"""
    config = load_config()
    if not config or "models" not in config or "mobilenet_v3" not in config["models"]:
        print("找不到MobileNetV3模型配置")
        return False
    
    model_config = config["models"]["mobilenet_v3"]
    
    # 获取模型信息
    model_url = model_config.get("url")
    model_hash = model_config.get("hash")
    model_hash_type = model_config.get("hash_type", "sha256")
    model_format = model_config.get("format", "keras")
    model_file = model_config.get("file_name", "mobilenet_v3_traffic_signs.h5")
    
    if not model_url:
        print("缺少MobileNetV3模型下载URL")
        return False
    
    # 下载目标路径
    model_dir = CURRENT_DIR / "mobilenet_v3"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / model_file
    
    # 检查文件是否已存在且校验通过
    if model_path.exists() and verify_file(model_path, model_hash, model_hash_type):
        print(f"MobileNetV3模型已存在且校验通过: {model_path}")
        return True
    
    # 临时下载路径
    temp_path = CACHE_DIR / f"mobilenet_v3_{int(time.time())}"
    
    # 下载模型
    print(f"正在下载MobileNetV3模型: {model_url}")
    if not download_file(model_url, temp_path, "下载MobileNetV3"):
        return False
    
    # 验证下载的文件
    if model_hash and not verify_file(temp_path, model_hash, model_hash_type):
        print("MobileNetV3模型校验失败")
        temp_path.unlink(missing_ok=True)
        return False
    
    # 如果是压缩文件，解压
    if model_url.endswith((".zip", ".tar.gz", ".tgz", ".tar")):
        extract_dir = CACHE_DIR / f"mobilenet_v3_extract_{int(time.time())}"
        extract_dir.mkdir(exist_ok=True)
        
        print(f"正在解压MobileNetV3模型: {temp_path}")
        if not extract_archive(temp_path, extract_dir):
            temp_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return False
        
        # 在解压目录中查找模型文件
        extracted_files = list(extract_dir.glob(f"**/{model_file}"))
        if not extracted_files:
            # 如果找不到指定的文件名，尝试查找所有相关格式的文件
            extensions = [".h5", ".keras", ".pb", ".tflite"]
            for ext in extensions:
                extracted_files = list(extract_dir.glob(f"**/*{ext}"))
                if extracted_files:
                    break
        
        if not extracted_files:
            print(f"在解压文件中找不到MobileNetV3模型文件")
            temp_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return False
        
        # 使用找到的第一个文件
        first_model_file = extracted_files[0]
        # 复制到最终目标位置
        shutil.copy2(first_model_file, model_path)
        
        # 清理临时文件
        temp_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    else:
        # 直接复制到目标位置
        shutil.move(temp_path, model_path)
    
    print(f"MobileNetV3模型下载完成: {model_path}")
    return True

def download_yolov11():
    """下载YOLOv11模型"""
    config = load_config()
    if not config or "models" not in config or "yolov11" not in config["models"]:
        print("找不到YOLOv11模型配置")
        return False
    
    model_config = config["models"]["yolov11"]
    
    # 获取模型信息
    model_url = model_config.get("url")
    model_hash = model_config.get("hash")
    model_hash_type = model_config.get("hash_type", "sha256")
    model_format = model_config.get("format", "pt")
    model_file = model_config.get("file_name", "yolov11_traffic_signs.pt")
    
    if not model_url:
        print("缺少YOLOv11模型下载URL")
        return False
    
    # 下载目标路径
    model_dir = CURRENT_DIR / "yolov11"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / model_file
    
    # 检查文件是否已存在且校验通过
    if model_path.exists() and verify_file(model_path, model_hash, model_hash_type):
        print(f"YOLOv11模型已存在且校验通过: {model_path}")
        return True
    
    # 临时下载路径
    temp_path = CACHE_DIR / f"yolov11_{int(time.time())}"
    
    # 下载模型
    print(f"正在下载YOLOv11模型: {model_url}")
    if not download_file(model_url, temp_path, "下载YOLOv11"):
        return False
    
    # 验证下载的文件
    if model_hash and not verify_file(temp_path, model_hash, model_hash_type):
        print("YOLOv11模型校验失败")
        temp_path.unlink(missing_ok=True)
        return False
    
    # 如果是压缩文件，解压
    if model_url.endswith((".zip", ".tar.gz", ".tgz", ".tar")):
        extract_dir = CACHE_DIR / f"yolov11_extract_{int(time.time())}"
        extract_dir.mkdir(exist_ok=True)
        
        print(f"正在解压YOLOv11模型: {temp_path}")
        if not extract_archive(temp_path, extract_dir):
            temp_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return False
        
        # 在解压目录中查找模型文件
        extracted_files = list(extract_dir.glob(f"**/{model_file}"))
        if not extracted_files:
            # 如果找不到指定的文件名，尝试查找所有相关格式的文件
            extensions = [".pt", ".onnx", ".pth", ".weights"]
            for ext in extensions:
                extracted_files = list(extract_dir.glob(f"**/*{ext}"))
                if extracted_files:
                    break
        
        if not extracted_files:
            print(f"在解压文件中找不到YOLOv11模型文件")
            temp_path.unlink(missing_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            return False
        
        # 使用找到的第一个文件
        first_model_file = extracted_files[0]
        # 复制到最终目标位置
        shutil.copy2(first_model_file, model_path)
        
        # 清理临时文件
        temp_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    else:
        # 直接复制到目标位置
        shutil.move(temp_path, model_path)
    
    print(f"YOLOv11模型下载完成: {model_path}")
    return True

def download_available_models():
    """下载默认配置文件中的所有可用模型"""
    success = True
    
    try:
        mobilenet_result = download_mobilenet_v3()
        if not mobilenet_result:
            print("下载MobileNetV3模型失败")
            success = False
        
        yolov11_result = download_yolov11()
        if not yolov11_result:
            print("下载YOLOv11模型失败")
            success = False
        
    except Exception as e:
        print(f"下载模型时发生错误: {e}")
        success = False
    
    return success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载交通标志识别模型权重")
    parser.add_argument(
        '--model', '-m', type=str, default='all',
        choices=['all', 'mobilenet_v3', 'yolov11'],
        help='指定要下载的模型 (默认: all)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        success = download_available_models()
    elif args.model == 'mobilenet_v3':
        success = download_mobilenet_v3()
    elif args.model == 'yolov11':
        success = download_yolov11()
    else:
        print(f"未知的模型类型: {args.model}")
        success = False
    
    if success:
        print("所有指定的模型已成功下载")
        return 0
    else:
        print("下载模型时出现错误")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 