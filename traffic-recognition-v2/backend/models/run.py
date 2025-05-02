#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一站式模型设置和测试脚本

该脚本会自动下载并测试模型，提供完整的用户体验
"""

import os
import sys
import argparse
from pathlib import Path

# 获取当前文件的绝对路径
CURRENT_DIR = Path(__file__).resolve().parent

def check_dependencies():
    """检查依赖项是否安装"""
    missing_deps = []
    
    # 检查主要依赖
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("pillow")
    
    try:
        import tqdm
    except ImportError:
        missing_deps.append("tqdm")
    
    # 检查MobileNetV3依赖
    try:
        import tensorflow
    except ImportError:
        missing_deps.append("tensorflow")
    
    # 检查YOLOv11依赖
    try:
        import torch
    except ImportError:
        missing_deps.append("torch torchvision")
    
    # 尝试检查Ultralytics（可选）
    try:
        import ultralytics
    except ImportError:
        # 只给出警告，不添加到必须安装的列表
        print("警告: ultralytics包未安装，YOLOv11模型可能无法使用部分功能")
        print("推荐安装: pip install ultralytics")
    
    return missing_deps

def run_download(model="all"):
    """
    运行模型下载
    
    Args:
        model: 要下载的模型，可以是'all', 'mobilenet_v3', 或 'yolov11'
    
    Returns:
        bool: 下载是否成功
    """
    download_script = CURRENT_DIR / "download.py"
    
    if not download_script.exists():
        print(f"错误: 下载脚本不存在 - {download_script}")
        return False
    
    print(f"\n{'='*20} 开始下载模型 {'='*20}\n")
    
    # 构建命令行参数
    cmd_args = [sys.executable, str(download_script)]
    if model != "all":
        cmd_args.extend(["--model", model])
    
    # 执行下载脚本
    return_code = os.system(" ".join(cmd_args))
    
    return return_code == 0

def run_test(model="all"):
    """
    运行模型测试
    
    Args:
        model: 要测试的模型，可以是'all', 'mobilenet_v3', 或 'yolov11'
    
    Returns:
        bool: 测试是否成功
    """
    test_script = CURRENT_DIR / "test_models.py"
    
    if not test_script.exists():
        print(f"错误: 测试脚本不存在 - {test_script}")
        return False
    
    print(f"\n{'='*20} 开始测试模型 {'='*20}\n")
    
    # 构建命令行参数
    cmd_args = [sys.executable, str(test_script)]
    if model != "all":
        cmd_args.extend(["--model", model])
    
    # 执行测试脚本
    return_code = os.system(" ".join(cmd_args))
    
    return return_code == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="一站式模型设置和测试")
    parser.add_argument(
        '--model', '-m', type=str, default='all',
        choices=['all', 'mobilenet_v3', 'yolov11'],
        help='指定要处理的模型 (默认: all)'
    )
    parser.add_argument(
        '--skip-download', '-s', action='store_true',
        help='跳过下载步骤，直接测试'
    )
    parser.add_argument(
        '--download-only', '-d', action='store_true',
        help='仅执行下载，不测试'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("交通标志识别模型一站式设置")
    print("=" * 80)
    
    # 检查依赖项
    print("\n检查依赖项...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"缺少以下依赖项: {', '.join(missing_deps)}")
        print("请使用以下命令安装:")
        print(f"pip install {' '.join(missing_deps)}")
        response = input("是否继续运行？(y/n): ")
        if response.lower() != 'y':
            return 1
    else:
        print("所有依赖项已安装")
    
    # 下载模型
    if not args.skip_download:
        if not run_download(args.model):
            print("模型下载失败")
            return 1
        print("模型下载完成")
    
    # 如果只需下载，到此结束
    if args.download_only:
        print("\n模型设置完成，跳过测试步骤")
        return 0
    
    # 测试模型
    if not run_test(args.model):
        print("模型测试失败")
        return 1
    
    print("\n" + "=" * 80)
    print("模型设置和测试完成")
    print("=" * 80)
    
    # 显示测试结果目录
    results_dir = CURRENT_DIR / "test_results"
    if results_dir.exists():
        print(f"\n查看测试结果目录: {results_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 