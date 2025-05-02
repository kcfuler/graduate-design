#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="tt100k-tools",
    version="0.1.0",
    description="用于处理TT100K交通标志数据集的工具",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.50.0",
        "pyyaml>=5.4.0",
        "tensorflow>=2.5.0",  # 用于MobileNet模型训练
        "ultralytics>=8.0.0", # 用于YOLO模型训练
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "process-tt100k=process_tt100k:main",
            "train-yolo=train_yolo:main",
            "train-mobilenet=train_mobilenet:main",
            "extract-weather=extract_weather_data:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 