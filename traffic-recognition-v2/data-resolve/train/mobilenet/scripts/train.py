#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

# 导入TensorFlow相关库
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
except ImportError:
    print("请先安装TensorFlow库: pip install tensorflow")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用MobileNet训练TT100K数据集')
    parser.add_argument('--data_dir', type=str, default='../../processed_data/mobilenet',
                      help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--img_size', type=int, default=224,
                      help='图像尺寸')
    parser.add_argument('--model_dir', type=str, default='../models',
                      help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='tt100k_mobilenet',
                      help='模型名称')
    parser.add_argument('--fine_tune', action='store_true',
                      help='是否进行微调训练')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                      help='微调训练轮数')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5,
                      help='微调学习率')
    
    return parser.parse_args()

def create_model(num_classes, img_size=224):
    """创建MobileNetV2模型"""
    # 创建基础模型
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # 冻结基础模型层
    base_model.trainable = False
    
    # 添加自定义层
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def main():
    """主函数"""
    args = parse_args()
    
    # 验证数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"错误: 训练集或验证集目录不存在")
        sys.exit(1)
    
    # 设置数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # 创建数据生成器
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    
    # 获取类别数量
    num_classes = len(train_generator.class_indices)
    print(f"检测到 {num_classes} 个类别")
    
    # 保存类别映射
    class_indices = train_generator.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / f"{args.model_name}_classes.json", 'w') as f:
        json.dump(class_mapping, f)
    
    # 创建模型
    model, base_model = create_model(num_classes, args.img_size)
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            model_dir / f"{args.model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=model_dir / 'logs',
            histogram_freq=1
        )
    ]
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks
    )
    
    # 保存模型
    model.save(model_dir / f"{args.model_name}.h5")
    
    # 微调模型（可选）
    if args.fine_tune:
        print("开始微调模型...")
        
        # 解冻基础模型的最后几层
        base_model.trainable = True
        for layer in base_model.layers[:-20]:  # 保持前面的层冻结
            layer.trainable = False
            
        # 重新编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 微调训练
        fine_tune_history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=args.fine_tune_epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=callbacks
        )
        
        # 保存微调后的模型
        model.save(model_dir / f"{args.model_name}_fine_tuned.h5")
        
        # 合并训练历史
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(model_dir / f"{args.model_name}_training_history.png")
    
    print(f"模型训练完成！模型已保存到 {model_dir}")

if __name__ == '__main__':
    main() 