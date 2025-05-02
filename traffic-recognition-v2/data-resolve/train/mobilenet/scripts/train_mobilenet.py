#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pathlib import Path
import time

def create_model(input_shape=(224, 224, 3), num_classes=200, weights='imagenet'):
    """创建MobileNet模型"""
    # 基础模型
    base_model = MobileNetV2(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # 添加自定义层
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def prepare_data_generators(train_dir, val_dir, test_dir=None, img_size=(224, 224), batch_size=32):
    """准备数据生成器"""
    # 训练数据增强
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
    
    # 验证/测试数据只需要归一化
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # 训练数据生成器
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    # 验证数据生成器
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # 测试数据生成器（如果提供）
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    return train_generator, val_generator, test_generator

def train_model(model, base_model, train_generator, val_generator, 
                output_dir='runs/mobilenet', epochs=50, initial_epochs=10, fine_tune_epochs=40,
                learning_rate=1e-4, fine_tune_lr=1e-5):
    """训练MobileNet模型，包括初始训练和微调阶段"""
    # 创建输出目录
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 模型保存路径
    model_path = output_dir / 'best_model.h5'
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        TensorBoard(
            log_dir=str(output_dir / 'logs' / time.strftime('%Y%m%d-%H%M%S')),
            histogram_freq=1
        )
    ]
    
    print("第1阶段：训练顶层分类器")
    # 冻结基础模型
    base_model.trainable = False
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 第一阶段训练
    history1 = model.fit(
        train_generator,
        epochs=initial_epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    print("第2阶段：微调模型")
    # 解冻基础模型用于微调
    base_model.trainable = True
    
    # 使用较小的学习率重新编译
    model.compile(
        optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 微调训练
    history2 = model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=initial_epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # 保存最终模型
    model.save(str(output_dir / 'final_model.h5'))
    
    return model, history1, history2

def evaluate_model(model, test_generator):
    """评估模型性能"""
    if test_generator is None:
        print("无测试数据，跳过评估")
        return None
    
    print("评估模型性能...")
    results = model.evaluate(test_generator)
    print(f"测试损失: {results[0]:.4f}")
    print(f"测试精度: {results[1]:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='训练MobileNet模型')
    parser.add_argument('--data_dir', type=str, default='./processed_data/mobilenet',
                        help='处理后的MobileNet格式数据目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='总训练轮数')
    parser.add_argument('--initial_epochs', type=int, default=10,
                        help='初始训练轮数（基础模型冻结）')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5,
                        help='微调学习率')
    parser.add_argument('--output_dir', type=str, default='runs/mobilenet',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 设置数据路径
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # 检查数据目录
    if not train_dir.exists() or not val_dir.exists():
        print(f"错误: 训练或验证数据目录不存在：{train_dir} 或 {val_dir}")
        return
    
    # 准备数据生成器
    train_generator, val_generator, test_generator = prepare_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # 获取类别数量
    num_classes = len(train_generator.class_indices)
    print(f"检测到 {num_classes} 个类别")
    
    # 创建模型
    model, base_model = create_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=num_classes
    )
    
    # 训练模型
    model, history1, history2 = train_model(
        model=model,
        base_model=base_model,
        train_generator=train_generator,
        val_generator=val_generator,
        output_dir=args.output_dir,
        epochs=args.epochs,
        initial_epochs=args.initial_epochs,
        learning_rate=args.learning_rate,
        fine_tune_lr=args.fine_tune_lr
    )
    
    # 评估模型
    if test_generator:
        evaluate_model(model, test_generator)
    
    print(f"模型训练完成，保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 