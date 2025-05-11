#!/bin/bash
# 训练命令示例脚本
# 此脚本展示了如何使用配置文件和参数注入启动训练

# 设置环境变量（可选）
export CUDA_VISIBLE_DEVICES=0  # 使用的GPU ID

# 基本训练命令
python train.py \
  --data configs/data.yaml \
  --cfg configs/yolo11s.yaml \
  --data_path /path/to/dataset \
  --epochs 50 \
  --batch_size 16 \
  --imgsz 640 \
  --lr0 0.01

# 高级训练示例（取消注释以使用）
# python train.py \
#   --data configs/data.yaml \
#   --cfg configs/yolo11s.yaml \
#   --data_path /path/to/dataset \
#   --epochs 100 \
#   --batch_size 32 \
#   --imgsz 640 \
#   --lr0 0.005 \
#   --weight_decay 0.0001 \
#   --momentum 0.95 \
#   --warmup_epochs 5 \
#   --workers 8 \
#   --mosaic 0.8 \
#   --mixup 0.1

# 使用预训练权重
# python train.py \
#   --data configs/data.yaml \
#   --cfg configs/yolo11s.yaml \
#   --data_path /path/to/dataset \
#   --weights /path/to/pretrained/weights.pt \
#   --epochs 30 \
#   --batch_size 16

# 分布式训练示例
# python -m torch.distributed.run --nproc_per_node=2 train.py \
#   --data configs/data.yaml \
#   --cfg configs/yolo11s.yaml \
#   --data_path /path/to/dataset \
#   --batch_size 16 \
#   --epochs 50 \
#   --sync_bn  # 同步批量归一化 