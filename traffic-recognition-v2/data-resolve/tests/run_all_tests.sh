#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/.." || exit

echo "===== 开始运行 TT100K 数据处理全部测试 ====="
echo "测试时间: $(date)"
echo

# 准备工作：如果有visualization目录，备份它
if [ -d "./visualization" ]; then
  echo "备份现有的visualization目录..."
  BACKUP_DIR="./visualization_backup_$(date +%Y%m%d_%H%M%S)"
  mv ./visualization "$BACKUP_DIR"
  echo "备份完成: $BACKUP_DIR"
  echo
fi

# 第一步：运行数据处理测试
echo "===== 步骤 1: 运行数据处理测试 ====="
bash tests/data_processor/run_test.sh
if [ $? -ne 0 ]; then
  echo "数据处理测试失败，终止测试"
  exit 1
fi

echo

# 第二步：运行YOLO格式验证测试
echo "===== 步骤 2: 运行YOLO格式验证测试 ====="
bash tests/yolo_validation/run_test.sh --data_dir ./tests/data_processor/output/yolo --samples 3
if [ $? -ne 0 ]; then
  echo "YOLO格式验证测试失败"
  exit 1
fi

echo
echo "===== 所有测试已完成 ====="
echo "测试结果保存在相应测试目录的output子目录中"
echo "可视化结果: ./tests/yolo_validation/output/" 