#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/../.." || exit

echo "===== 开始运行 YOLO 格式验证测试 ====="
echo "测试时间: $(date)"
echo

# 设置默认参数
DATA_DIR="./processed_data/yolo"
SAMPLES=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

echo "使用数据目录: $DATA_DIR"
echo "每个分割集采样数量: $SAMPLES"
echo

# 运行验证脚本
python tests/yolo_validation/test_yolo_data.py --data_dir "$DATA_DIR" --samples "$SAMPLES"

echo
echo "===== 测试完成 ====="
echo "可视化结果保存在: $(pwd)/tests/yolo_validation/output"
echo "请查看可视化结果验证边界框和标签的准确性" 