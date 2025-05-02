#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/../.." || exit

echo "===== 开始运行 TT100K 数据处理测试 ====="
echo "测试时间: $(date)"
echo

# 运行测试脚本，处理少量样本数据
python tests/data_processor/test_processor.py --sample_count 50

echo
echo "===== 测试完成 ====="
echo "输出目录: $(pwd)/tests/data_processor/output"
echo "请查看测试结果并验证数据处理的正确性" 