#!/bin/bash

# TT100K 数据处理和模型训练快速启动脚本
# 此脚本会自动处理TT100K数据集，并准备用于YOLO和MobileNet的训练数据

set -e  # 遇到错误立即退出

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3 命令"
    exit 1
fi

# 检查必要的Python包
echo "检查并安装必要的Python包..."
python3 -m pip install -e .

# 默认参数
DATA_DIR="./data"
OUTPUT_DIR="./processed_data"
PROCESS_ONLY=false
TRAIN_YOLO=false
TRAIN_MOBILENET=false
EXTRACT_WEATHER=false
WEATHER_CONDITION="rainy"
CLASS_FILTER=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --process-only)
            PROCESS_ONLY=true
            shift
            ;;
        --train-yolo)
            TRAIN_YOLO=true
            shift
            ;;
        --train-mobilenet)
            TRAIN_MOBILENET=true
            shift
            ;;
        --extract-weather)
            EXTRACT_WEATHER=true
            shift
            ;;
        --weather)
            WEATHER_CONDITION="$2"
            shift 2
            ;;
        --classes)
            CLASS_FILTER="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# 处理数据
echo "开始处理 TT100K 数据..."
python3 process_tt100k.py --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR"
echo "数据处理完成!"

# 如果只需要处理数据，这里就退出
if [ "$PROCESS_ONLY" = true ]; then
    echo "仅处理数据选项已启用，处理完成"
    exit 0
fi

# 提取特定天气条件的数据
if [ "$EXTRACT_WEATHER" = true ]; then
    echo "提取 $WEATHER_CONDITION 条件下的数据..."
    if [ -z "$CLASS_FILTER" ]; then
        python3 extract_weather_data.py --data_dir "$OUTPUT_DIR/weather_conditions" --weather "$WEATHER_CONDITION" --create_yaml
    else
        python3 extract_weather_data.py --data_dir "$OUTPUT_DIR/weather_conditions" --weather "$WEATHER_CONDITION" --classes $CLASS_FILTER
    fi
    echo "天气条件数据提取完成!"
fi

# 训练YOLO模型
if [ "$TRAIN_YOLO" = true ]; then
    echo "开始训练YOLO模型..."
    python3 train_yolo.py --data_dir "$OUTPUT_DIR/yolo"
    echo "YOLO模型训练完成！"
fi

# 训练MobileNet模型
if [ "$TRAIN_MOBILENET" = true ]; then
    echo "开始训练MobileNet模型..."
    python3 train_mobilenet.py --data_dir "$OUTPUT_DIR/mobilenet"
    echo "MobileNet模型训练完成！"
fi

echo "所有任务完成！"
echo "处理后的数据位于: $OUTPUT_DIR"

# 提示如何继续
echo ""
echo "下一步操作指南:"
echo "1. 要训练YOLO模型: python3 train_yolo.py --data_dir $OUTPUT_DIR/yolo"
echo "2. 要训练MobileNet模型: python3 train_mobilenet.py --data_dir $OUTPUT_DIR/mobilenet"
echo "3. 要提取特定天气条件下的数据: python3 extract_weather_data.py --data_dir $OUTPUT_DIR/weather_conditions --weather rainy"
echo "4. 要提取特定类别的数据: python3 extract_weather_data.py --weather rainy --classes p5 p10 p23"
echo ""
echo "更多信息请查看 README.md" 