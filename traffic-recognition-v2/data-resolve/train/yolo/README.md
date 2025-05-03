# YOLO训练系统

这个目录包含了使用YOLOv11训练交通标志识别模型的完整工作流程。

## 目录结构

```
train/yolo/
│
├── configs/              # 配置文件目录
│   └── tt100k.yaml      # TT100K数据集配置文件
│
├── outputs/              # 训练输出目录
│   ├── t-1/             # 训练实验结果
│   └── t-2/             # 训练实验结果
│
└── scripts/              # 训练和评估脚本
    ├── train.py         # 简化的训练脚本
    ├── train_yolo.py    # 完整的训练脚本（命令行工具封装）
    ├── test_model.py    # 模型测试和评估脚本
    ├── generate_report.py # 训练报告生成脚本
    └── yolo11n.pt       # YOLOv11-nano预训练模型
```

## 训练参数

YOLO训练系统支持以下主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| model | 模型类型 | yolov11n |
| epochs | 训练轮数 | 100 |
| batch-size | 批次大小 | 16 |
| img-size | 图像尺寸 | 640 |
| device | 训练设备 | GPU (0) |
| workers | 数据加载线程数 | 8 |
| name | 实验名称 | tt100k_yolov11 |
| pretrained | 预训练权重路径 | None (使用COCO预训练) |

## 使用步骤 (SOP)

### 1. 准备数据集

确保您的数据集已经按照YOLO格式进行组织：

```
processed_data/final/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── classes.txt     # 类别名称列表
```

### 2. 配置数据集

检查并修改`configs/tt100k.yaml`文件，确保路径和类别信息正确：

```yaml
path: ../../processed_data/final  # 数据集根目录
train: train/images  # 训练图像
val: val/images  # 验证图像
test: test/images  # 测试图像
nc: 0  # 类别数量（将在训练时自动填充）
names: []  # 类别名称（将在训练时自动填充）
```

### 3. 训练模型

有两种方式可以训练模型：

#### 方式1：使用简化的训练脚本

```bash
bun train/yolo/scripts/train.py \
  --config train/yolo/configs/tt100k.yaml \
  --model yolov11n \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --device 0 \
  --name tt100k_yolov11
```

#### 方式2：使用完整的训练脚本（命令行工具封装）

```bash
bun train/yolo/scripts/train_yolo.py \
  --data_dir ./processed_data/yolo \
  --model yolo11n.pt \
  --epochs 100 \
  --img_size 640 \
  --batch_size 16 \
  --device 0 \
  --project runs/train \
  --name tt100k
```

### 4. 测试模型

训练完成后，使用以下命令测试模型：

```bash
bun train/yolo/scripts/test_model.py \
  --model runs/train/tt100k/weights/best.pt \
  --data train/yolo/configs/tt100k.yaml \
  --imgsz 640 \
  --batch 16 \
  --conf 0.25 \
  --iou 0.7 \
  --output_dir ./results
```

测试脚本会生成评估指标、混淆矩阵、PR曲线等结果。

### 5. 生成报告

使用以下命令生成详细的训练和测试报告：

```bash
bun train/yolo/scripts/generate_report.py \
  --results_dir runs/train/tt100k \
  --output_dir ./reports
```

报告将包含训练过程中的损失曲线、验证指标、类别性能等信息。

## 输出目录结构

训练完成后，输出目录将包含以下内容：

```
runs/train/experiment_name/
│
├── weights/
│   ├── best.pt      # 验证集上性能最佳的模型
│   └── last.pt      # 最后一个epoch的模型
│
├── args.yaml        # 训练参数记录
├── results.csv      # 训练过程指标记录
├── confusion_matrix.png  # 混淆矩阵
│
└── val/
    ├── PR_curve.png # 精确率-召回率曲线
    ├── metrics.json # 详细评估指标
    └── ...          # 其他评估结果
```

## 注意事项

1. 确保已安装`ultralytics`包：`pip install ultralytics`
2. 推荐使用GPU进行训练以获得更好的性能
3. 对于大型数据集，请调整`batch-size`以适应您的GPU内存
4. 使用`--pretrained`参数可以利用预训练权重加速收敛
5. 模型训练结果将保存在`runs/train/`目录下 