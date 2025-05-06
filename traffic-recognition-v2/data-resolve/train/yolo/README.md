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
    ├── train_yolo.py    # 完整的训练脚本（Python API封装）
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

## 优化策略

系统支持三种优化策略，可以大幅提升模型性能：

### A1: 类重采样/Focal-Loss（长尾分布优化）

该策略针对类别不平衡问题，可提升长尾分布下的mAP 5~10%。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| --use_a1 | 启用A1优化 | - |
| --img_size | 图像尺寸 | 1280 |
| --batch_size | 批次大小 | 16 |
| --cls_pw | 类别权重参数 | 1.5-2.0 |
| --focal_loss | 使用Focal Loss | γ=2 (固定) |

使用要点：
- 当 `imgsz=1280, batch=16` 时自动启用 `mixup=0.2, copy_paste=0.1`
- 可选择调整 `cls_pw` 到 1.5-2.0 或尝试使用 Focal-Loss
- 注意：Focal Loss需要模型直接支持，在某些YOLO版本中可能并非直接可用

### A2: P2检测头 + Anchor重聚类（小目标优化）

该策略针对小目标识别，可提升小目标Recall 8~12%。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| --use_a2 | 启用A2优化 | - |
| --add_p2_head | 在Neck顶端添加P2检测头 | True |
| --anchors_num | 聚类的anchor数量 | 9-12 |

使用要点：
- 添加4×下采样特征到新检测头，提升小目标检测能力
- 使用k-means在标注框w/h上进行聚类，小anchor目标长边≈4-16像素
- 注意：P2检测头需要模型直接支持，在标准YOLO模型可能需要自定义修改

### A3: SIoU损失 & NMS IoU调节（精度优化）

该策略通过改进IoU计算方式，可提升mAP@0.5:0.95 3~5%。

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| --use_a3 | 启用A3优化 | - |
| --iou_loss | IoU损失类型 | siou |
| --iou_thres | IoU阈值 | 0.6 |
| --iou_type | IoU类型 | giou |

使用要点：
- 将box_loss设为siou，提升边界框回归精度
- 将iou_thres调至0.6，iou_type设为giou，优化NMS过程
- 标准参数iou_thres可以直接应用，其他高级设置可能需要模型支持

## 优化策略实现说明

本脚本使用Ultralytics的Python API实现优化策略：

1. 标准参数（如mixup, cls_pw等）直接通过YOLO的train函数应用
2. 高级参数会在训练前检查模型是否支持，并给出相应提示
3. 使用Python API可以直接控制训练流程，比命令行接口更灵活

注意：某些高级优化（如自定义损失函数、添加检测头等）可能需要自定义模型或使用支持这些功能的特定YOLO版本。

## 使用步骤 (SOP)

### 1. 准备数据集

确保您的数据集已经按照YOLO格式进行组织：

```
processed_data/yolo/1/final/
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

注意：processed_data使用版本管理机制，目录结构为`processed_data/模型/训练次数/final`。

### 2. 配置数据集

检查并修改`configs/tt100k.yaml`文件，确保路径和类别信息正确：

```yaml
path: ../../processed_data/yolo/1/final  # 数据集根目录（指向最新版本）
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
cd train/yolo/scripts && python train.py \
  --config ../configs/tt100k.yaml \
  --model yolov11n \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --device 0 \
  --name tt100k_yolo11n \
  --pretrained yolo11n.pt
```

#### 方式2：使用带优化策略的训练脚本

```bash
python train/yolo/scripts/train_yolo.py \
  --data_dir processed_data/yolo/1/final \
  --model yolov11n.pt \
  --epochs 100 \
  --img_size 896 \
  --batch_size 16 \
  --device 0 \
  --name t-3 \
  --use_a1 --focal_loss \
  --use_a2 --add_p2_head --anchors_num 10 \
  --use_a3 --iou_loss siou --iou_thres 0.6 --iou_type giou
```

训练脚本会使用Ultralytics的Python API，尝试应用所选的优化策略。对于标准参数会直接应用，对于高级参数会根据模型支持情况给出提示。

### 4. 测试模型

训练完成后，使用以下命令测试模型：

```bash
python train/yolo/scripts/test_model.py \
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
python train/yolo/scripts/generate_report.py \
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
6. 处理后的数据遵循版本管理机制，保存在`processed_data/模型/训练次数/final`目录结构中
   - 例如：`processed_data/yolo/1/final`表示使用yolo模型的第一次训练数据
   - 脚本现在会自动查找序号最大的版本目录（最新版本）
   - 如需使用特定版本的数据，请通过`--data_dir`参数明确指定路径
7. 某些高级优化可能需要模型直接支持，使用时请查阅YOLO文档确认兼容性 