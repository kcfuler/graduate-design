# YOLOv11配置文件使用指南

本目录包含YOLOv11训练和推理所需的配置文件，采用标准化YAML格式设计，支持灵活的参数注入和模型定制。

## 配置文件类型

主要包含两类配置文件：

1. **数据集配置（如data.yaml）**: 定义数据集路径、训练/验证/测试集划分、类别信息等
2. **模型配置（如yolo11s.yaml）**: 定义模型结构、优化器参数、训练超参数等

## 变量占位符

配置文件中使用`${变量名:默认值}`格式的占位符实现动态参数注入，例如：

```yaml
lr0: ${lr0:0.01}  # 初始学习率，默认值0.01
```

占位符值可通过以下方式注入：

1. 环境变量：`export lr0=0.005`
2. 命令行参数：`--lr0 0.005`

## 如何扩展配置

### 添加新的数据集配置

1. 复制`data.yaml`为新文件（如`coco.yaml`）
2. 修改数据路径和类别信息
3. 确保`nc`值与`names`列表长度一致

示例：
```yaml
path: ${data_path:/datasets/coco}
nc: 80  # COCO数据集有80个类别
names: ['person', 'bicycle', 'car', ...]  # 所有80个类别名称
```

### 创建新的模型变体

1. 复制现有模型配置（如`yolo11s.yaml`）
2. 修改模型架构（backbone/head部分）
3. 调整超参数以适应新模型

示例 - 创建更轻量级模型：
```yaml
# 更小的backbone
backbone:
  - [Conv, 3, 12, 1, 1]  # 减少初始通道数
  - [Conv, 12, 24, 3, 2]
  # ...其他层
```

## 验证配置文件

使用验证脚本检查配置文件是否符合规范：

```bash
python validate_configs.py configs/data.yaml data
python validate_configs.py configs/yolo11s.yaml model
```

## 命令示例

使用配置文件启动训练：

```bash
python train.py \
  --data configs/data.yaml \
  --cfg configs/yolo11s.yaml \
  --data_path /path/to/dataset \
  --epochs 100 \
  --batch_size 64 \
  --lr0 0.01
```

## 配置文件最佳实践

1. 对于团队共享的配置，尽量使用相对路径
2. 为所有关键参数提供合理的默认值
3. 保持配置文件简洁，使用注释说明参数用途
4. 针对特定硬件环境的参数（如batch_size）应使用占位符，便于不同设备适配

# 配置文件说明

本目录包含YOLO训练系统的配置文件，用于定义数据集路径、模型参数和训练优化策略。

## 数据集配置

`tt100k.yaml`: TT100K交通标志数据集配置，包含数据路径和类别信息。

## 模型配置

`yolo11s.yaml`: YOLOv11s模型架构配置，定义了backbone、head和超参数。

## 优化策略配置

系统提供了三种专门的优化策略配置文件，针对不同场景下的模型训练：

### 1. A1优化策略: 类重采样/Focal-Loss（长尾分布优化）

文件: `a1_class_resampling.yaml`

该策略针对类别不平衡问题，可提升长尾分布下的mAP 5~10%。主要参数包括：
- `mixup`: 混合增强比例
- `copy_paste`: 复制粘贴增强比例
- `cls_pw`: 类别权重参数
- `focal_loss`: 是否使用Focal Loss

使用场景：
- 类别不平衡严重的数据集
- 类别数量多且分布长尾的情况

### 2. A2优化策略: P2检测头 + Anchor重聚类（小目标优化）

文件: `a2_small_object.yaml`

该策略针对小目标识别，可提升小目标Recall 8~12%。主要参数包括：
- `add_p2_head`: 是否添加P2检测头
- `anchors_num`: 聚类的anchor数量
- `mosaic`: 镶嵌增强比例
- `scale`: 缩放增强

使用场景：
- 小目标检测场景
- 目标尺寸差异大的场景

### 3. A3优化策略: SIoU损失 & NMS IoU调节（精度优化）

文件: `a3_iou_optimization.yaml`

该策略通过改进IoU计算方式，可提升mAP@0.5:0.95 3~5%。主要参数包括：
- `iou_loss`: IoU损失类型
- `iou_thres`: IoU阈值
- `iou_type`: IoU计算类型
- `conf_thres`: 置信度阈值
- `nms_agnostic`: 是否使用类别无关的NMS

使用场景：
- 需要高精度检测的场景
- 目标形状复杂或旋转目标较多的场景

## 使用方法

每个策略配置文件都包含三种预设方案：标准方案、轻量方案和针对特定场景的极致方案。

使用示例：

```bash
# 使用A1优化策略训练
python train/yolo/scripts/train_yolo.py --config configs/a1_class_resampling.yaml

# 使用A2优化策略训练
python train/yolo/scripts/train_yolo.py --config configs/a2_small_object.yaml

# 使用A3优化策略训练
python train/yolo/scripts/train_yolo.py --config configs/a3_iou_optimization.yaml

# 使用A1优化策略的极致方案
python train/yolo/scripts/train_yolo.py --config configs/a1_class_resampling.yaml --preset extreme
```

您可以根据数据集特点和任务需求选择合适的优化策略，也可以组合使用多种策略以获得更好的效果。 