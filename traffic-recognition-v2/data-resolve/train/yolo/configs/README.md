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