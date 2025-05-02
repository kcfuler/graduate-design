# 数据处理测试

该目录包含对 TT100K 数据集处理功能的测试。

## 功能描述

测试 TT100K 数据集处理模块的功能，主要验证以下内容：

1. 加载标注信息
2. 处理并转换为 YOLO 格式数据
3. 验证生成的目录结构和文件完整性

## 测试文件

- `test_processor.py`: 主测试脚本，处理 TT100K 数据集并输出 YOLO 格式数据

## 使用方法

在项目根目录下运行：

```bash
python tests/data_processor/test_processor.py --data_dir ./data --output_dir ./tests/data_processor/output --sample_count 50
```

参数说明：
- `--data_dir`: TT100K 数据集根目录
- `--output_dir`: 测试输出目录
- `--sample_count`: 采样处理的图像数量，默认 100

## 输出结果

测试脚本将在 `output` 目录中生成以下内容：

1. `sampled_annotations.json`: 采样的标注信息
2. `yolo/`: YOLO 格式的数据
   - `train/`: 训练集图像和标签
   - `val/`: 验证集图像和标签
   - `test/`: 测试集图像和标签
   - `classes.txt`: 类别名称文件
   - `tt100k.yaml`: YOLO 训练配置文件

## 成功标准

测试成功的标准为：

1. 成功加载标注信息
2. 成功生成 YOLO 格式数据
3. 每个分割集（训练集、验证集、测试集）都有图像和对应的标签
4. 类别文件和配置文件正确生成 