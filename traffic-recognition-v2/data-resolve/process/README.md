# TT-100K 交通标志数据集优化处理

本目录包含用于处理和优化TT-100K交通标志数据集的工具和脚本，优化的主要目标是根据类别频次进行数据筛选，转换为YOLO格式以便于训练。

## 背景

TT-100K数据集包含了204个类别、约7000张标注图像的交通标志，但存在两个主要问题：

1. **类别分布极度不均衡**：部分常见标志有数百甚至上千个实例，而大部分罕见标志只有几个实例。
2. **目标尺寸小**：交通标志在图像中通常占比很小（<0.05×0.05），常规检测模型容易漏检。

## 优化解决方案

我们实现了一套简化的数据处理流水线，主要包括以下步骤：

### 1. 类别频次筛选

根据指定的类别频次阈值筛选数据：
- **保留类别**: 出现频次≥指定阈值的类别
- **丢弃类别**: 出现频次<指定阈值的类别

### 2. Anchor重聚类（可选）

使用K-means++算法在TT-100K数据集上重新聚类Anchor boxes，生成更适合小目标检测的先验框，提高召回率。

## 文件结构

```
process/
├── data/                 # 数据处理核心模块
│   ├── processor.py      # TT100K数据处理器
│   └── utils.py          # 数据处理工具函数
├── scripts/              # 处理脚本
│   ├── process_tt100k.py          # 基础处理脚本
│   ├── advanced_tt100k_process.py # 分层处理脚本
│   └── tt100k_simple_pipeline.py  # 简化处理流水线
└── README.md             # 本文档
```

## 使用方法

### 注意事项

在执行脚本前，请确保解决以下常见问题：

1. **模块导入问题**：脚本中导入了`data`模块，需要确保Python能够找到该模块。可通过以下两种方式解决：
   - 修改脚本中的导入顺序，确保在导入`data`模块前已添加系统路径
   - 在运行命令时添加环境变量：`$env:PYTHONPATH="./process"` (Windows PowerShell) 或 `PYTHONPATH=./process` (Linux/Mac)

2. **路径问题**：确保在项目根目录下执行命令，而不是在`process`目录内

### 快速开始：简化处理流水线

使用`tt100k_simple_pipeline.py`脚本执行简化处理流程：

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_simple_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo \
    --min_freq 100 \
    --num_clusters 9

# Linux/Mac
PYTHONPATH=./process python process/scripts/tt100k_simple_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo \
    --min_freq 100 \
    --num_clusters 9
```

## 版本管理机制

脚本实现了自动版本管理功能，具体工作方式如下：

1. 处理后的数据将按照`processed_data/模型/训练次数`的结构进行组织
2. 每次运行脚本时，会自动检测已存在的训练次数，并创建新的训练数据目录
3. 例如:
   - 第一次训练: `processed_data/yolo/1/`
   - 第二次训练: `processed_data/yolo/2/`
   - 对新模型的训练: `processed_data/mobilenet/1/`

通过这种方式，可以方便地对比不同处理参数的效果，或保留多个不同版本的训练数据，同时保持目录结构清晰。在实验过程中，可以通过`--model`参数指定不同模型名称，系统会自动为每个模型单独创建训练目录。

## 处理后的数据结构

成功处理后，版本化的输出目录将包含以下结构：

```
processed_data/
├── yolo/                         # 模型类型目录
│   ├── 1/                        # 第一次训练
│   │   └── final/                # 处理后的数据
│   │       ├── classes.txt       # 类别名称文件 
│   │       ├── tt100k.yaml       # YOLO配置文件
│   │       ├── DATASET_INFO.md   # 数据集信息说明
│   │       ├── train/            # 训练集
│   │       │   ├── images/       # 训练图像
│   │       │   └── labels/       # 训练标签
│   │       ├── val/              # 验证集
│   │       │   ├── images/       # 验证图像
│   │       │   └── labels/       # 验证标签
│   │       └── test/             # 测试集
│   │           ├── images/       # 测试图像
│   │           └── labels/       # 测试标签
│   └── 2/                        # 第二次训练(参数调整后)
│       └── final/
└── mobilenet/                    # 另一种模型的处理结果
    └── 1/
        └── final/
```

## 参数说明

主要参数解释：

- `--data_dir`：TT100K原始数据集根目录（必须指定）
- `--output_dir`：处理后数据的基础输出目录（必须指定）
- `--model`：模型名称，用于目录结构分类（默认：yolo）
- `--min_freq`：类别的最小频次阈值，小于此阈值的类别将被丢弃（默认：100）
- `--max_images_per_class`：每个类别最多保留的图片数量（默认：300，可选）。在按频次筛选后，进一步限制每个保留类别所包含的图片上限。
- `--num_clusters`：Anchor box聚类数量（默认：9，可选）
- `--selected_types`：只处理指定的类别（可选，默认处理所有符合频次阈值的类别）
- `--seed`：随机种子，用于数据集分割的可重复性（默认：42）

## 期望效果

通过简化的数据处理流程，我们可以专注于高频类别的训练，减少数据集中的噪声，提高模型性能。这种方法有助于：

1. 减少类别失衡问题
2. 减少训练时间和资源消耗
3. 提高高频类别的识别准确率

## 示例：不同频次阈值的使用

### 1. 只保留图片数量≥100的高频类别

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_simple_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_high_100 \
    --min_freq 100
```

### 2. 保留图片数量≥50的类别

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_simple_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_mid_50 \
    --min_freq 50
```

### 3. 保留图片数量≥10的类别（包含更多类别）

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_simple_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_low_10 \
    --min_freq 10
```

## 依赖项

- Python 3.8+
- numpy
- opencv-python
- scikit-learn (用于K-means聚类，可选)
- tqdm (进度显示)

可通过以下命令安装必要依赖：

```bash
pip install numpy opencv-python scikit-learn tqdm
```

## 常见问题解决

1. **ModuleNotFoundError: No module named 'data'**
   - 问题：脚本无法找到`data`模块
   - 解决方案：设置`PYTHONPATH`环境变量，或修改脚本中的导入顺序

2. **FileNotFoundError: 标注文件不存在**
   - 问题：脚本找不到annotations_all.json文件
   - 解决方案：确保`--data_dir`参数指向包含annotations_all.json的TT-100K数据集根目录

3. **处理失败或中断**
   - 问题：处理过程中断或失败
   - 解决方案：
     - 检查生成的中间文件是否完整
     - 确保磁盘空间充足，TT-100K数据集处理后会占用更多空间 

4. **PowerShell参数解析错误：Missing expression after unary operator '--'**
   - 问题：在PowerShell中，双连字符(--) 被解释为一元操作符，导致命令执行失败
   - 解决方案：
     - 确保输入完整的命令，而不是只输入参数部分
     - 将整个命令作为一行输入，不要分行输入
     - 正确格式示例：`$env:PYTHONPATH="./process"; python process/scripts/tt100k_simple_pipeline.py --data_dir ./data --output_dir ./processed_data --model yolo_high --min_freq 100`
     - 或者考虑使用CMD而不是PowerShell来避免这个问题 