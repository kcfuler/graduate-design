# TT-100K 交通标志数据集优化处理

本目录包含用于处理和优化TT-100K交通标志数据集的工具和脚本，优化的主要目标是提高检测模型对小目标交通标志的识别能力，解决类别分布不均衡问题。

## 背景

TT-100K数据集包含了204个类别、约7000张标注图像的交通标志，但存在两个主要问题：

1. **类别分布极度不均衡**：部分常见标志有数百甚至上千个实例，而大部分罕见标志只有几个实例。
2. **目标尺寸小**：交通标志在图像中通常占比很小（<0.05×0.05），常规检测模型容易漏检。

## 优化解决方案

我们实现了一套完整的数据处理流水线，主要包括以下步骤：

### 1. 类别频次分层处理

将204个类别根据出现频次分为三个档次：
- **A类** (高频类别): ≥50张样本的类别，保持原样
- **B类** (中频类别): 10-49张样本的类别，在训练时过采样3-5倍
- **C类** (低频类别): <10张样本的类别，合并为`unknown_rare`类别统一处理

### 2. Anchor重聚类

使用K-means++算法在TT-100K数据集上重新聚类Anchor boxes，生成更适合小目标检测的先验框，提高召回率。

### 3. 数据增强

实现针对交通标志小目标的特定增强策略：
- **Mosaic增强**：将4张图像拼接成一张，增加每批次训练的目标数量和上下文多样性
- **Mixup增强**：将两张图像按比例混合，增加训练样本的多样性和难度

## 文件结构

```
process/
├── data/                 # 数据处理核心模块
│   ├── processor.py      # TT100K数据处理器
│   └── utils.py          # 数据处理工具函数
├── scripts/              # 处理脚本
│   ├── process_tt100k.py          # 基础处理脚本
│   ├── advanced_tt100k_process.py # 分层处理脚本
│   ├── augment_tt100k.py          # 数据增强脚本
│   └── tt100k_enhanced_pipeline.py # 完整处理流水线
└── README.md             # 本文档
```

## 使用方法

### 注意事项

在执行脚本前，请确保解决以下常见问题：

1. **模块导入问题**：脚本中导入了`data`模块，需要确保Python能够找到该模块。可通过以下两种方式解决：
   - 修改脚本中的导入顺序，确保在导入`data`模块前已添加系统路径
   - 在运行命令时添加环境变量：`$env:PYTHONPATH="./process"` (Windows PowerShell) 或 `PYTHONPATH=./process` (Linux/Mac)

2. **路径问题**：确保在项目根目录下执行命令，而不是在`process`目录内

### 快速开始：一键处理流水线

使用`tt100k_enhanced_pipeline.py`脚本执行完整处理流程：

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --num_clusters 9 \
    --balance_factor 3 \
    --mosaic_count 1000 \
    --mixup_count 500 \
    --input_size 1280

# Linux/Mac
PYTHONPATH=./process python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --num_clusters 9 \
    --balance_factor 3 \
    --mosaic_count 1000 \
    --mixup_count 500 \
    --input_size 1280
```

### 分步执行

如果需要分步执行或自定义处理过程：

#### 1. 分层处理

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/advanced_tt100k_process.py \
    --data_dir ./data \
    --output_dir ./processed_data/stratified \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --num_clusters 9 \
    --balance_factor 3

# Linux/Mac
PYTHONPATH=./process python process/scripts/advanced_tt100k_process.py \
    --data_dir ./data \
    --output_dir ./processed_data/stratified \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --num_clusters 9 \
    --balance_factor 3
```

#### 2. 数据增强

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/augment_tt100k.py \
    --yolo_dir ./processed_data/stratified/yolo_stratified \
    --output_dir ./processed_data/final \
    --mosaic_count 1000 \
    --mixup_count 500 \
    --input_size 1280 \
    --copy_orig

# Linux/Mac
PYTHONPATH=./process python process/scripts/augment_tt100k.py \
    --yolo_dir ./processed_data/stratified/yolo_stratified \
    --output_dir ./processed_data/final \
    --mosaic_count 1000 \
    --mixup_count 500 \
    --input_size 1280 \
    --copy_orig
```

## 处理后的数据结构

成功处理后，输出目录将包含以下结构：

```
processed_data/
├── stratified/                # 分层处理的中间输出
│   ├── anchors.txt            # 聚类生成的anchor boxes
│   ├── class_frequency.json   # 类别频次统计
│   └── yolo_stratified/       # 分层采样后的YOLO格式数据
│       ├── classes.txt        # 类别名称文件
│       ├── tt100k.yaml        # YOLO配置文件
│       ├── train/             # 训练集
│       ├── val/               # 验证集
│       └── test/              # 测试集
└── final/                     # 增强后的最终数据
    ├── classes.txt            # 类别名称文件 
    ├── tt100k.yaml            # YOLO配置文件
    ├── DATASET_INFO.md        # 数据集信息说明
    ├── train/                 # 训练集
    ├── val/                   # 验证集
    └── test/                  # 测试集
```

## 参数说明

主要参数解释：

- `--min_freq_high`：高频类别的最小频次阈值（默认：50）
- `--min_freq_mid`：中频类别的最小频次阈值（默认：10）
- `--num_clusters`：Anchor box聚类数量（默认：9）
- `--balance_factor`：中频类别过采样倍数（默认：3）
- `--mosaic_count`：马赛克增强样本数量（默认：1000）
- `--mixup_count`：Mixup增强样本数量（默认：500）
- `--input_size`：输入图像分辨率（默认：1280）
- `--selected_types`：只处理指定的类别（可选，默认处理所有类别）

## 期望效果

经过上述优化，我们可以期待：

1. 全204类评估时，mAP@0.5可从原始的0.20-0.30提升至0.40-0.50
2. 对高频类别子集(如≥50张样本的45类)评估时，mAP@0.5可达到0.85-0.90以上
3. 小目标检测召回率显著提升，Precision-Recall曲线的最大Recall点可从0.3左右提升至0.6以上

## 依赖项

- Python 3.8+
- numpy
- opencv-python
- scikit-learn (用于K-means聚类)
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

3. **处理时间过长**
   - 问题：大数据集处理耗时长
   - 解决方案：可以通过`--selected_types`参数只处理部分类别，或调低`--mosaic_count`和`--mixup_count`参数 