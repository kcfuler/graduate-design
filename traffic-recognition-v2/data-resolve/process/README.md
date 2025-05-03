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
    --model yolo \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --num_clusters 9 \
    --balance_factor 3 \
    --mosaic_count 1000 \
    --mixup_count 500 \
    --frequency_level all

# Linux/Mac
PYTHONPATH=./process python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --num_clusters 9 \
    --balance_factor 3 \
    --mosaic_count 1000 \
    --mixup_count 500 \
    --frequency_level all
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
│   │   ├── stratified/           # 分层处理的中间输出
│   │   │   ├── anchors.txt       # 聚类生成的anchor boxes
│   │   │   ├── class_frequency.json  # 类别频次统计
│   │   │   └── yolo_stratified/  # 分层采样后的YOLO格式数据
│   │   └── final/                # 增强后的最终数据
│   │       ├── classes.txt       # 类别名称文件 
│   │       ├── tt100k.yaml       # YOLO配置文件
│   │       ├── DATASET_INFO.md   # 数据集信息说明
│   │       └── ...
│   └── 2/                        # 第二次训练(参数调整后)
│       ├── stratified/
│       └── final/
└── mobilenet/                    # 另一种模型的处理结果
    └── 1/
        ├── stratified/
        └── final/
```

## 参数说明

主要参数解释：

- `--data_dir`：TT100K原始数据集根目录（必须指定）
- `--output_dir`：处理后数据的基础输出目录（必须指定）
- `--model`：模型名称，用于目录结构分类（默认：yolo）
- `--min_freq_high`：高频类别的最小频次阈值（默认：50）
- `--min_freq_mid`：中频类别的最小频次阈值（默认：10）
- `--num_clusters`：Anchor box聚类数量（默认：9）
- `--balance_factor`：中频类别过采样倍数（默认：3）
- `--mosaic_count`：马赛克增强样本数量（默认：1000）
- `--mixup_count`：Mixup增强样本数量（默认：500）
- `--selected_types`：只处理指定的类别（可选，默认处理所有类别）
- `--frequency_level`：输出数据集包含的频率层级，可选值：'all'、'high'、'mid'、'low'或'high,mid'等组合，用逗号分隔（默认：all）

## 期望效果

经过上述优化，我们可以期待：

1. 全204类评估时，mAP@0.5可从原始的0.20-0.30提升至0.40-0.50
2. 对高频类别子集(如≥50张样本的45类)评估时，mAP@0.5可达到0.85-0.90以上
3. 小目标检测召回率显著提升，Precision-Recall曲线的最大Recall点可从0.3左右提升至0.6以上

## 使用不同频率层级的数据集

通过`--frequency_level`参数，您可以灵活控制输出数据集中包含的类别，以便评估不同类别频率对模型性能的影响：

### 频率过滤逻辑说明

最新版本实现了更严格的频率级别过滤机制，确保只保留纯粹包含指定频率级别类别的图像：

- 当指定`--frequency_level high`时，输出数据集只包含**纯高频类别**的图像，即只有高频类别标注、没有任何中频或低频类别标注的图像
- 当指定`--frequency_level high,mid`时，输出数据集只包含高频和中频类别的图像，不包含任何低频类别标注
- 其他组合以此类推

这种严格的过滤方式与之前的版本不同（之前只要图像中包含任何一个指定频率级别的类别就会被保留），能够更精确地控制训练数据集的组成，便于进行更纯粹的频率级别对比实验。

### 1. 仅使用高频类别训练

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_high \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --frequency_level high \
    --mosaic_count 1000 \
    --mixup_count 500
```

注意：上述命令将只处理包含纯高频类别标注的图像，不包含任何中频或低频类别的图像。这将大幅减少训练集图像数量，但确保了类别的纯粹性。

### 2. 使用高频和中频类别训练

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_high_mid \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --frequency_level high,mid \
    --mosaic_count 1000 \
    --mixup_count 500
```

上述命令将处理同时包含高频和中频类别的图像，或者只包含高频类别的图像，或者只包含中频类别的图像，但不包含任何低频类别标注的图像。

### 3. 仅使用中频类别训练

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_mid \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --frequency_level mid \
    --mosaic_count 800 \
    --mixup_count 400
```

该命令将只处理包含纯中频类别标注的图像，不包含任何高频或低频类别的图像。

### 4. 仅使用低频类别训练（所有低频类别合并为'unknown_rare'）

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_low \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --frequency_level low \
    --mosaic_count 500 \
    --mixup_count 200
```

该命令将只处理包含纯低频类别标注的图像，不包含任何高频或中频类别的图像。所有低频类别都将被合并为'unknown_rare'类别。

### 5. 使用全部频率级别（默认行为）

```bash
# Windows PowerShell
$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py \
    --data_dir ./data \
    --output_dir ./processed_data \
    --model yolo_all \
    --min_freq_high 50 \
    --min_freq_mid 10 \
    --frequency_level all \
    --mosaic_count 1000 \
    --mixup_count 500
```

使用`--frequency_level all`将处理所有图像，不进行频率过滤。这是默认行为。

### 模型比较与评估

通过训练不同频率层级的模型，您可以比较：

1. 高频类别模型：类别少但每类样本充足，可能有更高的准确率
2. 中频类别模型：类别适中，样本相对较少，挑战性更大
3. 混合模型：包含所有或多种频率层级的类别，更全面但也更具挑战性

这种对比实验有助于理解类别分布对模型性能的影响，并为实际应用中的数据处理策略提供指导。

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

4. **参数错误: unrecognized arguments: --input_size**
   - 问题：README中曾提到的`--input_size`参数在最新版本脚本中已被移除
   - 解决方案：请移除此参数，直接使用其他参数运行脚本
   
5. **处理失败或中断**
   - 问题：处理过程中断或失败
   - 解决方案：
     - 使用`--skip_step`或`--only_step`参数跳过已完成步骤或只执行特定步骤
     - 检查生成的中间文件是否完整
     - 确保磁盘空间充足，TT-100K数据集处理后会占用更多空间 

6. **PowerShell参数解析错误：Missing expression after unary operator '--'**
   - 问题：在PowerShell中，双连字符(--) 被解释为一元操作符，导致命令执行失败
   - 解决方案：
     - 确保输入完整的命令，而不是只输入参数部分
     - 将整个命令作为一行输入，不要分行输入
     - 正确格式示例：`$env:PYTHONPATH="./process"; python process/scripts/tt100k_enhanced_pipeline.py --data_dir ./data --output_dir ./processed_data --model yolo_high --min_freq_high 50 --min_freq_mid 10 --frequency_level high --mosaic_count 1000 --mixup_count 500`
     - 或者考虑使用CMD而不是PowerShell来避免这个问题 