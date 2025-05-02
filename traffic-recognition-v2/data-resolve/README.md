# TT100K 数据集处理与模型训练指南

本项目提供了处理 Tsinghua-Tencent 100K (TT100K) 交通标志数据集的工具，支持转换为YOLO和MobileNet等不同模型的训练格式，同时保持原始数据集的完整性。

## 目录结构

```
.
├── data/                  # 原始TT100K数据集
│   ├── train/             # 训练集图片
│   ├── test/              # 测试集图片
│   ├── other/             # 其他图片
│   ├── marks/             # 标准交通标志图片
│   ├── annotations.json   # 标注信息
│   └── lmdb/              # LMDB格式数据
│
├── process/               # 数据处理相关代码
│   ├── data/              # 数据处理核心模块
│   │   ├── processor.py   # 数据处理器
│   │   └── utils.py       # 工具函数
│   │
│   ├── scripts/           # 处理脚本
│   │   ├── process_tt100k.py    # 主处理脚本
│   │   └── extract_weather_data.py # 天气数据提取脚本
│   │
│   ├── weather_analysis/  # 天气分析工具
│   └── config/            # 配置文件
│
├── processed_data/        # 处理后的数据
│   ├── yolo/              # YOLO格式数据
│   │   ├── train/         # 训练集
│   │   ├── val/           # 验证集
│   │   └── test/          # 测试集
│   │
│   ├── mobilenet/         # MobileNet格式数据（按类别组织）
│   │   ├── train/         # 训练集
│   │   ├── val/           # 验证集
│   │   └── test/          # 测试集
│   │
│   └── weather_conditions/ # 按天气条件分类的数据
│       ├── rainy/         # 雨天图片
│       ├── foggy/         # 雾天图片
│       ├── night/         # 夜间图片
│       ├── snow/          # 雪天图片
│       └── normal/        # 正常天气图片
│
├── tests/                 # 测试代码
├── test_processor.py      # 处理器测试脚本
├── quick_start.sh         # 快速启动脚本 
├── setup.py               # 项目安装配置
└── README.md              # 使用说明
```

## 环境要求

- Python 3.6+
- OpenCV
- NumPy
- tqdm

安装依赖：

```bash
# 安装所有依赖
pip install -e .

# 或者单独安装核心依赖
pip install opencv-python numpy tqdm
```

## 快速开始

使用快速启动脚本可以一键完成数据处理和模型训练：

```bash
# 基本数据处理
./quick_start.sh --data_dir ./data --output_dir ./processed_data

# 数据处理并提取雨天场景数据
./quick_start.sh --data_dir ./data --output_dir ./processed_data --extract-weather --weather rainy

# 数据处理并训练YOLO模型
./quick_start.sh --data_dir ./data --output_dir ./processed_data --train-yolo

# 数据处理并训练MobileNet模型
./quick_start.sh --data_dir ./data --output_dir ./processed_data --train-mobilenet

# 提取特定类别的数据
./quick_start.sh --extract-weather --weather rainy --classes p5 p10 p23
```

## 数据处理

如果需要单独运行数据处理脚本：

```bash
python process/scripts/process_tt100k.py --data_dir ./data --output_dir ./processed_data
```

参数说明：
- `--data_dir`: TT100K数据集根目录
- `--output_dir`: 输出目录

处理完成后，将在输出目录中生成以下数据：
1. YOLO格式数据：用于训练YOLO模型
2. MobileNet格式数据：用于训练MobileNet等分类模型
3. 按天气条件分类的数据：用于研究不同天气条件下的交通标志识别

## 提取特定天气条件的数据

```bash
python process/scripts/extract_weather_data.py --data_dir ./processed_data/weather_conditions --weather rainy --create_yaml
```

参数说明：
- `--data_dir`: 天气数据根目录
- `--weather`: 要提取的天气条件（rainy, foggy, night, snow, normal）
- `--create_yaml`: 是否创建YOLO训练配置文件
- `--classes`: 可选，指定要提取的标志类别

## YOLO模型训练

### YOLO v11模型训练

1. 首先确保已安装YOLOv11环境
   ```bash
   pip install ultralytics
   ```

2. 使用项目中的训练脚本进行训练：

```bash
# 创建必要的输出目录
mkdir -p train/yolo/outputs

# 使用训练脚本进行训练 (使用预训练模型)
python train/yolo/scripts/train_yolo.py --data_dir ./processed_data/yolo --epochs 50 --batch_size 8 --project train/yolo/outputs --name tt100k_traffic_signs --pretrained

# 或者从头开始训练
python train/yolo/scripts/train_yolo.py --data_dir ./processed_data/yolo --epochs 50 --batch_size 8 --project train/yolo/outputs --name tt100k_traffic_signs
```

也可以直接使用 ultralytics 命令行进行训练：

```bash
# 使用预训练模型训练
yolo train model=yolo11n.pt data=processed_data/yolo/tt100k.yaml epochs=50 imgsz=640 batch=8 project=train/yolo/outputs name=tt100k_yolov11

# 从头开始训练
yolo train model=yolo11n data=processed_data/yolo/tt100k.yaml epochs=50 imgsz=640 batch=8 project=train/yolo/outputs name=tt100k_yolov11
```

### 根据天气条件训练特定模型

如果需要针对特定天气条件训练模型：

```bash
# 为雨天场景训练专门的模型
yolo train model=yolov11 data=processed_data/weather_conditions/rainy/tt100k_rainy.yaml epochs=100 imgsz=640
```

## MobileNet模型训练

MobileNet等分类模型的训练：

```python
# 示例代码，需要根据您的具体环境调整
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 加载训练数据
train_generator = train_datagen.flow_from_directory(
    'processed_data/mobilenet/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 加载验证数据
val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'processed_data/mobilenet/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 创建基础模型
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定义层
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# 冻结基础模型
base_model.trainable = False

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# 微调模型
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# 保存模型
model.save('tt100k_mobilenet.h5')
```

## 评估模型

评估YOLO模型：

```bash
# 使用YOLO命令行工具评估模型
yolo val model=train/yolo/outputs/tt100k_traffic_signs/weights/best.pt data=processed_data/yolo/tt100k.yaml

# 或使用我们的测试脚本生成详细评估报告
python train/yolo/scripts/test_model.py --model train/yolo/outputs/tt100k_traffic_signs/weights/best.pt --data processed_data/yolo/tt100k.yaml --output_dir test_results --device cpu
```

测试脚本会生成以下内容：
- 混淆矩阵可视化
- 精确率-召回率曲线
- 各类别的性能指标
- 详细的测试报告

如果需要在实际图像上测试：

```bash
# 在测试图像上运行模型
python train/yolo/scripts/test_model.py --model train/yolo/outputs/tt100k_traffic_signs/weights/best.pt --data processed_data/yolo/tt100k.yaml --test_images processed_data/yolo/test/images --output_dir test_results --device cpu
```

## 生成训练测试报告

完成YOLO模型训练后，可以使用以下命令生成详细的训练测试报告：

```bash
# 创建报告输出目录
mkdir -p reports

# 生成训练报告
python train/yolo/scripts/generate_report.py --results_dir train/yolo/outputs/tt100k_traffic_signs --output_dir reports
```

生成的报告将包含以下内容：
- 训练基本信息（模型类型、训练轮数、图像尺寸等）
- 训练参数配置
- 训练性能（学习率、训练时间）
- 训练损失曲线
- 验证指标（mAP、精确率、召回率）
- 各类别性能指标
- 训练结果可视化图表

报告文件会以markdown格式保存，可以轻松转换为PDF或HTML格式进行分享。

评估MobileNet模型：

```python
# 加载测试数据
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'processed_data/mobilenet/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 评估模型
model = tf.keras.models.load_model('tt100k_mobilenet.h5')
results = model.evaluate(test_generator)
print(f"测试精度: {results[1]*100:.2f}%")
```

## 测试数据处理

如果需要在一个小样本上测试数据处理功能：

```bash
python test_processor.py --data_dir ./data --output_dir ./test_output --sample_count 100
```

参数说明：
- `--data_dir`: 原始数据集目录
- `--output_dir`: 测试输出目录
- `--sample_count`: 采样处理的图像数量

## 注意事项

1. 原始数据集不会被修改，所有处理结果都保存在新的目录中
2. 天气条件分类使用简单的启发式方法，实际项目中可能需要更复杂的分类器
3. 根据您的需求，可以调整脚本中的参数以获得更好的结果

## 自定义数据集

如果需要提取特定类型的交通标志，可以修改处理脚本中的过滤条件：

```python
# 只处理某些类别的标志
selected_types = ['p5', 'p10', 'p23']  # 示例：只选择这些类型的标志
processor = TT100KProcessor(args.data_dir, args.output_dir, selected_types=selected_types)
```

也可以通过快速启动脚本实现：

```bash
./quick_start.sh --extract-weather --weather normal --classes p5 p10 p23
``` 