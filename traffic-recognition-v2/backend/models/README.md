# 交通标志识别模型

本目录包含用于交通标志识别的预训练模型。

## 模型说明

目前支持以下两种模型：

1. **MobileNetV3** - 轻量级分类模型，适用于资源受限场景
2. **YOLOv11** - 高性能目标检测模型，提供精确的标志检测和分类

## 模型下载

为了便于使用，我们提供了自动下载脚本。在使用模型前，请先运行下载脚本：

```bash
# 下载所有模型
python download.py

# 仅下载指定模型
python download.py --model mobilenet_v3
python download.py --model yolov11
```

下载的模型将保存在 `models/mobilenet_v3` 和 `models/yolov11` 目录下。

## 模型测试

下载模型后，可以使用测试脚本验证模型的加载和推理功能：

```bash
# 测试所有模型
python test/test_run/run.py

# 仅测试指定模型
python test/test_run/run.py --model mobilenet_v3
python test/test_run/run.py --model yolov11
```

测试结果将保存在 `models/test_results` 目录下。

## 模型配置

模型的配置信息存储在 `config.json` 文件中，包括：

- 模型名称和描述
- 下载URL和校验信息
- 模型格式和输入大小
- 支持的类别列表

## 依赖项

模型运行需要以下依赖：

- TensorFlow 2.x (用于MobileNetV3)
- PyTorch 2.x (用于YOLOv11)
- OpenCV
- NumPy
- Pillow
- Ultralytics (可选，用于YOLOv11)

可以使用以下命令安装依赖：

```bash
pip install tensorflow torch torchvision opencv-python numpy pillow ultralytics
```

## 在应用中使用模型

模型加载和推理的代码示例位于 `app/models/` 目录下：

- MobileNetV3: `app/models/mobilenet_v3/model.py`
- YOLOv11: `app/models/yolov11/model.py`

### 示例用法

```python
# 加载MobileNetV3模型
from app.models.mobilenet_v3.model import MobileNetV3Model
model = MobileNetV3Model("models/mobilenet_v3/mobilenet_v3_small_224_1.0_float_no_top.h5")
predictions = model.predict(image)  # image是OpenCV格式的图像(BGR)

# 加载YOLOv11模型
from app.models.yolov11.model import YOLOv11Model
model = YOLOv11Model("models/yolov11/yolov11n.pt")
predictions = model.predict(image)  # image是OpenCV格式的图像(BGR)
```

## 问题与帮助

如果在使用过程中遇到问题，请检查：

1. 模型文件是否成功下载
2. 依赖包是否正确安装
3. 输入图像格式是否正确

更多帮助和详细文档，请参考项目主页。 