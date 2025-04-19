from .base import ModelFactory
from .mobilenet import MobileNetModel
from .yolo_model import YOLOModel

# 注册 MobileNet 模型
ModelFactory.register_model("mobilenet", MobileNetModel)

# 注册 YOLOv8n 模型
ModelFactory.register_model("yolov8n.pt", YOLOModel)

# 你可以在这里注册更多模型，例如:
# ModelFactory.register_model("yolov8s.pt", YOLOModel, model_path="yolov8s.pt")
