from .base import ModelFactory
from .mobilenet import MobileNetModel

# 注册 MobileNet 模型
ModelFactory.register_model("mobilenet", MobileNetModel)
