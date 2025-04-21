import os
import json
from pathlib import Path
from typing import List, Dict, Any

# 定义项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]

# 定义模型目录
MODELS_DIR = ROOT_DIR / "models"

# 模型配置文件路径
MODEL_CONFIG_PATH = MODELS_DIR / "config.json"

# 加载模型配置
def load_model_config() -> Dict[str, Any]:
    """
    加载模型配置文件
    """
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"模型配置文件不存在：{MODEL_CONFIG_PATH}")
    
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config

# 获取注册的模型列表
def get_registered_models() -> List[Dict[str, Any]]:
    """
    获取所有注册的模型信息
    """
    config = load_model_config()
    return config.get("models", [])

# 根据ID获取模型信息
def get_model_by_id(model_id: str) -> Dict[str, Any]:
    """
    根据模型ID获取模型信息
    """
    models = get_registered_models()
    for model in models:
        if model.get("id") == model_id:
            return model
    return None 