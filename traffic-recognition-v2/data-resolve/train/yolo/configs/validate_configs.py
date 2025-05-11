#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件验证工具
用于验证YAML配置文件是否符合预定义的规则和结构
"""

import os
import sys
import yaml
import json
import re
import jsonschema
from jsonschema import validators


# 自定义格式检查器，用于处理占位符
def is_placeholder(checker, instance):
    if not isinstance(instance, str):
        return True
    # 检查是否符合${var_name:default_value}格式
    return bool(re.match(r'^\${[a-zA-Z0-9_]+(:.*)?}$', instance))


# 创建支持占位符的校验器
def create_validator():
    validator_class = jsonschema.validators.extend(
        jsonschema.validators.Draft7Validator,
        {'is_placeholder': is_placeholder}
    )
    return validator_class


def load_yaml(yaml_file):
    """加载YAML文件"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[错误] 无法加载YAML文件 '{yaml_file}': {str(e)}")
        return None


def preprocess_config(config):
    """预处理配置，将占位符值替换为默认值或假值"""
    if isinstance(config, dict):
        for key, value in list(config.items()):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # 提取默认值
                parts = value[2:-1].split(':', 1)
                if len(parts) > 1:
                    # 使用默认值
                    default_value = parts[1].strip()
                    try:
                        # 尝试转换为适当的类型
                        if default_value.lower() == 'true':
                            config[key] = True
                        elif default_value.lower() == 'false':
                            config[key] = False
                        elif default_value.replace('.', '').isdigit():
                            config[key] = float(default_value) if '.' in default_value else int(default_value)
                        else:
                            config[key] = default_value
                    except (ValueError, TypeError):
                        config[key] = default_value
                else:
                    # 没有默认值，使用适当的假值
                    if key in ['nc']:
                        config[key] = 1  # 数字类型
                    elif key in ['names']:
                        config[key] = ['class1']  # 数组类型
                    elif key.startswith(('is_', 'has_')):
                        config[key] = False  # 布尔类型
                    else:
                        config[key] = "placeholder_value"  # 字符串
            elif isinstance(value, (dict, list)):
                config[key] = preprocess_config(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = preprocess_config(item)
    return config


def validate_data_yaml(config):
    """验证data.yaml文件"""
    schema = {
        "type": "object",
        "required": ["path", "train", "val", "nc", "names"],
        "properties": {
            "path": {"type": "string"},
            "train": {"type": "string"},
            "val": {"type": "string"},
            "test": {"type": "string"},
            "nc": {"type": "integer", "minimum": 1},
            "names": {
                "type": "array",
                "items": {"type": "string"}
            },
            "mosaic": {"type": ["number", "string"]},
            "mixup": {"type": ["number", "string"]},
            "copy_paste": {"type": ["number", "string"]},
            "cache": {"type": ["boolean", "string"]},
            "rect": {"type": ["boolean", "string"]},
            "workers": {"type": ["integer", "string"]}
        }
    }
    
    # 预处理配置，处理占位符
    processed_config = preprocess_config(config.copy())
    
    try:
        jsonschema.validate(instance=processed_config, schema=schema)
        
        # 验证nc与names长度是否一致
        if isinstance(processed_config.get('names'), list) and isinstance(processed_config.get('nc'), int):
            if len(processed_config['names']) != processed_config['nc']:
                return False, f"类别数量 (nc={processed_config['nc']}) 与类别名称列表长度 (names={len(processed_config['names'])}) 不匹配"
        
        return True, "数据配置验证通过"
    except jsonschema.exceptions.ValidationError as e:
        return False, f"数据配置验证失败: {e.message}"


def validate_model_yaml(config):
    """验证模型配置文件"""
    schema = {
        "type": "object",
        "required": ["backbone", "head"],
        "properties": {
            "backbone": {"type": "array"},
            "head": {"type": "array"},
            "lr0": {"type": ["number", "string"]},
            "lrf": {"type": ["number", "string"]},
            "momentum": {"type": ["number", "string"]},
            "weight_decay": {"type": ["number", "string"]},
            "warmup_epochs": {"type": ["number", "string"]},
            "warmup_momentum": {"type": ["number", "string"]},
            "warmup_bias_lr": {"type": ["number", "string"]},
            "box": {"type": ["number", "string"]},
            "cls": {"type": ["number", "string"]},
            "obj": {"type": ["number", "string"]},
            "imgsz": {"type": ["integer", "string"]},
            "batch_size": {"type": ["integer", "string"]},
            "epochs": {"type": ["integer", "string"]}
        }
    }
    
    # 预处理配置，处理占位符
    processed_config = preprocess_config(config.copy())
    
    try:
        jsonschema.validate(instance=processed_config, schema=schema)
        return True, "模型配置验证通过"
    except jsonschema.exceptions.ValidationError as e:
        return False, f"模型配置验证失败: {e.message}"


def main():
    """主函数"""
    # 验证参数
    if len(sys.argv) < 2:
        print("使用方法: python validate_configs.py <config_file> [config_type]")
        print("可选config_type: data, model")
        return
    
    config_file = sys.argv[1]
    config_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 根据文件名猜测配置类型
    if not config_type:
        if "data" in config_file:
            config_type = "data"
        elif "yolo" in config_file:
            config_type = "model"
        else:
            print("[错误] 无法确定配置类型，请指定 config_type 参数")
            return
    
    # 加载配置
    config = load_yaml(config_file)
    if not config:
        return
    
    # 验证配置
    if config_type == "data":
        success, message = validate_data_yaml(config)
    elif config_type == "model":
        success, message = validate_model_yaml(config)
    else:
        print(f"[错误] 不支持的配置类型: {config_type}")
        return
    
    # 输出验证结果
    status = "通过" if success else "失败"
    print(f"配置文件 '{config_file}' 验证{status}: {message}")
    
    # 返回验证结果
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 