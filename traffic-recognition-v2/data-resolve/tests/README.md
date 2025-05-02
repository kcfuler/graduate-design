# TT100K 数据处理测试

本目录包含对 TT100K 交通标志数据集处理工具的测试用例。

## 测试目录结构

```
tests/
  ├── data_processor/        # 数据处理测试
  │   ├── README.md          # 测试说明文档
  │   ├── test_processor.py  # 测试脚本
  │   └── output/            # 测试输出结果
  │
  ├── yolo_validation/       # YOLO格式验证测试
  │   ├── README.md          # 测试说明文档
  │   ├── test_yolo_data.py  # 测试脚本
  │   └── output/            # 测试输出结果(可视化结果)
  │
  └── common/                # 通用测试函数
      └── __init__.py        # 初始化文件
```

## 测试用例说明

### 1. 数据处理测试 (data_processor)

测试 TT100K 数据集处理模块的功能，包括加载标注信息、处理并转换为 YOLO 格式数据等。

运行方法：
```bash
python tests/data_processor/test_processor.py
```

详情查看 [data_processor/README.md](data_processor/README.md)。

### 2. YOLO 格式验证测试 (yolo_validation)

验证生成的 YOLO 格式数据的正确性，包括检查目录结构、标签匹配度、边界框可视化等。

运行方法：
```bash
python tests/yolo_validation/test_yolo_data.py
```

详情查看 [yolo_validation/README.md](yolo_validation/README.md)。

## 运行全部测试

可以通过以下命令运行所有测试：

```bash
# 处理数据(小样本)
python tests/data_processor/test_processor.py --sample_count 50

# 验证生成的数据
python tests/yolo_validation/test_yolo_data.py --data_dir ./tests/data_processor/output/yolo --samples 3
```

## 注意事项

1. 确保已安装必要的依赖，如 OpenCV、NumPy、tqdm 等
2. 测试结果将保存在相应测试目录的 `output` 子目录中
3. 可视化结果可用于手动检查边界框的准确性 