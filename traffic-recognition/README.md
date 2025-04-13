# 交通标识识别系统

一个基于 Gradio 的交通标识识别演示系统，支持模型调试、效果验证和数据导出。

## 功能特点

- 支持多种模型切换（目前支持 MobileNet）
- 实时推理结果显示
- 性能指标监控
- 批量图像处理
- 数据导出功能
- 友好的用户界面

## 系统要求

- Python 3.8+
- CUDA 支持（可选，用于 GPU 加速）

## 安装步骤

1. 克隆项目：
```bash
git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
```

2. 安装 uv（如果尚未安装）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. 创建并激活虚拟环境：
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

4. 安装依赖：
```bash
uv pip install -r requirements.txt
```

## 使用方法

1. 启动应用：
```bash
python app/main.py
```

2. 在浏览器中访问：
```
http://localhost:7860
```

3. 界面操作：
   - 选择模型（目前支持 MobileNet）
   - 上传单张图片或批量图片
   - 查看推理结果和性能指标
   - 导出数据（支持 CSV 格式）

## 开发指南

### 项目结构

```
traffic_sign_recognition/
├── app/
│   ├── models/          # 模型管理
│   ├── processors/      # 数据处理
│   ├── metrics/         # 性能指标
│   ├── exporters/       # 数据导出
│   └── main.py          # 主应用
├── tests/               # 测试目录
├── requirements.txt     # 依赖管理
└── README.md           # 项目文档
```

### 添加新模型

1. 在 `app/models/` 目录下创建新模型文件
2. 实现 `BaseModel` 接口
3. 在 `app/models/__init__.py` 中注册模型

示例：
```python
from app.models.base import BaseModel

class NewModel(BaseModel):
    def load_model(self, model_path=None):
        # 实现模型加载
        pass
    
    def preprocess(self, image):
        # 实现图像预处理
        pass
    
    def postprocess(self, output):
        # 实现后处理
        pass
```

### 运行测试

```bash
python -m unittest discover tests
```

### 数据导出

系统支持三种数据导出格式：
1. 推理结果导出
2. 性能指标导出
3. 批量处理结果导出

导出的 CSV 文件包含时间戳、类别信息、置信度等数据。

## 性能优化

- 使用 GPU 加速（如果可用）
- 批量处理提高效率
- 图像预处理优化
- 内存使用监控

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至：your-email@example.com 