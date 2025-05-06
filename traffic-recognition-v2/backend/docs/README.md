# 交通标志识别API - 后端服务说明

## 1. 项目背景 & 设计目标

### 项目背景

本项目是一个深度学习交通标志识别系统，采用三层架构（前端、中间件、后端）实现。系统的核心目标是允许用户通过Web界面上传图片或视频，选择指定的AI模型进行交通标志的实时或离线识别，并清晰地展示识别结果。后端API作为此系统的关键组成部分，负责模型管理、执行推理任务，并为上层应用提供稳定、高效的数据接口和处理能力。

### 设计目标

后端API的设计旨在实现以下核心目标：

*   **准确性与高效性:**
    *   支持集成和运行高精度的交通标志识别模型（如MobileNet、YOLO等）。
    *   确保API请求的快速响应，优化推理过程以减少延迟。
*   **模块化与可扩展性:**
    *   采用模块化设计，将模型管理、推理服务、配置等功能解耦。
    *   方便未来添加新的深度学习模型或扩展现有功能（例如，增加对不同类型输入、新训练流程的支持）。
    *   最初采用配置文件驱动的模型管理，并预留扩展为数据库驱动的接口。
*   **易用性与集成性:**
    *   提供清晰、规范的API接口，遵循FastAPI的最佳实践，自动生成交互式API文档。
    *   简化与中间件及前端应用的集成流程。
*   **健壮性与可靠性:**
    *   实现恰当的错误处理机制，为客户端提供明确的错误信息。
    *   确保服务在高并发场景下的稳定性。
*   **可配置性:**
    *   支持通过配置文件或环境变量灵活配置应用参数，如模型路径、日志级别等。

## 2. 技术栈 & 开发规范

### 技术栈
*   **Web框架:** FastAPI
*   **ASGI服务器:** Uvicorn
*   **编程语言:** Python
*   **主要依赖:**
    *   `fastapi`: 用于构建API。
    *   `uvicorn`: 用于运行FastAPI应用。
    *   `pydantic`: 用于数据验证和设置管理。
    *   `python-multipart`: 用于处理表单数据 (例如文件上传)。
    *   `opencv-python-headless`: 用于图像处理。
    *   `numpy`: 用于数值计算。
    *   `pillow`: 用于图像处理。
    *   `tensorflow`: 用于加载和运行深度学习模型。

### 开发规范
*   **代码风格:** 遵循PEP 8编码规范。
*   **模块化:** 将不同功能的代码组织在独立的模块中 (例如 `app/api/routes/`, `app/models/`)。
*   **错误处理:** 对API请求进行适当的错误处理和响应。
*   **文档:** API接口应有清晰的文档说明 (FastAPI自动生成交互式文档)。
*   **虚拟环境:** 强烈建议使用Python虚拟环境 (如 `venv`) 来管理项目依赖。

## 3. API 定义与实现

当前后端服务主要包含以下API端点：

### 3.1 模型管理 (`/models`)

*   **模块:** `app.api.routes.models`
*   **功能:** 提供对系统中可用的深度学习模型的管理功能，例如列出模型、加载模型等。
*   **主要端点 (示例):**
    *   `GET /models/`: 获取可用模型列表。
    *   `POST /models/{model_name}/load`: 加载指定模型。
    *(具体的端点和请求/响应格式请参考 `app/api/routes/models.py` 和FastAPI自动生成的文档 `/docs`)*

### 3.2 推理 (`/infer`)

*   **模块:** `app.api.routes.inference`
*   **功能:** 使用加载的模型对输入的图像进行交通标志识别。
*   **主要端点 (示例):**
    *   `POST /infer/`: 接收图像数据，返回识别结果。
    *(具体的端点和请求/响应格式请参考 `app/api/routes/inference.py` 和FastAPI自动生成的文档 `/docs`)*

### 3.3 健康检查 (`/`)

*   **功能:** 检查API服务是否正常运行。
*   **端点:**
    *   `GET /`: 返回 `{"message": "交通标志识别API在线"}` 表示服务正常。

## 4. 如何启动服务

### 4.1 环境准备

1.  **克隆项目:**
    ```bash
    git clone <your-repository-url>
    cd traffic-recognition-v2
    ```
2.  **创建并激活Python虚拟环境:**
    ```bash
    python -m venv .venv  # 或者你习惯的虚拟环境名称
    source .venv/bin/activate  # macOS/Linux
    # .venv\\\\Scripts\\\\activate  # Windows
    ```
3.  **安装依赖:**
    项目根目录下的 `backend/requirements.txt` 文件列出了所有必要的Python依赖。
    ```bash
    pip install -r backend/requirements.txt
    ```

### 4.2 启动命令

在项目**根目录** (`traffic-recognition-v2`) 下执行以下命令启动服务：

```bash
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

或者，直接在项目**根目录**下执行：

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir backend
```
*(注意：第二种方式需要 Uvicorn 0.15.0 或更高版本才支持 `--app-dir` 参数。第一种方式更通用。)*


命令解释：
*   `cd backend`: 进入后端应用的根目录。
*   `python -m uvicorn`: 以模块方式运行 `uvicorn`。
*   `app.main:app`: 指定 `uvicorn` 加载 `app/main.py` 文件中的 `app` FastAPI实例。
*   `--host 0.0.0.0`: 使服务在所有网络接口上可用。
*   `--port 8000`: 指定服务监听的端口号。
*   `--reload`: 启用热重载，当代码文件发生变化时，服务会自动重启。

服务启动后，可以通过浏览器访问 `http://0.0.0.0:8000/docs` 查看自动生成的API交互文档。

### 4.3 故障排除

*   **`ModuleNotFoundError: No module named \'app\'` 或类似导入错误:**
    确保你是从 `backend` 目录或者正确设置了 `PYTHONPATH` 或使用了 `-m` 选项来运行 `uvicorn`，以便Python能够找到 `app` 模块。上述推荐的启动命令通常可以解决这个问题。

*   **`[Errno 48] Address already in use`:**
    这个错误表示指定的端口 (例如 `8000`) 已经被其他进程占用。
    *   **检查是否有旧的服务实例仍在运行:** 如果你之前在后台启动过服务，它可能仍在运行。你需要找到并停止该进程。
        *   在 macOS/Linux 上，你可以使用 `lsof -i :8000` 或 `ps aux | grep uvicorn` 来查找进程ID (PID)，然后使用 `kill <PID>` 来停止它。
        *   在 Windows 上，可以使用 `netstat -ano | findstr :8000` 找到使用该端口的进程PID，然后在任务管理器中结束该进程或使用 `taskkill /PID <PID> /F`。
    *   **更改端口:** 如果无法停止占用端口的进程，或者希望同时运行多个服务，可以在启动命令中修改 `--port` 参数为其他未被占用的端口，例如 `8001`。 