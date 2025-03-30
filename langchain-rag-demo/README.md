# LangChain RAG 演示

这是一个使用 LangChain 实现的检索增强生成 (RAG) 演示项目，它可以回答关于特定网页内容的问题。

## 环境变量配置

在项目根目录下创建 `.env` 文件，并添加以下内容：

```plaintext
OPENAI_API_KEY=your_api_key_here
```

将 `your_api_key_here` 替换为你实际的 OpenAI API 密钥。

## 安装依赖

本项目使用 `uv` 作为包管理器。安装依赖：

```bash
uv pip install -e .
```

或使用传统的 pip：

```bash
pip install -e .
```

项目依赖包括：
- langchain-community
- langchain-core
- langchain-openai
- langchain-text-splitters
- langchain-chroma
- langgraph
- python-dotenv
- bs4
- notebook

## 使用方法

运行 `main.py` 文件来执行 RAG 演示：

```bash
python main.py
```

默认情况下，程序会加载 Lilian Weng 的 "LLM-Powered Autonomous Agents" 博客文章，并尝试回答 "What is Task Decomposition?" 的问题。

## 项目结构

- `main.py`: 主程序，包含完整的 RAG 流程实现
- `.env`: 环境变量文件，存储 API 密钥
- `pyproject.toml`: 项目依赖配置

## 任务列表

- [x] 创建 `.env` 文件
- [x] 在 `.env` 文件中添加 `OPENAI_API_KEY` 环境变量
- [x] 分析 `main.py` 代码
- [x] 修改 `main.py`，添加环境变量加载、向量存储和语言模型初始化
- [x] 更新 `README.md`，添加项目说明和使用指南
