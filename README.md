# LangChain 学习项目

这是一个用于学习 LangChain 框架的演示项目，使用本地Ollama模型替代OpenAI API。

## 设置

1. 克隆此仓库
2. 创建虚拟环境：`python -m venv venv`
3. 激活虚拟环境：
   - Windows: `venv\Scripts\activate`
   - MacOS/Linux: `source venv/bin/activate`
4. 安装依赖：`pip install langchain langchain-ollama`
5. 安装并启动Ollama：
   - 从 [Ollama官网](https://ollama.ai/) 下载并安装
   - 运行Ollama服务
   - 拉取所需模型：`ollama pull llama3.2`

## 示例

本项目包含以下示例：

1. **基础示例** (`src/basic_example.py`): 展示了如何使用 LangChain 创建一个简单的链。
2. **链示例** (`src/chain_example.py`): 展示了如何创建和使用顺序链。
3. **代理示例** (`src/agent_example.py`): 展示了如何使用 LangChain 的代理功能。

## 运行示例

```bash
# 运行基础示例
python src/basic_example.py

# 运行链示例
python src/chain_example.py

# 运行代理示例
python src/agent_example.py