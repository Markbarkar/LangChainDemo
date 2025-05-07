import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI
from langserve import add_routes
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.schema import HumanMessage
import uvicorn

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"

# LANGSMITH_PROJECT="pr-untimely-streetcar-57"
# OPENAI_API_KEY="<your-openai-api-key>"

# 加载环境变量
load_dotenv()

def basic_langchain_example():
    # 初始化 LLM (使用内网Ollama服务器替代本地模型)
    llm = OllamaLLM(
        model="deepseek-r1:70b", 
        temperature=0.7,
        base_url="http://192.168.1.6:11434"  # 替换为实际的内网Ollama服务器地址
    )
    
    # 创建一个简单的提示模板
    template = """中科天目是一家致力于智能法务的公司，你是中科天目公司旗下的一个人工智能聊天助手，名为Tiko，善于解决法律问题。

    当前对话历史：
    {chat_history}

    用户: {input}
    Tiko: """
    prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])
    
    # 创建一个简单的链
    chain = prompt | llm
    
    # 格式化聊天历史
    def format_chat_history(messages):
        return "\n".join([f"{'用户' if isinstance(m, HumanMessage) else 'Tiko'}: {m.content}" for m in messages])

    # Redis配置
    redis_url = "redis://192.168.1.6:6379/0"

    def get_message_history(session_id: str) -> RedisChatMessageHistory:
        # 使用Redis存储聊天历史
        return RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url,
            key_prefix="chat_history:"
        )
    
    # 创建带有消息历史的可运行链
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_message_history,
        input_messages_key="chat_history",
        history_transform_fn=format_chat_history,
        history_messages_key="chat_history"
    )

    # 创建FastAPI应用
    app = FastAPI()

    # 添加聊天路由
    add_routes(app, chain_with_history, path="/chat")

    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    basic_langchain_example()
    
