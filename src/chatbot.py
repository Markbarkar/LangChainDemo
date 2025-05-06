from operator import add
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI
from langserve import add_routes
from langchain_community.chat_message_histories import RedisChatMessageHistory, ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.schema import HumanMessage
from uvicorn import config

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
    template = "中科天目是一家致力于智能法务的公司，你是中科天目公司旗下的一个{topic}的聊天助手，名为Tiko，善于解决法律问题。"
    prompt = PromptTemplate(template=template, input_variables=["topic"])
    
    # 创建一个简单的链
    chain = prompt | llm

    store_msg = {}

    config = {
        "session_id": "123"
    }

    def get_session_message_history(session_id: str) -> ChatMessageHistory:
        # 若有该值则存储，否则添加该值并返回设定值
        return store_msg.setdefault(session_id, ChatMessageHistory())
    
    chain_with_history = RunnableWithMessageHistory(
        chain, 
        get_session_message_history,
        input_messages_key="messages",  
    )

    repo = chain_with_history.invoke(
        input = {
        "topic": "中文", 
        "messages": [HumanMessage(content="我是小明，你是谁？")]
        },
        config=config
    )
    
    print(repo)

    repo2 = chain_with_history.invoke(
        input = {
        "topic": "中文", 
        "messages": [HumanMessage(content="我叫什么名字？")]
        },
        config=config
    )
    
    print(repo2)

if __name__ == "__main__":
    basic_langchain_example()
    
