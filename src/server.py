from operator import add
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi import FastAPI
from langserve import add_routes
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
        model="llama3.2", 
        temperature=0.7,
        base_url="http://192.168.1.6:11434"  # 替换为实际的内网Ollama服务器地址
    )
    
    # 创建一个简单的提示模板
    template = "给我讲一个关于{topic}的短故事，不超过100字。"
    prompt = PromptTemplate(template=template, input_variables=["topic"])
    
    # 创建一个简单的链
    chain = prompt | llm
    
    # 运行链
    # result = chain.invoke({"topic": "人工智能"})    
    # print(result)

    app = FastAPI()

    add_routes(app, chain, path="/chain")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    basic_langchain_example()