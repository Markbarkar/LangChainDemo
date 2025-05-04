import os
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_ollama import OllamaLLM

# 加载环境变量
load_dotenv()

def agent_example():
    # 初始化 LLM (使用本地Ollama模型替代OpenAI)
    llm = OllamaLLM(model="llama3.2", temperature=0)
    
    # 加载工具
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    
    # 初始化代理
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # 运行代理
    agent.run("今天的日期是什么？计算 (15 * 27) / 3 的结果是多少？")

if __name__ == "__main__":
    agent_example()