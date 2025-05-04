import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain

# 加载环境变量
load_dotenv()

def sequential_chain_example():
    # 初始化 LLM (使用本地Ollama模型替代OpenAI)
    llm = OllamaLLM(model="llama3.2", temperature=0.7)
    
    # 第一个链：生成一个故事主题
    first_prompt = PromptTemplate(
        input_variables=["subject"],
        template="为一个关于{subject}的故事想一个有创意的主题。"
    )
    first_chain = LLMChain(llm=llm, prompt=first_prompt)
    
    # 第二个链：根据主题生成故事
    second_prompt = PromptTemplate(
        input_variables=["theme"],
        template="写一个关于以下主题的短故事，不超过200字：{theme}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt)
    
    # 将两个链连接起来
    overall_chain = SimpleSequentialChain(
        chains=[first_chain, second_chain],
        verbose=True
    )
    
    # 运行链
    story = overall_chain.run("太空探索")
    print(story)

if __name__ == "__main__":
    sequential_chain_example()