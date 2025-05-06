import os
import ssl
import nltk
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import SequentialChain
from langchain.agents import initialize_agent, AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 解决SSL证书验证问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载NLTK资源
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    print(f"NLTK下载错误: {e}")

# 加载环境变量
load_dotenv()

def sequential_chain_example():
    loader = UnstructuredFileLoader("data/test.txt")
    documents = loader.load()

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )

    split_documents = text_splitter.split_documents(documents)
    print("文本分割长度：",len(split_documents))

    texts = text_splitter.split_documents(documents)

    # 初始化 LLM (使用内网Ollama服务器替代本地模型)
    llm = OllamaLLM(
        model="deepseek-r1:70b", 
        temperature=0.7,
        base_url="http://192.168.1.6:11434"  # 替换为实际的内网Ollama服务器地址
    )
    
    # # 只使用llm-math工具，不使用需要API密钥的serpapi工具
    # tools = load_tools(["llm-math"], llm=llm)

    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True
    # )
    # # 修改示例问题，只使用数学计算功能
    # agent.invoke({"input": "计算 (15 * 27) / 3 的结果是多少？"})

    chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

    # 执行总结链，（为了快速演示，只总结前5段）
    # 使用invoke方法替代run方法，并捕获输出结果
    result = chain.invoke({"input_documents": split_documents[:]})
    
    # 打印模型生成的摘要结果
    print("\n摘要结果:")
    print(result['output_text'])


if __name__ == "__main__":
    sequential_chain_example()