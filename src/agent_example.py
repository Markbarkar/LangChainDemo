import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# 加载环境变量
load_dotenv()

def agent_example():
    # 初始化 LLM (使用本地Ollama模型替代OpenAI)
    llm = OllamaLLM(
    model="deepseek-r1:70b", 
    temperature=0.7,
    base_url="http://192.168.1.6:11434"  # 替换为实际的内网Ollama服务器地址
    )

    # # 初始化聊天消息历史记录
    # history = ChatMessageHistory()

    # 消息列表
    messages = [
        HumanMessage(content="你好，我叫Bob"),
        AIMessage(content="你好，Bob！我是一个人工智能助手。有什么我可以帮助你的吗？"),
        HumanMessage(content="我叫什么名字？"),
    ]

    res = llm.invoke(messages)
    print(res)

def langgraph_example(workflow=None, query=None, config=None):
    # 初始化 LLM (使用本地Ollama模型替代OpenAI)
    llm = OllamaLLM(
        model="deepseek-r1:70b",
        temperature=0.7,
        base_url="http://192.168.1.6:11434"  # 替换为实际的内网Ollama服务器地址
    )

    # 定义一个新graph
    workflow = workflow if workflow else StateGraph(state_schema=MessagesState)

    prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个海盗，尽你所能回答问题。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
    # 定义调用模型的函数
    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = llm.invoke(prompt)
        return {"messages": response}

    # 定义简单的节点
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # 增加记忆
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)



    while True:
        query = input("请输入你的问题：")
        if query == "q":
            break
        if query:
            config = {"configurable": {"thread_id": "abc123"}}
            input_messages = [HumanMessage(query)]
            output = app.invoke({"messages": input_messages}, config)
            output["messages"][-1].pretty_print()  # 输出在state中的消息

    # config = {"configurable": {"thread_id": "abc123"}}
    # query = "你好！我叫bob"

    # input_messages = [HumanMessage(query)]
    # output = app.invoke({"messages": input_messages}, config)
    # output["messages"][-1].pretty_print()  # 输出在state中的消息

if __name__ == "__main__":
    # agent_example()

    # config = {"configurable": {"thread_id": "abc123"}}
    # query = "你好！我叫bob"
    # # query = "我叫什么名字？"

    langgraph_example()
    # langgraph_example(workflow=workflow, query=query, config=config)