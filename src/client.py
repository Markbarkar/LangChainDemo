import uuid
from langserve import RemoteRunnable
from langchain.schema import HumanMessage, AIMessage

def format_chat_history(messages):
    return "\n".join([f"{'用户' if isinstance(m, HumanMessage) else 'Tiko'}: {m.content}" for m in messages])

def chat_session():
    session_id = str(uuid.uuid4())
    runnable = RemoteRunnable("http://127.0.0.1:8000/chat")
    messages = []
    
    print("开始与Tiko对话（输入'退出'结束对话）...")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['退出', 'quit', 'exit']:
            break
            
        messages.append(HumanMessage(content=user_input))
        response = runnable.invoke({
            "chat_history": messages,
            "input": user_input,
            "config": {"configurable": {"session_id": session_id}}
        })
        
        ai_message = AIMessage(content=str(response))
        messages.append(ai_message)
        print(f"\nTiko: {ai_message.content}")

if __name__ == "__main__":
    chat_session()