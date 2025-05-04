from langserve import RemoteRunnable

if __name__ == "__main__":
    runnable = RemoteRunnable("http://127.0.0.1:8000/chain")
    output = runnable.invoke({
        "topic": "人工智能"
    })
    print(output)