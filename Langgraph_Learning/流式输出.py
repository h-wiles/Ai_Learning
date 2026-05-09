from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek
import os

class State(TypedDict):
    topic: str
    joke: str

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

def refine_topic(state: State):
    return {"topic": state["topic"] + "和小狗"}

def generate_joke(state: State):
    input_msg = [{"role": "user", "content": f"生成一个关于{state["topic"]}的笑话"}]
    response = model.invoke(input_msg)
    return {"joke": response.content}

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# 会返回所有的值
for chunk in graph.stream({"topic":"冰淇淋"}, stream_mode="values"):
    print(chunk)

# 同样topic只返回最后更改的值
for chunk in graph.stream({"topic":"冰淇淋"}, stream_mode="updates"):
    print(chunk)

# debug模式输出所有的值
for chunk in graph.stream({"topic":"冰淇淋"}, stream_mode="debug"):
    print(chunk)

# 只返回消息，打字机效果
for message_chunk, metadata in graph.stream({"topic":"冰淇淋"}, stream_mode="messages"):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)