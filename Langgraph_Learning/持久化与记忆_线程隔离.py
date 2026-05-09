import operator
from types import AsyncGeneratorType
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, END,StateGraph, MessagesState

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)
graph = builder.compile()

# 没有激活持久化层，无法进行多轮对话
input_messages = {"role": "user", "content": "hi,我是wiles"}
for chunk in graph.stream({"messages": [input_messages]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_messages = {"role": "user", "content": "我叫什么名字"}
for chunk in graph.stream({"messages": [input_messages]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


# 激活持久化层，需要加上config才能使用
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}    # 线程id是1，不同的id不会共享信息
input_messages = {"role": "user", "content": "hi,我是wiles"}
for chunk in graph.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_messages = {"role": "user", "content": "我叫什么名字"}
for chunk in graph.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
