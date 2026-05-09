from typing import Literal
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
import os

memory = MemorySaver()

@tool
def search(query: str):
    """调用函数可以浏览网络"""
    return "北京天气晴朗"

tools = [search]
tool_node = ToolNode(tools)

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
bound_model = model.bind_tools(tools)

def should_continue(state: MessagesState):
    """返回下一个要执行的节点"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:        # 没有函数调用则结束
        return END
    else:
        return "action"

def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    return {
        "messages": response,
    }

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)      # 工具调用作为一个节点

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue,
                               ["action", END])     # 这个条件边可能去往的所有节点
workflow.add_edge("action", "agent")

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "2"}}    # 线程id是1，不同的id不会共享信息
input_messages = {"role": "user", "content": "hi,我是wiles"}
for chunk in app.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_messages = HumanMessage(content="我叫什么名字")
for chunk in app.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
