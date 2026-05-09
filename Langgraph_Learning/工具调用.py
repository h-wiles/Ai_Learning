from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
import os

@tool
def get_weather(location: str):
    """查询天气"""
    return f"我查询了{location}。结果：天气晴朗，温度25度"

@tool
def get_coldest():
    """查询最冷城市"""
    return "哈尔滨"

tools = [get_weather, get_coldest]
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
model.bind_tools(tools)
tool_node = ToolNode(tools)
print(model.invoke("深圳天气如何").tool_calls)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "北京"},
            "id": "tool_call_id",
        }
    ]
)

# 关键：提供 configurable.tools（即使和初始化时一样）
config = {"configurable": {"tools": tools}}
result = tool_node.invoke({"messages": [message_with_single_tool_call]}, config)
print(result)