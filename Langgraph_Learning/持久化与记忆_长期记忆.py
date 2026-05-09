import pymongo
from typing import Literal
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
import os

client = pymongo.MongoClient("mongodb://localhost:27017")
try:
    client.admin.command("ping")
    print("mongo db链接成功")
except Exception as e:
    print("mongo db链接失败"+ e)

@tool
def get_weather(city: Literal["北京", "深圳"]):
    """用于返回天气信息的工具函数"""
    if city=="北京":
        return "北京天气晴朗"
    elif city=="深圳":
        return "深圳天气多云"
    else:
        return AssertionError("unknown city")

tools = [get_weather]
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# 使用mongodb进行长期记忆
MONGODB_URL = "localhost:27017"
with MongoDBSaver.from_conn_string(MONGODB_URL) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    response = graph.invoke({
        "messages": [("human", "北京今天天气如何？")]
    }, config)

print(response)