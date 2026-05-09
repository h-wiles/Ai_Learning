from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_deepseek import ChatDeepSeek
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
import os


@tool
def search(query: str):
    """调用此函数浏览网络"""
    return f"我查询了{query}。结果：天气晴朗，温度25度"

tools = [search]
tool_node = ToolNode(tools)
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

class AskHuman(BaseModel):
    question: str

model = model.bind_tools(tools + [AskHuman])

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    else:
        return "action"

def call_model(state):
    message = state["messages"]
    response = model.invoke(message)
    return {
        "messages": [response]
    }

def ask_human(state):
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    ask = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])
    location = interrupt(ask.question)
    tool_messages = {"tool_call_id": tool_call_id, "type": "tool", "content": location}
    return {
        "messages": [tool_messages]
    }

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("action", tool_node)
builder.add_node("ask_human", ask_human)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, [END, "ask_human", "action"])
builder.add_edge("ask_human", "agent")
builder.add_edge("action", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["ask_human"])    # 在运行到ask_human前终止

input_message = {"messages": [("user", "询问用户在哪里，然后查询那里的天气")]}
for event in graph.stream(input_message,
                          {"configurable": {"thread_id": "1"}}, stream_mode="values"):
    event["messages"][-1].pretty_print()

graph.update_state({"configurable": {"thread_id": "1"}}, {"input": "你好"})    # 人工介入，更新状态
# 重新运行
for event in graph.stream(None,{"configurable": {"thread_id": "1"}}, stream_mode="values"):
    event["messages"][-1].pretty_print()
