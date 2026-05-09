from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_deepseek import ChatDeepSeek
from langgraph.types import interrupt, Command
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
import os

@tool
def weather_search(query: str):
    """调用此函数浏览网络"""
    return f"我查询了{query}。结果：天气晴朗，温度25度"

tools = [weather_search]
tool_node = ToolNode(tools)
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

model = model.bind_tools(tools)
class State(MessagesState):
    """简单状态"""

def call_llm(state):
    return {"messages": [model.invoke(state["messages"])]}

def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]

    human_review = interrupt({"question": "这是正确的吗?", "tool_call": tool_call})

    review_action = human_review["action"]
    review_data = human_review.get("data")

    # 若批准，则调用工具
    if review_action == "continue":
        return Command(goto="run_tool")
    elif review_action == "update":
        update_message = {
            "role": "ai",
            "content": last_message.content,
            "tool_calls":[
                {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": review_data,
                }
            ],
            "id": last_message.id,
        }
        return Command(goto="run_tool", update={"messages": [update_message]})

    elif review_action == "feedback":
        tool_message = {
            "role": "tool",
            "content": review_data,
            "name": tool_call["name"],
            "tool_call_id": tool_call["id"]
        }
        return Command(goto="call_llm", update={"messages": [tool_message]})

def run_tool(state):
    new_messages = []
    tools = {"weather_search": weather_search}
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}

def router_after_llm(state) -> Literal[END, "human_review_node"]:
    if len(state["messages"][-1].tool_calls) == 0:
        return END
    else:
        return "human_review_node"

builder = StateGraph(MessagesState)
builder.add_node("call_llm", call_llm)
builder.add_node("run_tool", run_tool)
builder.add_node("human_review_node", human_review_node)

builder.add_edge(START, "call_llm")
builder.add_conditional_edges("call_llm", router_after_llm)
builder.add_edge("run_tool", "call_llm")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

input_message = {"messages": [("user", "北京那里的天气")]}
for event in graph.stream(input_message,
                          {"configurable": {"thread_id": "1"}}, stream_mode="updates"):
    print(event)
    print("\n")

for event in graph.stream(Command(resume={"action": "continue"}),
                          {"configurable": {"thread_id": "1"}}, stream_mode="updates"):
    print(event)
    print("\n")

# 改参数
for event in graph.stream(Command(resume={"action": "update", "city": "上海，中国"}),
                          {"configurable": {"thread_id": "1"}}, stream_mode="updates"):
    print(event)
    print("\n")