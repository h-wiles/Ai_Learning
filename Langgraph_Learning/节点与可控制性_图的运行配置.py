import operator
from types import AsyncGeneratorType
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import START, END,StateGraph

model1 = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

model2 = ChatOpenAI(
    model="",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_URL"),
)

models = {"deepseek": model1, "openai": model2}

class AgentState(TypedDict):
    # message 是一个字段，它的类型是 Sequence[BaseMessage]，也就是一个存放 BaseMessage 对象的序列（比如列表）。
    # 对于序列来说，operator.add 的作用就是列表拼接。
    # 节点 A 返回 message = [msg1, msg2]，节点 B 返回 message = [msg3]，最终 message 的值会变成 [msg1, msg2, msg3]。
    message: Annotated[Sequence[BaseMessage], operator.add]

def _call_model(state: AgentState, config: RunnableConfig):
    model_name = config["configurable"].get("model", "deepseek")
    model = models[model_name]
    response = model.invoke(state["message"])
    return {
        "message": [response],
    }

builder = StateGraph(AgentState)
builder.add_node("model", _call_model)

builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
config = {"configurable": {"model": "deepseek"}}
print(graph.invoke({"message": [HumanMessage(content="你是谁")]}, config=config))