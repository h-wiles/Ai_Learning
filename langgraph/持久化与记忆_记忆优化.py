from typing import Literal
from langchain_deepseek import ChatDeepSeek
from typing_extensions import TypedDict, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
import os

memory = MemorySaver()

class State(MessagesState):
    summary: str

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

def should_continue(state: State) -> Literal["summarize", END]:
    """返回下一个要执行的节点"""
    if len(state["messages"])>2:        # 超过2次就对之前对话总结
        return "summarize"
    else:
        return END

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"这是迄今为止对话的摘要: {summary} \n\n"
            "考虑上面的新消息，扩展摘要:"
        )
    else:
        summary_message = "创建上述对话的摘要："
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]       # 删掉最后两条之外的消息
    return {
        "summary": response.content,
        "messages": delete_messages
    }

def filter_messages(messages: State):
    # 永远只记忆前一条消息，失去了记忆功能
    return messages[-1:]

def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"之前的对话摘要: {summary}"
        messages = state["messages"] + [SystemMessage(content=system_message)]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [response],
    }

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize", summarize_conversation)      # 工具调用作为一个节点

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue,
                               ["summarize", END])     # 这个条件边可能去往的所有节点
workflow.add_edge("summarize", END)

app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "5"}}    # 线程id是1，不同的id不会共享信息
input_messages = {"role": "user", "content": "hi,我是wiles"}
for chunk in app.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_messages = HumanMessage(content="我叫什么名字")
for chunk in app.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_messages = HumanMessage(content="你是谁")
for chunk in app.stream({"messages": [input_messages]},config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
