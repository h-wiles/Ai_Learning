from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import os

@tool
def play_song_on_qq(song:str):
    """在qq音乐上播放歌曲"""
    return f"成功在qq音乐上播放{song}"

@tool
def play_song_on_163(song:str):
    """在网易云音乐上播放歌曲"""
    return f"成功在网易云音乐上播放{song}"

tools = [play_song_on_163, play_song_on_qq]

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
model.bind_tools(tools)
tool_node = ToolNode(tools)

def should_continue(state):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [response]
    }

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"end": END, "continue": "action"})
workflow.add_edge("action", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

input_messages = HumanMessage(content="已经提供了播放歌曲的工具，调用工具播放一首周杰伦的歌曲")
for event in graph.stream({"messages":input_messages}, {"configurable":{"thread_id":"1"}}, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 查看记录重放
config = {"configurable":{"thread_id":"1"}}
print(graph.get_state(config).values["messages"])
all_state = []
for state in graph.get_state_history(config):
    all_state.append(state)

to_replay = all_state[2]
print(to_replay.next)

# 从to_replay节点重新执行
for event in graph.stream(None, to_replay.config):
    for v in event.values():
        print(v)


# 分叉执行
last_message = to_replay.values["messages"][-1]
last_message.tool_calls[0]["name"] = "play_song_on_163"     # 改这个节点使用的工具

branch_config = graph.update_state(
    to_replay.config,
    {"messages": [last_message]},
)