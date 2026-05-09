import operator
from typing import Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]    # 给类型附加额外的“元数据”

def a(state:State):
    print(f"添加A到{state["aggregate"]}")
    return {
        "aggregate": ["A"],
    }

def b(state:State):
    print(f"添加B到{state["aggregate"]}")
    return {
        "aggregate": ["B"],
    }

def c(state:State):
    print(f"添加C到{state["aggregate"]}")
    return {
        "aggregate": ["C"],
    }

def d(state:State):
    print(f"添加D到{state["aggregate"]}")
    return {
        "aggregate": ["D"],
    }

graph_builder = StateGraph(State)
graph_builder.add_node(a)
graph_builder.add_node(b)
graph_builder.add_node(c)
graph_builder.add_node(d)

graph_builder.add_edge(START, "a")
graph_builder.add_edge("a", "b")
graph_builder.add_edge("a", "c")
graph_builder.add_edge("b", "d")
graph_builder.add_edge("c", "d")       # 分支
graph_builder.add_edge("d", END)

graph = graph_builder.compile()

png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

print(graph.invoke({"aggregate": []}, config={"configurable": {"thread_id":"foo"}}))
