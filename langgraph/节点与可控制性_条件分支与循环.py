import operator
from typing import Any, Annotated, Literal
from langgraph.errors import GraphRecursionError
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


graph_builder = StateGraph(State)
graph_builder.add_node(a)
graph_builder.add_node(b)

def router(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

graph_builder.add_edge(START, "a")
graph_builder.add_conditional_edges("a", router)   #
graph_builder.add_edge("b", "a")

graph = graph_builder.compile()

png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

try:
    res = graph.invoke({"aggregate": []},
                       config={"configurable": {"thread_id":"foo"},
                        "recursion_limit":4})     # recursion_limit参数防止大量无用调用
    print(res)
except GraphRecursionError:
    print("Recursion error")

