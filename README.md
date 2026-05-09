# AI Learning Records

一个记录 AI 学习与实践过程的项目，主要包含：

- LangGraph 工作流
- MCP 工具协议
- PyTorch 深度学习
- Agent / RAG 实验
- 数学与 AI 结合研究

---

## 🧠 Tech Stack

- LangGraph
- MCP
- PyTorch
- FastAPI
- React
- Pandas / NumPy

---

## 📂 Project Structure

```bash
.
├── langgraph/     # LangGraph experiments
├── mcp/           # MCP server & client
├── pytorch/       # Deep learning experiments
├── notes/         # Study notes
└── README.md
```

---

## 🚀 LangGraph Example

```python
from langgraph.graph import StateGraph

graph = StateGraph(dict)

def chatbot(state):
    return {"msg": "hello"}

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")

app = graph.compile()
```

---

## 🔌 MCP Example

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")

@mcp.tool()
def hello(name: str):
    return f"hello {name}"

mcp.run()
```

---

## 🔥 PyTorch Example

```python
import torch
import torch.nn as nn

model = nn.Linear(1, 1)

x = torch.randn(10, 1)
y = model(x)

print(y.shape)
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📌 Learning Goals

- Agent Workflow
- RAG Systems
- Multi-Agent
- Rational Neural Networks
- AI + Mathematics

---

## ⭐ Notes

This repository is used for recording:

- Learning notes
- Experimental code
- AI workflow practice
- Deep learning research
