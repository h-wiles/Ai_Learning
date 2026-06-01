import os
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# 设置 LangSmith 环境变量
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "deepseek-demo"

# 包装 OpenAI 客户端（DeepSeek 兼容）
client = wrap_openai(OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
))

@traceable(name="deepseek_chat")
def chat_with_deepseek(prompt: str):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# 调用并自动追踪
result = chat_with_deepseek("如何写出高效的代码？")
print(result)