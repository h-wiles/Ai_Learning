import os
from openai import OpenAI
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "deepseek-demo"

# ========== 配置 ==========
# DeepSeek 客户端
deepseek_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

# LangSmith 客户端
langsmith_client = Client()

# ========== 目标函数（被评测的模型） ==========
def call_deepseek(inputs: dict) -> dict:
    """接收问题，返回模型生成的回答"""
    question = inputs["question"]
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个准确的问答助手，请直接给出答案，不要多余的解释。"},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return {"output": answer}

# ========== 数据集 ==========
dataset_name = "Simple QA Dataset"
# 如果数据集已存在，可以注释下面创建代码；否则第一次运行会创建
try:
    dataset = langsmith_client.create_dataset(dataset_name, description="简单问答测试集")
except:
    dataset = langsmith_client.read_dataset(dataset_name=dataset_name)

# 向数据集中添加示例（如果为空）
examples = [
    {"question": "中国的首都是哪里？", "answer": "北京"},
    {"question": "《红楼梦》的作者是谁？", "answer": "曹雪芹"},
    {"question": "Python 是编译型还是解释型语言？", "answer": "解释型"},
    {"question": "水的沸点是多少摄氏度（标准大气压下）？", "answer": "100"},
]
for ex in examples:
    try:
        langsmith_client.create_example(
            inputs={"question": ex["question"]},
            outputs={"answer": ex["answer"]},
            dataset_id=dataset.id,
        )
    except:
        pass  # 如果示例已存在则跳过

# ========== 3种评估器 ==========
# 1. 精确匹配评估器
def exact_match(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if prediction.strip() == reference.strip() else 0.0
    return {"key": "exact_match", "score": score}

# 2. 自定义检查：答案是否包含参考答案中的关键词（简单模糊匹配）
def contains_answer(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    score = 1.0 if reference.lower() in prediction.lower() else 0.0
    return {"key": "contains", "score": score}

# 3. 使用 LLM 作为评判者（更智能）
def llm_judge(run: Run, example: Example) -> dict:
    judge_client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    prompt = f"""请判断以下预测答案是否正确（是否与参考答案语义一致），只回答 "是" 或 "否"。
问题: {example.inputs["question"]}
参考答案: {example.outputs["answer"]}
预测答案: {run.outputs.get("output", "")}
正确？"""
    response = judge_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    verdict = response.choices[0].message.content.strip()
    score = 1.0 if "是" in verdict.lower() else 0.0
    return {"key": "llm_judge", "score": score}

# ========== 运行评估 ==========
evaluation_results = evaluate(
    call_deepseek,           # 第1个位置参数：target
    dataset_name,            # 第2个位置参数：data
    evaluators=[exact_match, contains_answer, llm_judge],
    experiment_prefix="deepseek_eval",
    num_repetitions=1,
    client=langsmith_client,
)

# 打印总结
print("评估完成！")
print(evaluation_results.to_pandas())