# BitFit微调方式：只微调模型的bias层参数，其他参数的require_grad=False
from transformers import pipeline
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

ds = Dataset.load_from_disk("./data/alpaca_data_zh/")
tokenizer = AutoTokenizer.from_pretrained("../../pretrained_models/bloom-1b7")
def process_func(example):
    """数据处理"""
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
tokenizer.decode(tokenized_ds[1]["input_ids"])
model = AutoModelForCausalLM.from_pretrained("../../pretrained_models/bloom-1b7", low_cpu_mem_usage=True)

# 选择模型参数里面的所有不是bias部分,不计算梯度
num_param = 0
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()
print(num_param / sum(param.numel() for param in model.parameters()))   # 需要更新梯度的参数量

# 开始训练
args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# 模型推理
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: "  # 两个{}分别是instruction和input
print(pipe(ipt, max_length=256, do_sample=True, ))
