# 权重更新：W+BA，B、A是低秩矩阵
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
model = AutoModelForCausalLM.from_pretrained("../../pretrained_models/bloom-1b7", low_cpu_mem_usage=True)

# 配置文件,默认对query_key_value进行调整，可以通过改target_modules更改
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['query_key_value', 'dense_4h_to_h'],
                    modules_to_save=['word_embeddings'])  # modules_to_save:除了lora外还需训练哪些

# 创建模型
model = get_peft_model(model, config)

# 开始训练
args = TrainingArguments(
    output_dir="./saved_models",
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

# 加载训练好的模型
from peft import PeftModel
model = PeftModel.from_pretrained(model=model, model_id='./save_models/checkpoint-500')    # 这里的model是base的model，31行

# 模型推理
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
ipt = "Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: "  # 两个{}分别是instruction和input
print(pipe(ipt, max_length=256, do_sample=True, ))