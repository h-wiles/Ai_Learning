import torch
from torch import nn
from peft import LoraConfig, get_peft_model, PeftModel

# 1. 自定义模型适配
net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
config = LoraConfig(target_modules=["0"])
model1 = get_peft_model(net1, config)


# 2. 多适配器加载与切换
net2 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
config1 = LoraConfig(target_modules=["0"])
model2 = get_peft_model(net2, config1)
model2.save_pretrained("./loraA")        # 存两个不同的Lora，其中一个为LoraA

config2 = LoraConfig(target_modules=["2"])
model2 = get_peft_model(net2, config2)
model2.save_pretrained("./loraB")       # # 存两个不同的Lora，其中一个为LoraB

# 自定义模型读取与选择特定Lora
model = PeftModel.from_pretrained(net2, model_id="./loraA/", adapter_name="loraA")
model.load_adapter("./loraB/", adapter_name="loraB")  # 加入另一个Lora
print("当前适配器：", model.active_adapter)

model.set_adapter("loraB")   # 切换适配器到LoraB
print("当前适配器：", model2.active_adapter)

# 3. 禁用适配器
model.set_adapter("loraA")
with model.disable_adapter():
    print(model(torch.arange(0, 10).view(1, 10).float()))