import torch
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=10, target_dim=1):
        """
        :param num_samples: number of samples
        :param input_dim: input dimension
        :param target_dim: target dimension
        """
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, input_dim)
        self.targets = torch.randn(num_samples, target_dim)

    def __len__(self):
        """返回数据集大小"""
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

dataset = RandomDataset(num_samples=100, input_dim=5, target_dim=2)

batch_size = 64
shuffle = True
num_workers = 0
drop_last = False   # 是否丢弃最后一个不足batch的数据

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    drop_last=drop_last,
)

print(f"数据集总样本数: {len(dataset)}")
print(f"每个batch大小: {batch_size}")
print(f"总batch数: {len(dataloader)}")

for epoch in range(2):
    print(f"\n Epoch {epoch+1}")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}: inputs shape={inputs.shape}, targets shape={targets.shape}")
        if batch_idx >= 2:
            break
