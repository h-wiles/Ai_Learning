import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ---------- 简化版 FSDP ----------
class MyFSDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 保存每个参数的完整形状和分片副本
        self.full_shapes = {}
        self.shard_backup = {}   # 每次 forward 前备份分片数据，用于 backward 后恢复

        # 替换模型参数为分片
        for name, param in module.named_parameters():
            full_shape = param.shape
            self.full_shapes[name] = full_shape
            if full_shape[0] % self.world_size != 0:
                raise ValueError(f"Parameter {name} first dimension {full_shape[0]} not divisible by world_size {self.world_size}")

            # 计算当前 rank 的分片
            shard_size = full_shape[0] // self.world_size
            start = self.rank * shard_size
            end = start + shard_size
            shard_data = param.data[start:end].clone()

            # 用分片数据替换原参数
            param.data = shard_data   # 此时 param 变为分片

    def _gather_full_params(self):
        """收集所有分片，组装完整参数并临时赋值给模型"""
        for name, param in self.module.named_parameters():
            full_shape = self.full_shapes[name]
            # 创建完整大小的零张量
            full_tensor = torch.zeros(full_shape, dtype=param.dtype, device=param.device)
            # 将 full_tensor 按第一维切成 world_size 个块，用于接收
            tensor_list = list(full_tensor.chunk(self.world_size, dim=0))
            # All-Gather：每个进程把自己的分片填入对应块
            dist.all_gather(tensor_list, param)
            # 将完整张量赋给参数（替换原分片）
            param.data = full_tensor

    def _restore_sharded_params(self):
        """恢复参数为分片（使用备份的分片数据）"""
        for name, param in self.module.named_parameters():
            # 恢复为 forward 前备份的分片数据
            param.data = self.shard_backup[name]

    def forward(self, x):
        # 1. 备份当前分片数据（用于反向后恢复）
        for name, param in self.module.named_parameters():
            self.shard_backup[name] = param.data.clone()

        # 2. 收集完整参数并赋值
        self._gather_full_params()

        # 3. 执行前向计算
        out = self.module(x)

        # 注意：这里不恢复分片，因为反向传播需要完整参数计算梯度
        return out

    def backward(self, loss):
        """自定义反向传播：计算梯度，聚合梯度，恢复分片"""
        # 1. 反向传播，此时参数是完整的，梯度会计算到 param.grad
        loss.backward()

        # 2. 对每个参数进行梯度分片聚合
        for name, param in self.module.named_parameters():
            # 完整梯度
            full_grad = param.grad
            if full_grad is None:
                continue

            # 切分成 world_size 块
            grad_chunks = list(full_grad.chunk(self.world_size, dim=0))

            # 创建输出张量（分片大小）
            shard_size = self.full_shapes[name][0] // self.world_size
            shard_grad = torch.zeros(
                (shard_size, *self.full_shapes[name][1:]),
                dtype=full_grad.dtype,
                device=full_grad.device
            )

            # Reduce-Scatter：各进程将自己的块列表聚合到对应 rank，得到自己的分片梯度
            dist.reduce_scatter(shard_grad, grad_chunks, op=dist.ReduceOp.SUM)

            # 3. 将分片梯度赋值给参数（param 当前还是完整张量，但 grad 会被替换）
            #    同时恢复参数为分片数据（用备份）
            param.grad = shard_grad
            param.data = self.shard_backup[name]

        # 清理备份，释放显存（可选）
        self.shard_backup.clear()


# ---------- 简单模型和数据 ----------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)


class DummyDataset(Dataset):
    def __init__(self, size=1000, dim=512):
        self.data = torch.randn(size, dim)
        self.target = torch.randn(size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# ---------- 主训练函数 ----------
def train(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 模型
    model = SimpleModel().cuda(rank)
    fsdp_model = MyFSDP(model)

    # 优化器（直接传入分片参数）
    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)

    # 数据集和分布式采样器
    dataset = DummyDataset(size=200, dim=512)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 训练循环
    for epoch in range(2):
        sampler.set_epoch(epoch)   # 确保 shuffle 不同
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(rank), y.cuda(rank)

            # 前向
            out = fsdp_model(x)
            loss = nn.functional.mse_loss(out, y)

            # 反向（自定义）
            optimizer.zero_grad()   # 清空旧梯度
            fsdp_model.backward(loss)

            # 优化器更新分片参数
            optimizer.step()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

    # 清理
    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()