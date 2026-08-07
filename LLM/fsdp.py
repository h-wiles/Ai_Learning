import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# 初始化分布式环境
def init_distribute():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

# 简化版FSDP封装器
class SimpleFSDP(nn.Module):
    def __init__(self, module:nn.Module):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()     # 当前进程rank
        self.world_size = dist.get_world_size()     # 总进程数

        # 对模型参数进行分片，每个进程只保留自己的分片
        self.param_shards = self._shard_parameters()

        # 临时存储组装后的完整参数（仅在计算时用，计算后释放）
        self.full_params = None

    def _shard_parameters(self):
        """
        将模型参数按维度分片，每个进程持有一部分
        """
        param_shards = {}
        for name, param in self.module.named_parameters():
            # 将参数按第一个维度均分，实际上FSDP会按照nume1分
            shard_size = param.size(0)  // self.world_size  # 计算每个进程的shard_size

            # 计算当前进程的分片范围
            start = self.rank * shard_size
            end = start + shard_size if self.rank != self.world_size - 1 else param.size(0)

            # 每个进程只保留自己的分片（深拷贝，避免引入完整参数）
            param_shards[name] = param.data[start:end].clone()

            # 释放原始参数的显存，只保留分片
            del param

        return param_shards

    def _gather_full_params(self):
        """
        通过all_gather组装完整参数（前向计算时调用）
        """
        self.full_params = {}
        for name, shard in self.param_shards.items():
            # 初始化完整参数的存储
            full_param = torch.zeros(
                self._get_full_param_size(name),
                dtype=shard.dtype,
                device=shard.device,
            )
            dist.all_gather(
                list(full_param.chunk(self.world_size, dim=0)),
                shard
            )
            self.full_params[name] = full_param

    def _get_full_param_size(self, name):
        """
        获取参数的完整尺寸。简化版，实际上可以通过通信获取
        """
        # 简化：假如所有进程知道完整参数尺寸，这里模拟原始参数尺寸
        if "weight" in name:
            return torch.Size([1024, 512])
        elif "bias" in name:
            return torch.Size([1024])
        return shard.size()

    def forward(self, x):
        # 1, 前向计算前：组装完整参数
        self._gather_full_params()

        # 2, 将完整参数赋值给模型，执行前向计算
        for name, param in self.module.named_parameters():
            param.data = self.full_params[name]

        out = self.module(x)

        # 3, 计算后释放完整参数，节省显存
        self.full_params = None
        return out

    def backward(self, loss):
        """
        反向传播+梯度分片聚合，简化版
        """
        # 1，反向计算梯度
        loss.backward()

        # 2, 对梯度进行分片，只保留自己的分片
        grad_shards = {}

        for name, param in self.module.named_parameters():
            shard_size = param.size(0) // self.world_size
            start = self.rank * shard_size
            end = start + shard_size if self.rank != self.world_size - 1 else param.grad.size(0)
            grad_shards[name] = param.grad[start:end].clone()

        # 3, reduce_scatter聚合梯度（各进程只保留自己分片的全局梯度）
        for name, grad_shard in grad_shards.items():
            dist.reduce_scatter(
                grad_shard,
                list(self._get_full_param_size(name).chunk(self.world_size, dim=0)),
                op=dist.ReduceOp.SUM,
            )

        # 4, 更新本地参数分片，优化器只更新分片
        self._update_param_shards(grad_shards)

    def _get_full_grad(self, name):
        """
        获取完整梯度，简化版
        """
        return self.module.named_parameters()[name].grad

    def _update_param_shards(self, grad_shards, lr=1e-3):
        """
        用聚合后的梯度分片更新本地参数分片
        """
        for name, shard in self.param_shards.items():
            shard.data -= lr * grad_shards[name]


if __name__ == '__main__':
    # 初始化分布式
    init_distribute()

    # 1，定义简单模型（线性层为例）
    model = nn.Linear(512, 1024).to(torch.device(f"cuda: {dist.get_rank()}")
                                    if torch.cuda.is_available() else torch.device(f"cpu"))

    # 2, 用FSDP封装模型
    fsdp_model = SimpleFSDP(model)

    # 3, 模拟前向+反向
    x = torch.randn(32, 512).to(next(fsdp_model.parameters()).device)
    out = fsdp_model(x)
    loss = out.sum()
    fsdp_model.backward(loss)
    print(f"Rank {fsdp_model.rank}: 参数分片尺寸{list(fsdp_model.param_shards['weight'].size())}")

