import torch
from torch import nn
from sequence_mask import sequence_mask

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带屏蔽的softmax交叉熵损失函数"""

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweight_loss = super().forward(pred.permute(0, 2, 1), label)
        weight_loss = (unweight_loss * weights).mean(dim=1)
        return weight_loss


if __name__ == "__main__":
    loss = MaskedSoftmaxCELoss()
    print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0])))
