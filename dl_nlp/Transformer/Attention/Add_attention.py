import torch
import torch.nn.functional as nn

def masked_softmax(X, valid_lens):
    """在最后一个轴上屏蔽元素来执行softmax操作"""
    if valid_lens is None:
        return nn.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim()==1:
            valid_lens = torch.repeat_interleave(valid_lens, shape)
        else:
            valid_lens = valid_lens.reshape(-1)          # 变为一个向量
        X =