import torch

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    # print(mask)
    X[~mask] = value
    return X


if __name__ == "__main__":
    X = torch.tensor([[1,2,3], [4,5,6]])
    sequence_mask(X, torch.tensor([1,2]))