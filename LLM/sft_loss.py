import numpy as np

def sft_loss(logits, labels, ignore_index=-100):
    # 实现safe softmax,使用最大对数似然函数作为loss，等价于交叉熵
    shift_logits = logits-np.max(logits, axis=-1, keepdims=True)

    # 分母
    exp_logits = np.exp(shift_logits)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)

    # 计算log
    log_probs = shift_logits - np.log(sum_exp)

    seq_len = labels.shape[0]
    arrange_idx = np.arange(seq_len)

    clean_labels = np.where(labels==ignore_index, 0, labels)
    target_log_probs = log_probs[arrange_idx, clean_labels]

    mask = (labels!=ignore_index)
    masked_loss = -target_log_probs * mask

    actual_num_token = np.sum(mask)
    if actual_num_token > 0:
        final_loss = np.sum(masked_loss) / actual_num_token
        return final_loss
    else:
        return 0


if __name__ == '__main__':
    # mock data: 序列长度4，词表大小5
    # 假设前2个是prompt，后两个是answer
    test_logits = np.array([
        [2,1,1.0,0.5,0.1],
        [0.2,1.8,0.3,1.1,0.5],
        [0.5,0.1,2.1,0.4,0.8],
        [1.1,0.2,0.3,0.5,4.2]
    ])
    test_labels = np.array([-100,-100,2,4])
    loss = sft_loss(test_logits, test_labels)
    print(f"numpy计算的sft loss: {loss: .6f}")
