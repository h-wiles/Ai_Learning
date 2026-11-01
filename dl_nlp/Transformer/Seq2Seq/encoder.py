import torch
from torch import nn
from basic_structure import Encoder


class Seq2SeqEncoder(Encoder):
    """序列到序列学习的RNN编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        """output为每一个时刻最后一层的输出，state为最后一个单元的输出"""
        return output, state


if __name__ == '__main__':
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=20, num_layers=3)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print(output.shape, state.shape)
