import torch
from torch import nn
from basic_structure import Decoder
from encoder import Seq2SeqEncoder

encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=20, num_layers=3)
class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)          # (时间步, batch_size, embed_size)
        X_and_context = torch.cat((X, context), dim=2)          # 在embed_size那个维度拼接
        output, state = self.rnn(X_and_context, state)            # decoder中的RNN, state为RNN的初始隐藏状态
        output = self.dense(output).permute(1, 0, 2)
        return output, state


if __name__ == '__main__':
    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=20, num_layers=3)
    decoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    enc_outputs = encoder(X)
    state = decoder.init_state(enc_outputs)
    output, state = decoder(X, state)
    print(output.shape, state.shape)
