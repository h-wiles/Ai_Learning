from torch import nn


class Encoder(nn.Module):
    """编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        """state为编码器的输出，X为解码器接受的额外输入"""
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    """整体架构"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_output = self.encoder(enc_X, *args)        # 编码器的输出
        dec_state = self.decoder.init_state(enc_output, *args)       # 加入解码器的初始状态（在编码器输出上做的）
        return self.decoder(dec_X, dec_state)


