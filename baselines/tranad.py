import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerEncoder
from .dlutils import (
    ConvLSTM,
    PositionalEncoding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class TranAD(nn.Module):
    def __init__(self, feats: int, n_window: int):
        super(TranAD, self).__init__()
        self.name = "TranAD"
        self.epoch_n = 1
        self.forward_n = 1
        self.batch = 128
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        # src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src):

        if self.training:
            self.forward_n += 1

        src_shape = src.shape
        tgt = src.clone().detach().requires_grad_(True)
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src).double()
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        c = c.double()
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1.view(src_shape), x2.view(src_shape)
