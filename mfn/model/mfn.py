from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from mfn.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class MFN(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int = 114,
    ):
        super().__init__()

        # 分别定义两个独立的encoder
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.gram_encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        
        # 定义decoder
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        img: FloatTensor, 
        img_mask: LongTensor, 
        tgt: LongTensor,
        gram_img: FloatTensor, 
        gram_img_mask: LongTensor
    ) -> FloatTensor:
        # 对原始图像编码
        feature, mask = self.encoder(img, img_mask)  # [b, t, d] #b,h,w,d
        format_feature, format_mask = self.gram_encoder(gram_img, gram_img_mask)  # [b, t', d] #b,h,w,d
        fused_feature = torch.cat([feature, format_feature], dim=1)  # [b, t, 2d] #b,h,w,d
        fused_mask = torch.cat((mask, format_mask), dim=1) #b,h,w

        # 与原先逻辑一致，将特征复制两份
        fused_feature = torch.cat((fused_feature, fused_feature), dim=0)  # [2b, t, d] # 2b, w, h, d
        fused_mask = torch.cat((fused_mask, fused_mask), dim=0)          # [2b, t], # 2b, w, h

        exp_out, imp_out, fusion_out = self.decoder(fused_feature, fused_mask, tgt) #  [2b, l, vocab_size]
        return exp_out, imp_out, fusion_out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        gram_img: FloatTensor,
        gram_img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        # 编码原始图像
        feature, mask = self.encoder(img, img_mask)
        format_feature, format_mask = self.gram_encoder(gram_img, gram_img_mask)
        fused_feature = torch.cat([feature, format_feature], dim=1)  # [b, t, 2d] #b,w,h,d
        fused_mask = torch.cat((mask, format_mask), dim=1) #b,w,h

        # fused_mask = mask & format_mask
        return self.decoder.beam_search(
            [fused_feature], [fused_mask], beam_size, max_len, alpha, early_stopping, temperature
        )
