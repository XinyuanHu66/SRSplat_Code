import torch
import torch.nn as nn
from ..components.utils import MASA_weight_init
from .encoder import EncoderMARINER
from .decoder import DecoderMARINER
import pdb
# from .attention import PatchMatchAttentionMARINER


class ArchitectureMARINER(nn.Module):
    def __init__(
        self,
        # encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        attention: torch.nn.Module,
        in_out_skip: bool = True,
        weight_init: bool = True,
    ):
        super(ArchitectureMARINER, self).__init__()
        if weight_init:
            MASA_weight_init(self, scale=0.1)
        # self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.in_out_skip = in_out_skip


    def forward(self, fea_r_l, fea_ref_l):
        # fea_r_l = self.encoder(image)
        # fea_ref_l = self.encoder(ref)
        
        warp_fea_ref_l = self.attention(fea_r_l, fea_ref_l)
        
        
        out1 = self.decoder(fea_r_l, warp_fea_ref_l)
        
        if self.in_out_skip:
            # import pdb
            # pdb.set_trace()
            out1 = out1 + fea_r_l[0]
        

        return out1
