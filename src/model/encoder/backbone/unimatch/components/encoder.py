import torch
import torch.nn as nn
import functools
from ..components.utils import ResidualBlock, make_layer

class EncoderMARINER(nn.Module):
    def __init__(
        self,
        in_chl: int = 3,
        nf: list[int] = [64, 64, 64],
        n_blks=[4, 4, 4],
        act: str = "relu",
    ) -> None:
        super(EncoderMARINER, self).__init__()

        block1 = functools.partial(ResidualBlock, nf=nf[0])
        block2 = functools.partial(ResidualBlock, nf=nf[1])
        block3 = functools.partial(ResidualBlock, nf=nf[2])

        self.conv_L1 = nn.Conv2d(in_chl, nf[0], 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block1, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf[0], nf[1], 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block2, n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf[1], nf[2], 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block3, n_layers=n_blks[2])

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        :param x: [N, in_chl, H, W]: A tensor to be encoded.
        :return: [[N, nf[0], H, W], [N, nf[1], H / 2, W / 2], [N, nf[2], H / 4, W / 4]]
        """
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))

        return [fea_L1, fea_L2, fea_L3]
