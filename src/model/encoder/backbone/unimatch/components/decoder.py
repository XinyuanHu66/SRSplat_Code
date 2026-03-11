import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from ..components.utils import ResidualBlock, make_layer 

class DRAM(nn.Module):
    """Dual Residual Aggregation Module. Fuses low res and reference features."""

    def __init__(self, nf: int) -> None:
        super(DRAM, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        :param lr: [N, C, H, W]: A tensor of the low resolution features.
        :param ref_lr: [N, C, H * 2, W * 2]: A tensor of the reference features.
        :return: [N, C, H * 2, W * 2]: A tensor of the fused features
        """
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out


class SAM(nn.Module):
    """Spatial Adaptation Module. Remaps the distribution of of the extracted Ref features to that of the low res features."""

    def __init__(
        self, nf: int, use_residual: bool = True, learnable: bool = True
    ) -> None:
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(
                nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True), nn.ReLU(inplace=True)
            )
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        :param lr: [N, C, H, W]: A tensor of the low resolution features.
        :param ref_lr: [N, C, H, W]: A tensor of the reference features.
        :return: [N, C, H, W]: A tensor of the remaped reference features.
        """
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out


class DecoderMARINER(nn.Module):
    def __init__(
        self,
        nf: list[int] = [128, 128, 128],
        out_chl: int = 3,
        n_blks=[12, 8, 4],
        use_sam: bool = True,
        use_dram: bool = True,
    ) -> None:
        super(DecoderMARINER, self).__init__()

        block3 = functools.partial(ResidualBlock, nf=nf[1])
        block2 = functools.partial(ResidualBlock, nf=nf[2])
        block1 = functools.partial(ResidualBlock, nf=nf[2])

        self.merge_warp_x1 = nn.Conv2d(nf[0] * 2, nf[1], 3, 1, 1, bias=True)
        self.blk_x1 = make_layer(block3, n_blks[0])

        self.merge_warp_x2 = nn.Conv2d(nf[1], nf[2], 3, 1, 1, bias=True)
        self.blk_x2 = make_layer(block2, n_blks[1])

        self.blk_x4 = make_layer(block1, n_blks[2])

        # self.conv_out = nn.Conv2d(nf[2], out_chl, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

        if use_dram:
            self.fusion_x2 = DRAM(nf[1])
            self.fusion_x4 = DRAM(nf[2])
        else:
            self.fusion_x2 = nn.Conv2d(nf[1] * 2, nf[1], 3, 1, 1, bias=True)
            self.fusion_x4 = nn.Conv2d(nf[2] * 2, nf[2], 3, 1, 1, bias=True)

        if use_sam:
            self.sam_x1 = SAM(nf[0], use_residual=True, learnable=True)
            self.sam_x2 = SAM(nf[1], use_residual=True, learnable=True)
            self.sam_x4 = SAM(nf[2], use_residual=True, learnable=True)
        else:
            self.sam_x1 = lambda _, x: x
            self.sam_x2 = lambda _, x: x
            self.sam_x4 = lambda _, x: x
        self.use_dram = use_dram

    def forward(
        self, fea_r_l: list[torch.Tensor], fea_ref_l: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        :param fea_r_l: [[N, C_x4, H_r_x4, W_r_x4], [N, C_x2, H_r_x2, W_r_x2], [N, C_x1, H_r_x1, W_r_x1]]: A tensor of the encoded rendering.
        :param fea_ref_l: [[N, C_x4, H_r_x4, W_r_x4], [N, C_x2, H_r_x2, W_r_x2], [N, C_x1, H_r_x1, W_r_x1]]: A tensor of the encoded and warped reference.
        :return: [N, out_chl, W_r_x4, W_r_x4]: The decoded output image
        """
        warp_ref_x1 = self.sam_x1(fea_r_l[2], fea_ref_l[2])
        fea_x1 = self.act(
            self.merge_warp_x1(torch.cat([warp_ref_x1, fea_r_l[2]], dim=1))
        )
        fea_x1 = self.blk_x1(fea_x1)
        fea_x1_up = F.interpolate(
            fea_x1, scale_factor=2, mode="bilinear", align_corners=False
        )

        warp_ref_x2 = self.sam_x2(fea_x1_up, fea_ref_l[1])
        if self.use_dram:
            fea_x2 = self.fusion_x2(fea_x1, warp_ref_x2)
        else:
            fea_x2 = self.fusion_x2(torch.cat([fea_x1_up, warp_ref_x2], dim=1))
        fea_x2 = self.act(self.merge_warp_x2(fea_x2))
        fea_x2 = self.blk_x2(fea_x2)
        fea_x2_up = F.interpolate(
            fea_x2, scale_factor=2, mode="bilinear", align_corners=False
        )

        warp_ref_x4 = self.sam_x4(fea_x2_up, fea_ref_l[0])
        if self.use_dram:
            fea_x4 = self.fusion_x4(fea_x2, warp_ref_x4)
        else:
            fea_x4 = self.fusion_x4(torch.cat([fea_x2_up, warp_ref_x4], dim=1))
        fea_x4 = self.blk_x4(fea_x4)
        # out = self.conv_out(fea_x4)

        return fea_x4
