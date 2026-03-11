import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from ..components.utils import (
    make_block,
    find_best_k_ref_patch_centers_within_blocks,
    transfer,
    find_best_k_ref_block_centers,
    find_best_matching_block,
)
# from .utilss.visualizations import LineVisualization


class PathMatchAttention(nn.Module):
    visualization = None


class PatchMatchAttentionMARINER(PathMatchAttention):
    def __init__(
        self,
        patch_size: int = 3,
        r_block_size: int = 8,
        rel_ref_block_size: float = 1.5,
        dilations: list[int] = [1, 2, 3],
        visualize: bool = True,
        dynamic_patch_block_size: bool = True,
    ):
        super(PatchMatchAttentionMARINER, self).__init__()
        self.patch_size = patch_size
        self.r_block_size = r_block_size
        self.rel_ref_block_size = rel_ref_block_size
        self.dilations = dilations
        self.visualization = None
        self.visualize = visualize
        self.dynamic_patch_block_size = dynamic_patch_block_size

    def forward(self, fea_r_l: list[torch.Tensor], fea_ref_l: list[torch.Tensor]) -> list[torch.Tensor]:
        """Finds patch correlation between the deepest rendering features (fea_r_l[2]) and deepest reference features (fea_ref_l[2]).
        Transfers then the patches with the best correlations to the rendering features.

        :param fea_r_l: [[N, C_x4, Hr_x4, Wr_x4], [N, C_x2, Hr_x2, Wr_x2], [N, C_x1, Hr_x1, Wr_x1]]: A list of feature tensors from the encoded rendering.
        :param fea_ref_l: [[N, C_x4, Href_x4, Wref_x4], [N, C_x2, Href_x2, Wref_x2], [N, C_x1, Href_x1, Wref_x1]]: A list of feature tensors from the encoded reference.
        :return: [[N, C_x4, Hr_x4, Wr_x4], [N, C_x2, Hr_x2, Wr_x2], [N, C_x1, Hr_x1, Wr_x1]]: A list of warped reference feature tensors.
        """

        ## Stage 1: Find best matching ref block for every r block
        N, C_x1, Hr_x1, Wr_x1 = fea_r_l[2].size()
        _, C_x2, Hr_x2, Wr_x2 = fea_r_l[1].size()
        _, C_x4, Hr_x4, Wr_x4 = fea_r_l[0].size()

        if self.dynamic_patch_block_size:
            self.r_block_size = min(Hr_x1, Wr_x1) // 8  # 5
            self.patch_size = round(self.r_block_size / 2)
            if self.patch_size % 2 == 0:
                self.patch_size -= 1

        n_r_blocks_W = Wr_x1 // self.r_block_size
        n_r_blocks_H = Hr_x1 // self.r_block_size
        B = n_r_blocks_H * n_r_blocks_W  # Total number of blocks
        W_r_block = Wr_x1 // n_r_blocks_W
        H_r_block = Hr_x1 // n_r_blocks_H

        block_padding = self.patch_size // 2

        # Ref block size is scaled render block size
        _, _, Href_x1, Wref_x1 = fea_ref_l[2].size()
        W_ref_block = (
            2 * int(Wref_x1 // (2 * n_r_blocks_W) * self.rel_ref_block_size) + 1
        )
        H_ref_block = (
            2 * int(Href_x1 // (2 * n_r_blocks_H) * self.rel_ref_block_size) + 1
        )

        # The padding is needed and removed by find_best_k_ref_patch_centers_within_blocks
        r_blocks = F.pad(
            fea_r_l[2],
            pad=(block_padding, block_padding, block_padding, block_padding),
            mode="replicate",
        )  # [N, C_x1, Hr_x1 + 2, Wr_x1 + 2]
        r_blocks = F.unfold(
            r_blocks,
            kernel_size=(H_r_block + 2 * block_padding, W_r_block + 2 * block_padding),
            padding=(0, 0),
            stride=(H_r_block, W_r_block),
        )  # [N, C_x1 * (H_r_block + 2) * (W_r_block + 2), B]
        r_blocks = r_blocks.view(
            N,
            C_x1,
            H_r_block + 2 * block_padding,
            W_r_block + 2 * block_padding,
            n_r_blocks_H * n_r_blocks_W,
        ).permute(
            0, 4, 1, 2, 3
        )  # [N, B, C_x1, H_r_block + 2, W_r_block + 2] Blocks overlap by one pixel because of padding.

        ## Finds for each lr block the ref block center where the center patch has the highest correlation
        sorted_corr, ind_l = find_best_k_ref_block_centers(
            r_blocks,
            fea_ref_l[2],
            k=1,
            patch_size=self.patch_size,
            patch_stride=1,
            patch_dilations=self.dilations,
        )  #  [N, B, k]

        # Crop a block with [W_ref_block + 2, H_ref_block + 2] size around each pixel index.
        # Transforms indexes from [H * W] to [H, W]
        index = ind_l[:, :, 0]  # [N, B]
        idx_w = index % Wref_x1
        idx_h = index // Wref_x1

        # Calculates block indexes
        # There is a padding added here, which is later needed and removed in find_best_k_ref_patch_centers_within_blocks
        idx_w1 = idx_w - W_ref_block // 2 - block_padding  # left
        idx_w2 = idx_w + W_ref_block // 2 + block_padding  # right
        idx_h1 = idx_h - H_ref_block // 2 - block_padding  # up
        idx_h2 = idx_h + H_ref_block // 2 + block_padding  # down

        # Visualize the attention
        if self.visualize:
            src_w = torch.arange(W_r_block // 2, n_r_blocks_W * W_r_block, W_r_block)
            src_h = torch.arange(H_r_block // 2, n_r_blocks_H * H_r_block, H_r_block)
            hgrid, wgrid = torch.meshgrid(src_h, src_w, indexing="ij")
            src_points = torch.stack(
                [hgrid.contiguous().view(-1), wgrid.contiguous().view(-1)], dim=-1
            ).expand(N, -1, -1)
            confidence = sorted_corr[:, :, 0]
            dst_points = torch.stack([idx_h, idx_w], dim=-1)
            self.visualization = LineVisualization(
                src_points, dst_points, confidence, "plasma", 0, 3
            )

        # Clamp Block indices and shift blocks to ensure (W_ref_block+2, H_ref_block+2) sized blocks
        mask = (idx_w1 < 0).long()
        idx_w1 = idx_w1 * (1 - mask)
        idx_w2 = idx_w2 * (1 - mask) + (W_ref_block + 2 * block_padding - 1) * mask

        mask = (idx_w2 > Wref_x1 - 1).long()
        idx_w2 = idx_w2 * (1 - mask) + (Wref_x1 - 1) * mask
        idx_w1 = (
            idx_w1 * (1 - mask)
            + (idx_w2 - (W_ref_block + 2 * block_padding - 1)) * mask
        )

        mask = (idx_h1 < 0).long()
        idx_h1 = idx_h1 * (1 - mask)
        idx_h2 = idx_h2 * (1 - mask) + (H_ref_block + 2 * block_padding - 1) * mask

        mask = (idx_h2 > Href_x1 - 1).long()
        idx_h2 = idx_h2 * (1 - mask) + (Href_x1 - 1) * mask
        idx_h1 = (
            idx_h1 * (1 - mask)
            + (idx_h2 - (H_ref_block + 2 * block_padding - 1)) * mask
        )

        # For every top left index w1, h1, the indexes of the entire block are created. ind_w = [w1, w1+1, .. w1+W_ref_block+2, w1, w1+1, .. w1+W_ref_block+2, ... block height times].
        # All ind_w are concatenated over all dimensions
        ind_h_x1, ind_w_x1 = make_block(
            idx_w1,
            idx_h1,
            W_ref_block + 2 * block_padding,
            H_ref_block + 2 * block_padding,
            1,
        )  # [N * B * (W_ref_block + 2) * 1 * (H_ref_block + 2) * 1]
        ind_h_x2, ind_w_x2 = make_block(
            idx_w1,
            idx_h1,
            W_ref_block + 2 * block_padding,
            H_ref_block + 2 * block_padding,
            2,
        )  # [N * B * (W_ref_block + 2) * 2 * (H_ref_block + 2) * 2]
        ind_h_x4, ind_w_x4 = make_block(
            idx_w1,
            idx_h1,
            W_ref_block + 2 * block_padding,
            H_ref_block + 2 * block_padding,
            4,
        )  # [N * B * (W_ref_block + 2) * 4 * (H_ref_block + 2) * 4]

        # Batch index of the blocks: every batch index is repeated B*(H_ref_block+2)*scale * (W_ref_block+2)*scale times
        ind_batch = torch.repeat_interleave(
            torch.arange(0, N, dtype=torch.long, device=idx_w1.device),
            n_r_blocks_H
            * n_r_blocks_W
            * (H_ref_block + 2 * block_padding)
            * (W_ref_block + 2 * block_padding),
        )  # [B * (H_ref_block + 2) * 1 * (W_ref_block +2 ) * 1]
        ind_batch_x2 = torch.repeat_interleave(
            torch.arange(0, N, dtype=torch.long, device=idx_w1.device),
            n_r_blocks_H
            * n_r_blocks_W
            * ((H_ref_block + 2 * block_padding) * 2)
            * ((W_ref_block + 2 * block_padding) * 2),
        )  # [B * (H_ref_block + 2) * 2 * (W_ref_block + 2) * 2]
        ind_batch_x4 = torch.repeat_interleave(
            torch.arange(0, N, dtype=torch.long, device=idx_w1.device),
            n_r_blocks_H
            * n_r_blocks_W
            * ((H_ref_block + 2 * block_padding) * 4)
            * ((W_ref_block + 2 * block_padding) * 4),
        )  # [B * (H_ref_block + 2) * 4 * (W_ref_block + 2) * 4]

        # Stores for every rendering block the best matching ref block. The blocksize depends on the scale.
        # tensor[[index_list], :, :] copies the dimensions in the index_list into the output tensor
        ref_blocks_x1 = (
            fea_ref_l[2][ind_batch, :, ind_h_x1, ind_w_x1]
            .view(
                N * B,
                H_ref_block + 2 * block_padding,
                W_ref_block + 2 * block_padding,
                C_x1,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [N * B, C_x1, H_ref_block + 2, W_ref_block + 2]
        ref_blocks_x2 = (
            fea_ref_l[1][ind_batch_x2, :, ind_h_x2, ind_w_x2]
            .view(
                N * B,
                (H_ref_block + 2 * block_padding) * 2,
                (W_ref_block + 2 * block_padding) * 2,
                C_x2,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [N * B, C_x2, 2 * (H_ref_block + 2), 2 * (W_ref_block + 2)]
        ref_blocks_x4 = (
            fea_ref_l[0][ind_batch_x4, :, ind_h_x4, ind_w_x4]
            .view(
                N * B,
                (H_ref_block + 2 * block_padding) * 4,
                (W_ref_block + 2 * block_padding) * 4,
                C_x4,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # [N * B, C_x4, 4 * (H_ref_block + 2), 4 * (W_ref_block + 2)]

        ## Stage 2: Within blocks patch matching
        # calculate correlation between rendering patches and their corresponding ref patches within the blocks
        r_blocks = r_blocks.contiguous().view(
            N * B,
            C_x1,
            H_r_block + 2 * block_padding,
            W_r_block + 2 * block_padding,
        )
        corr_all_l, index_all_l = find_best_k_ref_patch_centers_within_blocks(
            r_blocks, ref_blocks_x1, 1, patch_size=self.patch_size, patch_stride=1
        )  # [N * B, H_r_block, W_r_block, k]
        index_all = index_all_l[:, :, :, 0]  # [N * B, H_r_block, W_r_block]
        soft_att_all = corr_all_l[:, :, :, 0:1].permute(
            0, 3, 1, 2
        )  # [N  * B, 1, H_r_block, W_r_block]

        ## Stage 3: Transfer of the best matching patches on all scale features
        # Transfers the features from the reference block to the rendering block
        warp_ref_patches_x1 = transfer(
            ref_blocks_x1,
            index_all,
            soft_att_all,
            patch_size=self.patch_size,
            patch_fold_padding=self.patch_size // 2,
            scale=1,
        )  # [N * B, C_x1, H_r_block, W_r_block]
        warp_ref_patches_x2 = transfer(
            ref_blocks_x2,
            index_all,
            soft_att_all,
            patch_size=self.patch_size * 2,
            patch_fold_padding=self.patch_size // 2 * 2,
            scale=2,
        )  # [N * B, C_x2, H_r_block * 2, W_r_block * 2]
        warp_ref_patches_x4 = transfer(
            ref_blocks_x4,
            index_all,
            soft_att_all,
            patch_size=self.patch_size * 4,
            patch_fold_padding=self.patch_size // 2 * 4,
            scale=4,
        )  # [N * B, C_x4, H_r_block * 4, W_r_block * 4]

        # Reassembles the blocks
        warp_ref_patches_x1 = (
            warp_ref_patches_x1.view(
                N,
                n_r_blocks_H,
                n_r_blocks_W,
                C_x1,
                Hr_x1 // n_r_blocks_H,
                Wr_x1 // n_r_blocks_W,
            )
            .permute(
                0, 3, 1, 4, 2, 5
            )  # [N, C_x1, n_r_blocks_H, Hr_x1 // n_r_blocks_H, n_r_blocks_W, Wr_x1 // n_r_blocks_W,]
            .contiguous()
        )  # [N, C_x1, n_r_blocks_H, Hr_x1, n_r_blocks_W, Wr_x1]
        warp_ref_patches_x1 = warp_ref_patches_x1.view(N, C_x1, Hr_x1, Wr_x1)
        warp_ref_patches_x2 = (
            warp_ref_patches_x2.view(
                N,
                n_r_blocks_H,
                n_r_blocks_W,
                C_x2,
                Hr_x1 // n_r_blocks_H * 2,
                Wr_x1 // n_r_blocks_W * 2,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
        )  # [N, C_x2, n_r_blocks_H, Hr_x2, n_r_blocks_W, Wr_x2]
        warp_ref_patches_x2 = warp_ref_patches_x2.view(N, C_x2, Hr_x2, Wr_x2)
        warp_ref_patches_x4 = (
            warp_ref_patches_x4.view(
                N,
                n_r_blocks_H,
                n_r_blocks_W,
                C_x4,
                Hr_x1 // n_r_blocks_H * 4,
                Wr_x1 // n_r_blocks_W * 4,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
        )  # [N, C_x4, n_r_blocks_H, Hr_x4, n_r_blocks_W, Wr_x4]
        warp_ref_patches_x4 = warp_ref_patches_x4.view(N, C_x4, Hr_x4, Wr_x4)

        warp_ref_l = [warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]
        return warp_ref_l
