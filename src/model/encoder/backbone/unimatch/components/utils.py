import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import functools


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        # 如果传入的是 functools.partial，需要调用它来实例化
        if isinstance(block, functools.partial):
            block = block()  # 实例化 block
        layers.append(block)
    return nn.Sequential(*layers)


def MASA_weight_init(module: nn.Module, scale: float = 0.1) -> None:
    for name, m in module.named_modules():
        classname = m.__class__.__name__
        if classname == "DCN":
            continue
        elif classname == "Conv2d" or classname == "ConvTranspose2d":
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, 0.5 * math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("BatchNorm") != -1:
            if m.weight is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        elif classname.find("Linear") != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            if m.bias:
                m.bias.data = torch.ones(m.bias.data.size())

    for name, m in module.named_modules():
        classname = m.__class__.__name__
        if classname == "ResidualBlock":
            m.conv1.weight.data *= scale
            m.conv2.weight.data *= scale
        if classname == "SAM":
            # initialization
            m.conv_gamma.weight.data.zero_()
            m.conv_beta.weight.data.zero_()


class ResidualBlock(nn.Module):
    def __init__(
        self,
        nf: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        act: str = "relu",
    ) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            nf,
            nf,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            nf,
            nf,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


def bis(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Batch index select: Select the patches specified by index tensor from the input tensor at dimension dim.

    :param input: [N, C, H*W]: A tensor of H*W patches of size C
    :param dim: The dimension where the indexes will be taken from. Here the H*W because C specifies the patch content.
    :param index: [N, Hi, Wi]: A tensor of [Hi, Wi] indexes of patches to be selected
    :return: A tensor [N, C, Hi*Wi] of selected input patches of size C for each index Hi*Wi
    """ 
    
    # input_size = input.size(dim)  # 获取目标维度的尺寸
    # index = torch.clamp(index, min=0, max=input_size - 1)  # 限制索引范围
    views = [input.size(0)] + [
        1 if i != dim else -1 for i in range(1, len(input.size()))
    ]  # views = [N, 1, -1]
    expanse = list(input.size())
    expanse[0] = -1
    expanse[dim] = -1  # expanse = [-1, C, -1]
    index = (
        index.clone().view(views).expand(expanse)
    )  # [N, Hi, Wi] -> [N, 1, Hi*Wi] - > [N, C, Hi*Wi]
    return torch.gather(input, dim, index)  # [N, C, Hi*Wi]


def find_best_k_ref_patch_centers_within_blocks(
    src_blocks: torch.Tensor,
    ref_blocks: torch.Tensor,
    k: int = 1,
    patch_size: int = 3,
    patch_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Searches in each src_block the top k ref patches in the corresponding ref_block. Requires the blocks to be padded by 1.

    :param src_blocks: [B, C, H_src_block, W_src_block]: A tensor of B blocks for which best matching patches in the corresponding ref_block are searched.
    :param ref: [B, C, H_ref_block, W_ref_block]:  A tensor of B reference blocks in which the patches are searched.
    :param k: The top k patches with the highest correlation that are returned
    :param patch_size: Size of the patches
    :param patch_stride: The stride in which the patches in the reference block are taken
    :return: A tuple containing (in order):
        - A tensor [B, H_src_block - 2, W_src_block - 2, k] of sorted correlations of the k best reference patches.
        - A tensor [B, H_src_block - 2, W_src_block - 2, k] of sorted patch center indexes of the k reference patches with highest correlation.
    """
    B, _, H_src_block, W_src_block = src_blocks.size()
    _, _, H_ref_block, W_ref_block = ref_blocks.size()
    block_padding = patch_size // 2

    # For every pixel in each ref block, a patch of C * patch_size * patch_size is created
    reflr_unfold = F.unfold(
        ref_blocks, kernel_size=(patch_size, patch_size), padding=0, stride=patch_stride
    )  # [B, C*patch_size*patch_size, (H_ref_block-2)*(W_ref_block-2)]

    # For every pixel in each lr block, a patch of C * patch_size * patch_size is created
    lr_unfold = F.unfold(
        src_blocks, kernel_size=(patch_size, patch_size), padding=0, stride=patch_stride
    )  # [B, C*patch_size*patch_size, (H_src_block-2)* (W_src_block-2)]
    lr_unfold = lr_unfold.permute(
        0, 2, 1
    )  # [B, (H_src_block-2)* (W_src_block-2), C*patch_size*patch_size]

    # Normalizes the values of every patch: Divides by the norm of the max value in each patch
    lr_unfold = F.normalize(lr_unfold, dim=2)
    reflr_unfold = F.normalize(reflr_unfold, dim=1)

    # Batch Matrix Matrix multiplication. At each entry (i, j) is the dot product similarity between src block patch i and ref block patch j
    corr = torch.bmm(
        lr_unfold, reflr_unfold
    )  # [B, (H_src_block-2)*(W_src_block-2), (H_ref_block-2)*(W_ref_block-2)]
    corr = corr.view(
        B,
        H_src_block - 2 * block_padding,
        W_src_block - 2 * block_padding,
        (H_ref_block - 2 * block_padding) * (W_ref_block - 2 * block_padding),
    )  # [B, (H_src_block-2), (W_src_block-2), (H_ref_block-2)*(W_ref_block-2)]

    # Finds the top k similar ref patches for every lr patch
    sorted_corr, ind_l = torch.topk(
        corr, k, dim=-1, largest=True, sorted=True
    )  # [B, H_src_block-2), (W_src_block-2), k]

    return sorted_corr, ind_l


def find_best_matching_block(src_blocks, dst_blocks):
    # 256 , 768 -> 256 blocks each dim 1x1
    dst_b_T = dst_blocks.permute(0, 2, 1)

    src_norm = F.normalize(src_blocks, dim=2)
    dst_norm = F.normalize(dst_b_T, dim=1)

    corr = torch.bmm(src_norm, dst_norm)
    sorted_corr, ind_l = torch.topk(corr, 1, dim=-1, largest=True, sorted=True)
    return sorted_corr, ind_l


def find_best_k_ref_block_centers(
    src_blocks: torch.Tensor,
    ref: torch.Tensor,
    k: int,
    patch_size: int = 3,
    patch_stride: int = 1,
    patch_dilations: list[int] = [1, 2, 3],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Searches the top k ref patches for the dilated center patch of each block in src_blocks.

    :param src_blocks: [N, B, C, H_b, W_b]: A tensor of B blocks for which best matching patches are searched.
    :param ref: [N, C, H, W]:  A tensor of the reference in which the patches are searched.
    :param k: The top k patch matches that are returned
    :param patch_size: Size of the patches
    :param patch_stride: The stride in which the patches in the reference are taken
    :param patch_dilations: The dilations in which the patches are compared. Applied both to the src and ref patches
    :return: A tuple containing (in order):
        - A tensor of sorted correlations of the k reference patches with the highest dot product similarity to the center patch of each src_blocks block.
        - A tensor of sorted patch center indexes of the k reference patches with highest dot product similarity.
    """
    N, C, _, _ = ref.size()
    _, _, _, H_b, W_b = src_blocks.size()
    block_padding = patch_size // 2

    # (x, y) = Block center
    x, y = H_b // 2, W_b // 2

    # Sums the dot product patch correlation over different dilations of the center patch in each block
    corr_sum = 0
    for dilation in patch_dilations:
        # For every pixel in ref, a patch of C * px * py = C * patch_size * patch_size is created
        ref_patches = F.unfold(
            ref,
            kernel_size=(patch_size, patch_size),
            padding=block_padding * dilation,
            stride=patch_stride,
            dilation=dilation,
        )  # [N, C*px*py, H*W]

        # The center patch is cropped from every block. Depending on the dilation a different stride is used in the indexing but the result is always patch_size * patch_size
        dilated_patch_size = patch_size * dilation - (dilation - 1)
        src_blocks_patches = src_blocks[
            :,
            :,
            :,
            y - dilated_patch_size // 2 : y + dilated_patch_size // 2 + 1 : dilation,
            x - dilated_patch_size // 2 : x + dilated_patch_size // 2 + 1 : dilation,
        ]  # [N, B, C, px, py]

        # Each patch is contiguous in memory
        src_blocks_patches = src_blocks_patches.contiguous().view(
            N, -1, C * patch_size * patch_size
        )  # [N, B, C*px*py]

        # Normalizes the values of every patch: Divides by the norm of the max value in each patch
        src_blocks_patches = F.normalize(src_blocks_patches, dim=2)
        ref_patches = F.normalize(ref_patches, dim=1)

        # Batch Matrix Matrix multiplication. [N, B, C*px*py], [N, C*px*py, H*W] = [N, B, H*W] At each entry (i, j) is the dot product similarity between src_blocks center patch i and ref patch j
        corr = torch.bmm(src_blocks_patches, ref_patches)  # [N, B, H*W]
        corr_sum = corr_sum + corr

    # Finds the correlation (= dot product similarity score) of the most similar k ref lr patchs and their index in reflr_patches
    sorted_corr, ind_l = torch.topk(
        corr_sum, k, dim=-1, largest=True, sorted=True
    )  # ([N, B, k], [N, B, k])

    return sorted_corr, ind_l


def transfer(
    ref_feature_blocks: torch.Tensor,
    src_block_best_patch_index: torch.Tensor,
    soft_att: torch.Tensor,
    patch_size: int = 3,
    patch_fold_padding: int = 1,
    scale: int = 1,
) -> torch.Tensor:
    """Finds the scaled patches in ref_feature_blocks that are specified by src_block_best_patch_index.
    Folds then those patches together to get for every src_block a block (H_src_block, W_src_block) with warped features from ref_feature_blocks.
    The elements from the block are then multiplied by soft_att to reduce weight to elements with low correlation to any feature.

    :param ref_feature_blocks: [N*B, C, H_ref_block, W_ref_block]: A tensor of B ref feature blocks from which patches are created and transfered based on the src_block_best_patch_index.
    :param src_block_best_patch_index: [N*B, H_src_block, W_src_block]: A tensor of indexes where each index points to the ref_feature_blocks patch with highest correlation for every block element.
    :param soft_att: [N*B, 1, H_src_block, W_src_block]: A tensor of correlations (Dot product similarity) between the src patches and ref patches at index.
    :param patch_size: Size of the patches
    :param patch_fold_padding: The padding used when folding the indexed ref patches
    :param scale: The desired scaling of the output blocks.
    :return: A tensor [N*B, C, scale*H_src_block, scale*W_src_block] where every src_block contains a blend of warped ref_feature_blocks patches
    """

    # Patches of the best matching ref block
    fea_unfold = F.unfold(
        ref_feature_blocks,
        kernel_size=(patch_size, patch_size),
        padding=0,
        stride=scale,
    )  # [N*B, C*patch_size*patch_size, (H_ref_block-2)*(W_ref_block-2)]

    # Finds for every pixel in every block the patch in fea specified by index
    out_unfold = bis(
        fea_unfold, 2, src_block_best_patch_index
    )  # [N*B, C*patch_size*patch_size, H_src_block*W_src_block]
    divisor = torch.ones_like(
        out_unfold
    )  # [N*B, C*patch_size*patch_size, H_src_block*W_src_block]

    # Folds and sums the patches back together. See divisor as to how many patches where summed for each pixel.
    _, H_src_block, W_src_block = src_block_best_patch_index.size()
    out_fold = F.fold(
        out_unfold,
        output_size=(H_src_block * scale, W_src_block * scale),
        kernel_size=(patch_size, patch_size),
        padding=patch_fold_padding,
        stride=scale,
    )  # [N*B, C, scale*H_src_block, scale*W_src_block]

    # How many patches were added on every pixel location. Divide by that to get everywhere the pixel average over all patches that were summed.
    # Padding and stride are selected like this to ensure at most 9 patches impact the final pixel value. Otherwise on larger scale with smaller stride, more patches would impact.
    divisor = F.fold(
        divisor,
        output_size=(H_src_block * scale, W_src_block * scale),
        kernel_size=(patch_size, patch_size),
        padding=patch_fold_padding,
        stride=scale,
    )  # [N*B, C, scale*H_src_block, scale*W_src_block]
    soft_att_resize = F.interpolate(
        soft_att, size=(H_src_block * scale, W_src_block * scale), mode="bilinear"
    )  # [N*B, 1, scale * H_src_block, scale * W_src_block]

    # Gives more weight to pixels which had a strong corresponding ref patch. For bad correspondences, the value goes to 0.
    out_fold = out_fold / divisor * soft_att_resize
    return out_fold


def make_block(
    idx_w: torch.Tensor,
    idx_h: torch.Tensor,
    block_width: int,
    block_height: int,
    scale,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Creates an index block with top left corner (idx_h, idx_w) and size (block_height, block_width)

    :param idx_w: [N, B]: A tensor of N batches and B blocks with the top left corner width index
    :param idx_h: [N, B]: A tensor of N batches and B blocks with the top left corner height index
    :param block_width:  The width of the block
    :param block_height: The height of the block
    :param scale: The scale with which idx_w, idx_h, block_width, block_height are scaled
    :return: A tuple containing (in order):
        - A tensor [N * B * block_height * scale * block_height * scale] of the height indexes of the block elements
        - A tensor [N * B * block_width * scale * block_width * scale] of the width indexes of the block elements
    """
    idx_w = idx_w * scale  # [N, B]
    idx_h = idx_h * scale  # [N, B]

    # Replicates the start index scaled blocksize times
    idx_w = idx_w.view(-1, 1).repeat(
        1, block_width * scale
    )  # [N*B, block_width * scale]
    idx_h = idx_h.view(-1, 1).repeat(
        1, block_height * scale
    )  # [N*B, block_height * scale]

    # Adds a tensor [0, 1, ..., (block_size * scale) -1] to idx to create the coordinates of the block entries
    idx_w = idx_w + torch.arange(
        0, block_width * scale, dtype=torch.long, device=idx_w.device
    ).view(1, -1)
    idx_h = idx_h + torch.arange(
        0, block_height * scale, dtype=torch.long, device=idx_h.device
    ).view(1, -1)

    # Creates for every block a height index and width index list of the block entrys
    # Uses a for loop because torch.meshgrid needs a 1D tensor as input
    ind_h_l = []
    ind_w_l = []
    for i in range(idx_w.size(0)):  # N*B
        grid_h, grid_w = torch.meshgrid(
            idx_h[i], idx_w[i], indexing="ij"
        )  # [block_width * scale, block_width * scale]
        ind_h_l.append(
            grid_h.contiguous().view(-1)
        )  # [block_width * scale * block_width * scale]
        ind_w_l.append(
            grid_w.contiguous().view(-1)
        )  # [block_height * scale * block_height * scale]
    ind_h = torch.cat(ind_h_l)
    ind_w = torch.cat(ind_w_l)

    return ind_h, ind_w
