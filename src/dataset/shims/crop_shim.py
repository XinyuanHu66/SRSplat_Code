import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews
import pdb


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    # print(f"Rescale target shape: {shape}")
    # print(f"Input image shape: {images.shape[-2:]}")
    # pdb.set_trace()
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], tuple([256,256]))
    images_HR, intrinsics_1 = rescale_and_crop(views["image_HR"], views["intrinsics"], tuple([256,256]))
    images_LR, intrinsics_2 = rescale_and_crop(views["image_LR"], views["intrinsics"], tuple([64,64]))
    texture, intrinsics_3 = rescale_and_crop(views["Texture"], views["intrinsics"], tuple([256,256]))
    # image_shangcaiyang, intrinsics_4 = rescale_and_crop(views["image_shangcaiyang"], views["intrinsics"], tuple([256,256]))
    
    bv,_, h,w = texture.shape
    for ss in range(bv):
        texture[ss] = (texture[ss] - texture[ss].min()) / (texture[ss].max() - texture[ss].min())
    # print(images_LR.shape)
    # print(texture[0].max(),texture[0].min())
    # print(texture[1].max(),texture[1].min())
    # print(texture.shape)
    # # assert 0
    # texture = (texture - texture.min()) / (texture.max() - texture.min())
    
    
    return {
        **views,
        "image_HR":images_HR,
        "image": images,
        "image_LR": images_LR,
        "intrinsics": intrinsics,
        "Texture":texture,
    }

# def apply_crop_shim_to_views_HR(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
#     images, intrinsics = rescale_and_crop(views["image_HR"], views["intrinsics"], shape)
#     return {
#         **views,
#         "image_HR": images,
#         "intrinsics": intrinsics,
#     }

def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
        
        # "target": apply_crop_shim_to_views_HR(example["target"], tuple([256,256])),
    }
