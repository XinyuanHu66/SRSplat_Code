import io
from pathlib import Path
from typing import Union
import skvideo.io

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


# def save_video(
#     images: list[FloatImage],
#     path: Union[Path, str],
# ) -> None:
#     """Save an image. Assumed to be in range 0-1."""

#     # Create the parent directory if it doesn't already exist.
#     path = Path(path)
#     path.parent.mkdir(exist_ok=True, parents=True)

#     # Save the image.
#     # Image.fromarray(prep_image(image)).save(path)
#     frames = []
#     for image in images:
#         frames.append(prep_image(image))

#     writer = skvideo.io.FFmpegWriter(path, 
#                                      outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', 
#                                                  '-vf': f'setpts=1.*PTS'})
#     for frame in frames:
#         writer.writeFrame(frame)
#     writer.close()
from pathlib import Path
import skvideo.io
from moviepy.editor import ImageSequenceClip
from typing import Union, List
from ..misc.image_io import prep_image, FloatImage

def save_video(
    images: List[FloatImage],
    path: Union[Path, str],
    fps: int = 30,
) -> None:
    """Save a list of 0–1 float images to disk as a video file."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # 转成 uint8 H×W×3 numpy array 的列表
    frames = [prep_image(img) for img in images]

    if path.suffix.lower() == ".mp4":
        # MoviePy 写 MP4（指定 libx264 编码）
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(
            str(path),
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
        )
    else:
        # 其它后缀仍然用 skvideo
        writer = skvideo.io.FFmpegWriter(
            str(path),
            outputdict={
                "-pix_fmt": "yuv420p",
                "-crf": "21",
                "-vf": f"setpts=1.*PTS",
            },
        )
        for frame in frames:
            writer.writeFrame(frame)
        writer.close()