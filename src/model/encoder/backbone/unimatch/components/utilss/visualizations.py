import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from itertools import pairwise


def color_by_colormap(
    value: float,
    min_value: float = 0.0,
    max_value: float = 1.0,
    map_name: str = "plasma",
) -> tuple[int, int, int]:
    if map_name == "smootCoolWarm":
        value = max(min_value, min(value, max_value))
        map = [(0.0, (59, 76, 192)), (0.5, (221, 221, 221)), (1.0, (180, 4, 38))]
    elif map_name == "plasma":
        value = max(min_value, min(value, max_value))
        map = [
            (0.0, (13, 8, 135)),
            (0.14, (84, 2, 163)),
            (0.29, (139, 10, 165)),
            (0.43, (185, 50, 137)),
            (0.57, (219, 92, 104)),
            (0.71, (244, 136, 73)),
            (0.86, (254, 188, 43)),
            (1.0, (240, 249, 33)),
        ]
    elif map_name == "distinct":
        colors = [
            (192, 192, 192),
            (47, 79, 79),
            (127, 0, 0),
            (0, 100, 0),
            (128, 128, 0),
            (72, 61, 139),
            (0, 0, 128),
            (154, 205, 50),
            (139, 0, 139),
            (102, 205, 170),
            (255, 0, 0),
            (255, 140, 0),
            (255, 215, 0),
            (124, 252, 0),
            (138, 43, 226),
            (0, 0, 0),
            (0, 191, 255),
            (0, 0, 255),
            (255, 255, 255),
            (30, 144, 255),
            (219, 112, 147),
            (240, 230, 140),
            (255, 20, 147),
            (255, 160, 122),
            (238, 130, 238),
        ]
        index = int(value) % len(colors)
        c = colors[index]
        return c[0], c[1], c[2]
    else:
        return (0, 0, 0)
    frac = (value - min_value) / (max_value - min_value)
    for b1, b2 in pairwise(map):
        if frac >= b1[0] and frac <= b2[0]:
            lower_bucket = b1
            upper_bucket = b2
            break
    frac_b = (frac - lower_bucket[0]) / (upper_bucket[0] - lower_bucket[0])
    r = frac_b * upper_bucket[1][0] + (1 - frac_b) * lower_bucket[1][0]
    g = frac_b * upper_bucket[1][1] + (1 - frac_b) * lower_bucket[1][1]
    b = frac_b * upper_bucket[1][2] + (1 - frac_b) * lower_bucket[1][2]

    return int(r), int(g), int(b)


class Visualization:
    src_points = None
    dst_points = None
    src_scale = 1.0
    dst_scale = 1.0
    src_offset = []
    dst_offset = []

    """Visualization base class"""

    def draw(self, image, index_in_batch: int) -> None:
        """Code to draw on the image"""


class LineVisualization(Visualization):
    def __init__(
        self,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
        value: torch.Tensor = None,
        color_map: str = "plasma",
        color_map_min_val: float = 0.0,
        color_map_max_val: float = 1.0,
        scale: float = 4.0,
    ) -> None:
        self.src_points = src_points.detach().tolist()
        self.dst_points = dst_points.detach().tolist()
        self.value = value.detach().tolist()
        self.color_map = color_map
        self.color_map_min_val = color_map_min_val
        self.color_map_max_val = color_map_max_val
        self.src_scale = scale
        self.dst_scale = scale
        self.src_offset = [0, 0]
        self.dst_offset = [0, 0]

    def draw(self, image, index_in_batch: int) -> None:
        draw = ImageDraw.Draw(image)
        idx = 0
        for src_point, dst_point, val in zip(
            self.src_points[index_in_batch],
            self.dst_points[index_in_batch],
            self.value[index_in_batch],
        ):
            src_h = self.src_scale * src_point[0] + self.src_offset[0]
            src_w = self.src_scale * src_point[1] + self.src_offset[1]
            dst_h = self.dst_scale * dst_point[0] + self.dst_offset[0]
            dst_w = self.dst_scale * dst_point[1] + self.dst_offset[1]

            r, g, b = color_by_colormap(
                val, self.color_map_min_val, self.color_map_max_val, "plasma"
            )
            color = (r, g, b, 255)
            draw.line([(src_w, src_h), (dst_w, dst_h)], fill=color, width=0)
            radius = 2
            r, g, b = color_by_colormap(idx, map_name="distinct")
            color = (r, g, b, 255)
            draw.ellipse(
                (src_w - radius, src_h - radius, src_w + radius, src_h + radius), color
            )
            draw.ellipse(
                (dst_w - radius, dst_h - radius, dst_w + radius, dst_h + radius), color
            )
            idx = idx + 1


def pad_to_size(
    x: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.tensor, list[int]]:
    _, _, h, w = x.shape
    pad_H_t = (size[0] - h) // 2
    pad_H_d = (size[0] - h) // 2 + (size[0] - h) % 2
    pad_H_l = (size[1] - w) // 2
    pad_H_r = (size[1] - w) // 2 + (size[1] - w) % 2
    padding = [pad_H_t, pad_H_d, pad_H_l, pad_H_r]
    x_pad = F.pad(x, pad=(pad_H_l, pad_H_r, pad_H_t, pad_H_d), mode="constant")
    return x_pad, padding


def batch_to_images(
    batch: dict[str, torch.Tensor],
    outputs: list[torch.Tensor],
    order: list[str] = ["IN", "REF", "OUT"],
    visualization: Visualization = None,
    max_batch_images: int = 5,
) -> Image:
    N, _, _, _ = batch["IN"].shape
    H_max = 0
    W_max = 0
    for name in order:
        if name != "OUT":
            _, _, h, w = batch[name].shape
        else:
            _, _, h, w = outputs[0].shape
        if h > H_max:
            H_max = h
        if w > W_max:
            W_max = w

    images = []
    paddings = []
    for name in order:
        if name == "OUT":
            for output in outputs:
                out_pad, padding = pad_to_size(output, (H_max, W_max))
                paddings.append(padding)
                images.append(out_pad)
        else:
            image_pad, padding = pad_to_size(batch[name], (H_max, W_max))
            images.append(image_pad)
            paddings.append(padding)

    out_pil_images = []
    for j in range(min(N, max_batch_images)):
        batch_image = [image[j] for image in images]
        batch_image = torch.stack(batch_image)
        B = batch_image[:, 0:1, :, :]
        G = batch_image[:, 1:2, :, :]
        R = batch_image[:, 2:3, :, :]
        batch_image = torch.cat([R, G, B], dim=1)
        to_im = ToPILImage()
        between_image_padding = 2
        grid_imgs = make_grid(batch_image, padding=between_image_padding)
        ndarr = (
            grid_imgs.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        image = to_im(ndarr)
        if visualization:
            if j == 0:
                # To fit all images together on one line, a padding is added to each of them.
                # This padding has then to be added to the src and dst points too.
                im_0_pad_t, _, im_0_pad_l, _ = paddings[
                    0
                ]  # paddings of the first image in the grid. Where the src points are.
                visualization.src_offset[0] += between_image_padding + im_0_pad_t
                visualization.src_offset[1] += between_image_padding + im_0_pad_l

                im_1_pad_t, _, im_1_pad_l, _ = paddings[
                    1
                ]  # paddings of the second image in the grid. Where the dst points are.
                im_0_width = images[0].shape[3]
                visualization.dst_offset[0] += between_image_padding + im_1_pad_t
                visualization.dst_offset[1] += (
                    2 * between_image_padding + im_0_width + im_1_pad_l
                )

            visualization.draw(image, j)

        out_pil_images.append(image)
    return out_pil_images
