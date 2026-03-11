import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import numpy as np

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset



import torch.nn.functional as F
from kornia.filters import spatial_gradient
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import pdb
import cv2


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]


    def __iter__(self):
        # 下采样比例
        scale_factor = 0.25
        # 目标分辨率
        # target_h, target_w = int(self.cfg.image_shape[0] * scale_factor), int(self.cfg.image_shape[1] * scale_factor)

        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # print(chunk_path)
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.test_times_per_scene
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])
                    # print(context_indices)
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images_HR = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images_HR = self.convert_images_HR(context_images_HR)
                target_images_HR = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images_HR = self.convert_images_HR(target_images_HR)
                
                # 加载图像并下采样
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images, scale_factor=scale_factor)  # 添加下采样比例
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images, scale_factor=scale_factor)  # 添加下采样比例
                
                target_images_LR = target_images
                context_images_LR = context_images
                

                self.save_tensor_as_png(target_images_HR[0],'outputs/target_HR')
                self.save_tensor_as_png(target_images[0],'outputs/target')
                
                context_images_shangcaiyang = self.upsample_images(context_images, scale_factor=(1/scale_factor))
                target_images_shangcaiyang = self.upsample_images(target_images, scale_factor=(1/scale_factor))
                self.save_tensor_as_png(target_images[0],'outputs/target-shangcaiyang')
                
                target_images = target_images_shangcaiyang
                context_images = context_images_shangcaiyang
                
                # pdb.set_trace()
                # 调试日志
                # print(f"Original context image shape: {[image.size for image in example['images']]}")
                # print(f"Downsampled context image shape: {context_images.shape}")
                # print(f"Downsampled target image shape: {target_images.shape}")

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                # context_image_invalid = context_images.shape[1:] != (3, 360*scale_factor, 640*scale_factor)
                # target_image_invalid = target_images.shape[1:] != (3, 360*scale_factor, 640*scale_factor)
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                

                edgemap_con = (spatial_gradient(context_images_HR, order=1).abs().sum(1).mean(1) + spatial_gradient(context_images_HR, order=2).abs().sum(1).mean(1)).unsqueeze(1).repeat(1,3,1,1)

                edgemap_tar = (spatial_gradient(target_images_HR, order=1).abs().sum(1).mean(1) + spatial_gradient(target_images_HR, order=2).abs().sum(1).mean(1)).unsqueeze(1).repeat(1,3,1,1)

                # edgemap_con = self.compute_canny_edgemap(context_images_HR)
                # edgemap_tar = self.compute_canny_edgemap(target_images_HR)


                
                
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "image_HR": context_images_HR,
                        "image_LR": context_images_LR,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                        "Texture": edgemap_con,
                        # "image_shangcaiyang": context_images_shangcaiyang,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "image_HR":target_images_HR,
                        "image_LR":target_images_LR,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                        "Texture": edgemap_tar,
                        # "image_shangcaiyang": target_images_shangcaiyang,
                    },
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))
    
    
    
    def save_tensor_as_png(self,tensor, output_path):
        # 将图像张量转换为 NumPy 数组
        image_array = tensor.detach().cpu().numpy()

        # 将像素值重新缩放到 0-255 范围内，确保为整数类型
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

        # 从 NumPy 数组创建图像
        image = Image.fromarray(image_array.transpose(1, 2, 0))  # 调整维度顺序

        # 保存图像到指定路径，并指定文件扩展名
        image.save(output_path + ".png")

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images_HR(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
        scale_factor: float = 0.5,  # 新增下采样比例
    ) -> Float[Tensor, "batch 3 height width"]:
        """
        将图像从列表转换为张量，并执行下采样操作.
        """
        torch_images = []
        for image in images:
            # 从字节流加载图像
            image = Image.open(BytesIO(image.numpy().tobytes()))
            # 转换为 PyTorch 张量
            image_tensor = self.to_tensor(image)
            # 下采样操作
            h, w = image_tensor.shape[1:]  # 获取原始高度和宽度
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)  # 计算新尺寸
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),  # 添加 batch 维度
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # 移除 batch 维度
            torch_images.append(image_tensor)
        return torch.stack(torch_images)
    
    def upsample_images(
        self,
        images: torch.Tensor,
        scale_factor: float = 2.0  # 默认上采样比例
    ) -> torch.Tensor:
        """
        对输入的图像张量进行上采样操作，保留原始数据类型。
        """
        # 确保输入是浮点类型以便进行插值
        images_dtype = images.dtype
        if images_dtype != torch.float32 and images_dtype != torch.float64:
            images = images.to(torch.float32) / 255.0  # 假设 uint8 转换为浮点型 0-1 范围

        # 上采样操作
        upsampled_images = F.interpolate(
            images,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False
        )

        # 如果原始类型是 uint8，重新转换为 uint8
        if images_dtype == torch.uint8:
            upsampled_images = (upsampled_images * 255).to(torch.uint8)

        return upsampled_images

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )
