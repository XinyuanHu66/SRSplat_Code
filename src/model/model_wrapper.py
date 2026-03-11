from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import numpy as np
from PIL import Image
from jaxtyping import UInt8

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from kornia.filters import spatial_gradient
import pdb

from ..geometry.projection import get_fov, homogenize_points, project
from .point_decoder.decoder import PointDecoder
from .types import Gaussians


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        pointdecoder_cfg,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            
        # Our Point Decoder module
        self.k_num = 30_000
        self.grid_size = 0.01
        self.point_decoder = PointDecoder(
                            stride=[2],
                            dec_depths=[2],
                            dec_channels=[256],
                            dec_num_head=[16],
                            dec_patch_size=[32],
                            mlp_ratio=4,
                            qkv_bias=True,
                            # global pooling
                            enable_ada_lnnorm=False,
                            # upscale block
                            upscale_factor=[2],
                            n_frequencies=15,
                            enable_absolute_pe=False,
                            enable_upscale_drop_path=True,
                            # mask block
                            non_leaf_ratio=[1.0],
        )


        # Fine details cross-attention module
        self.in_dim = 160
        cond_dim = 8
        self.sh_degree = 4
        self.sh_dim = (self.sh_degree + 1)**2*3  # sh_degree is 4
        self.mlp1 = nn.Sequential(
            nn.Linear(163, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, self.in_dim))
        self.norm = nn.LayerNorm(self.in_dim)
        self.cross_att = nn.MultiheadAttention(
            embed_dim=self.in_dim, num_heads=16, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)

        # # TODO: not using this after concat module
        self.mlp2 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, 128)) # + self.sh_dim))

        self.mlp3 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, 128)) # + self.sh_dim))

        self.init(self.mlp1)
        self.init(self.mlp2)
        self.init(self.mlp3)

        
    def init(self, layers):
        # MLP initialization as in mipnerf360
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue 
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
                
                
    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):
        # Not using grad_mask here
        # points = points[mask]
        
        # pdb.set_trace()
        points = points.view(-1,3).unsqueeze(0).repeat(n_views_sel, 1, 1)
        n_points = points.shape[1]

        h,w = img_ref.shape[-2:]
        src_ixts = batch["context"]["intrinsics"][idx].reshape(-1,3,3)
        src_w2cs = batch["context"]["extrinsics"][idx].reshape(-1,4,4)
        src_ixts = src_ixts.unsqueeze(dim=1)
        src_w2cs = src_w2cs.unsqueeze(dim=1)

        img_wh = torch.tensor([w,h], device=self.device)
        point_xy, __, point_z = project(points, src_w2cs, src_ixts)
        point_xy = point_xy * 2 - 1.0

        imgs_coarse = torch.cat((renderings.color[idx],renderings.alpha[idx].unsqueeze(dim=1),renderings.depth[idx].unsqueeze(dim=1)), dim=1)
        # pdb.set_trace()
        imgs_coarse = torch.cat((img_ref, imgs_coarse),dim=1)
        ### padding_mode='border'
        feats_coarse = nn.functional.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False, padding_mode='border').view(n_views_sel,-1,n_points).to(imgs_coarse)

        z_diff = (feats_coarse[:,-1:] - point_z.view(n_views_sel,-1, n_points)).abs()

        point_feats = torch.cat((feats_coarse[:,:-1],z_diff), dim=1)

        return point_feats.detach()



    def score_based_mask_calculation(self, batch, score_maps, topk=30000):
        b, v, c, h, w = score_maps.shape
        assert c == 3, "score_maps 应该是 RGB 重复后的形状"
        
        # 计算灰度图
        score_map_gray = score_maps.mean(dim=2)  # [B, V, H, W]
        
        # reshape 到 [B, V*H*W]
        flat_scores = score_map_gray.view(b, -1)  # [B, V*H*W]
        
        # 创建空掩码
        selected_pts_mask = torch.zeros_like(flat_scores, dtype=torch.bool)  # [B, V*H*W]
        
        for i in range(b):
            # 获取每个样本中分数前 topk 的索引
            topk_vals, topk_inds = torch.topk(flat_scores[i], k=topk, dim=-1)
            selected_pts_mask[i].scatter_(0, topk_inds, True)
        
        total_selected = selected_pts_mask.sum().item()
        return selected_pts_mask, total_selected


    def rendering_fine(self, batch, gaussians, selected_gradient_mask, k_num, output_context):
        def save_tensor_as_png(tensor, output_path):
            # 将图像张量转换为 NumPy 数组
            image_array = tensor.detach().cpu().numpy()

            # 将像素值重新缩放到 0-255 范围内，确保为整数类型
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

            # 从 NumPy 数组创建图像
            image = Image.fromarray(image_array.transpose(1, 2, 0))  # 调整维度顺序

            # 保存图像到指定路径，并指定文件扩展名
            image.save(output_path + ".png")
        b, v, _, h, w = batch["context"]["image"].shape
        gaussians.means = rearrange(gaussians.means, "b (v h w) xyz -> b v (h w) xyz", v=v, h=h, w=w)
        gaussians.covariances = rearrange(gaussians.covariances, "b (v h w) i j -> b v (h w) i j", v=v, h=h, w=w)
        gaussians.harmonics = rearrange(gaussians.harmonics, "b (v h w) m n -> b v (h w) m n", v=v, h=h, w=w)
        gaussians.opacities = rearrange(gaussians.opacities, "b (v h w) -> b v (h w)", v=v, h=h, w=w)
        gaussians.features = rearrange(gaussians.features, "b (v h w) f -> b v (h w) f", v=v, h=h, w=w)
        selected_gradient_mask = rearrange(selected_gradient_mask, "b (v h w) -> b v (h w)", v=v, h=h, w=w)

        all_gaussians_fine_means = []
        all_gaussians_fine_covariances = []
        all_gaussians_fine_harmonics = []
        all_gaussians_fine_opacities = []

        all_before_means = []
        all_before_covariances = []
        all_before_harmonics = []
        all_before_opacities = []

        for i_b in range(b):
            # We extract fine details features from cross-attention
            point_feats = self.get_point_feats(i_b, batch["context"]["image"][i_b], output_context, \
                v, batch, gaussians.means[i_b], torch.flatten(selected_gradient_mask[i_b]))
            point_feats =  torch.einsum('lcb->blc', point_feats)
            coarse_features = gaussians.features[i_b].view(-1, 163).contiguous()
            coarse_features = self.mlp1(coarse_features)
            coarse_features = self.norm(coarse_features.unsqueeze(1))
            # pdb.set_trace()
            feats_fine = self.cross_att(coarse_features, point_feats, point_feats, need_weights=False)[0]
            _feats_out = self.mlp2(feats_fine)
            _feats_fine = _feats_out

            # pdb.set_trace()

            _features_fine = _feats_fine[torch.flatten(selected_gradient_mask[i_b][:v])].squeeze()
            centers_offset = 0

            for i_v in range(v):
                _centers = gaussians.means[i_b,i_v,:,:][selected_gradient_mask[i_b,i_v,:]]
                _features = gaussians.features[i_b,i_v,:,:][selected_gradient_mask[i_b,i_v,:]]
                _features = self.mlp1(_features)

                _features = self.mlp3(_features)

                _features = torch.concat((_features, _features_fine[centers_offset:(centers_offset+_centers.shape[0])].squeeze()), dim=1)
                centers_offset = _centers.shape[0]
                _offset = (torch.arange(1, 1 + 1) * _centers.shape[0]).long().to(self.device, non_blocking=True)
                data_dict = {
                    "grid_size": self.grid_size,
                    "coord": _centers,
                    "feat": _features,
                    "offset": _offset,
                }
                extrinsics = batch["context"]["extrinsics"][i_b,i_v,:,:]
                intrinsics = batch["context"]["intrinsics"][i_b,i_v,:,:]
                gaussians_dense = self.point_decoder(  
                    data_dict, 
                    batch,
                    extrinsics,
                    intrinsics,
                    (h,w))
                update_harmonics = gaussians.harmonics[i_b,i_v,:,:,:][~selected_gradient_mask[i_b,i_v,:]]
                if i_v == 0:
                    gaussians_fine_means = torch.cat([gaussians_dense.means.squeeze(), \
                                    gaussians.means[i_b,i_v,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_covariances = torch.cat([gaussians_dense.covariances.squeeze(), \
                                    gaussians.covariances[i_b,i_v,:,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_harmonics = torch.cat([gaussians_dense.harmonics.squeeze(), update_harmonics], dim=0)
                    gaussians_fine_opacities = torch.cat([gaussians_dense.opacities.squeeze(), \
                                    gaussians.opacities[i_b,i_v,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)




                else:
                    gaussians_fine_means = torch.cat([gaussians_fine_means, gaussians_dense.means.squeeze(), \
                                    gaussians.means[i_b,i_v,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_covariances = torch.cat([gaussians_fine_covariances, gaussians_dense.covariances.squeeze(), \
                                    gaussians.covariances[i_b,i_v,:,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_harmonics = torch.cat([gaussians_fine_harmonics, gaussians_dense.harmonics.squeeze(), \
                                    update_harmonics], dim=0)
                    gaussians_fine_opacities = torch.cat([gaussians_fine_opacities, gaussians_dense.opacities.squeeze(), \
                                    gaussians.opacities[i_b,i_v,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)

            

            
            all_gaussians_fine_means.append(gaussians_fine_means)
            all_gaussians_fine_covariances.append(gaussians_fine_covariances)
            all_gaussians_fine_harmonics.append(gaussians_fine_harmonics)
            all_gaussians_fine_opacities.append(gaussians_fine_opacities)


        # pdb.set_trace()
        all_gaussians_fine_means = torch.stack(all_gaussians_fine_means)
        all_gaussians_fine_covariances = torch.stack(all_gaussians_fine_covariances)
        all_gaussians_fine_harmonics = torch.stack(all_gaussians_fine_harmonics)
        all_gaussians_fine_opacities = torch.stack(all_gaussians_fine_opacities)

        # pdb.set_trace()

        gaussians_fine = Gaussians(
                all_gaussians_fine_means,
                all_gaussians_fine_covariances,
                all_gaussians_fine_harmonics,
                all_gaussians_fine_opacities,
        )
        output_fine = self.decoder.forward(
            gaussians_fine,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        output_fine = output_fine.color
        
        return output_fine, gaussians_fine
    def training_step(self, batch, batch_idx):
        def save_tensor_as_png(tensor, output_path):
            # 将图像张量转换为 NumPy 数组
            image_array = tensor.detach().cpu().numpy()

            # 将像素值重新缩放到 0-255 范围内，确保为整数类型
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

            # 从 NumPy 数组创建图像
            image = Image.fromarray(image_array.transpose(1, 2, 0))  # 调整维度顺序

            # 保存图像到指定路径，并指定文件扩展名
            image.save(output_path + ".png")
        def convert_images(
            images: torch.Tensor | list[UInt8[Tensor, "..."]],
            scale_factor: float = 0.5,  # 下采样比例
        ) -> Float[Tensor, "batch views channels height width"]:
            """
            对输入图像张量进行下采样，支持五维或四维输入。
            假设输入是 PyTorch 张量或字节流形式的图像。
            """
            if isinstance(images, list):  # 处理字节流形式的图像
                torch_images = []
                for image in images:
                    # 从字节流加载图像
                    image = Image.open(BytesIO(image.numpy().tobytes()))
                    # 转换为 PyTorch 张量
                    image_tensor = self.to_tensor(image)
                    torch_images.append(image_tensor)
                images = torch.stack(torch_images)  # 转换为形状为 [batch, views, channels, height, width] 或 [batch, channels, height, width]

            if images.ndim == 4:  # 如果是四维张量 (batch, channels, height, width)
                # 添加视角维度，假设 views = 1
                images = images.unsqueeze(1)  # 转换为五维 (batch, views=1, channels, height, width)

            if images.ndim != 5:  # 确保是五维输入
                raise ValueError(f"Expected 4D or 5D tensor, but got {images.ndim}D tensor with shape {images.shape}")

            # 获取形状信息
            batch, views, channels, h, w = images.shape

            # 计算新高度和宽度
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)

            # 将视角维度和批量维度合并，进行下采样
            images_reshaped = images.view(batch * views, channels, h, w)  # 转换为 (batch * views, channels, height, width)
            images_downsampled = torch.nn.functional.interpolate(
                images_reshaped,
                size=(new_h, new_w),  # 新尺寸
                mode="bilinear",
                align_corners=False,
            )

            # 将张量重新调整为原始五维形状
            images_downsampled = images_downsampled.view(batch, views, channels, new_h, new_w)

            return images_downsampled
        
        
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        


        # Run the model.
        gaussians = self.encoder(
            0,batch["context"], self.global_step, False, scene_names=batch["scene"]
        )

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        
        output_context = self.decoder.forward(
            gaussians,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        selected_gradient_mask, k_num = self.score_based_mask_calculation(batch,batch["context"]["score_maps"])
        # print(k_num)
        # pdb.set_trace()
        fine_color, gs_fine = self.rendering_fine(batch, gaussians, selected_gradient_mask, k_num, output_context)
        
        target_gt = batch["target"]["image_HR"]
        
        output.LR = convert_images(output.color, scale_factor=0.25)
        
        save_tensor_as_png(batch["target"]["image_LR"][0][0],'outputs/input_LR')
        save_tensor_as_png(output.color[0][0],'outputs/predict')
        save_tensor_as_png(fine_color[0][0],'outputs/predict_fine')
        save_tensor_as_png(batch["target"]["image_HR"][0][0],'outputs/gt_HR')
        save_tensor_as_png(output.LR[0][0],'outputs/predict_LR')
        
        # pdb.set_trace()

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss1 = loss_fn.forward(output.color, batch, gaussians, self.global_step, batch["target"]["image_HR"])
            loss_fine = loss_fn.forward(fine_color, batch, gaussians, self.global_step,batch["target"]["image_HR"])
            
            self.log(f"loss/{loss_fn.name}", loss1)
            total_loss = total_loss + loss1 + loss_fine
        

        loss4 = (batch["context"]["score_maps"].mean(dim=2, keepdim=True) - batch["context"]["Texture"].mean(dim=2, keepdim=True)).abs().mean()
        total_loss = total_loss + loss4 * 0.05

        # pdb.set_trace()
        
        
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}; "
                f"num = {gaussians.means.shape[2]*gaussians.means.shape[1]}; "
                f"num_fine = {gs_fine.means.shape[1]}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        def save_tensor_as_png(tensor, output_path):
            # 将图像张量转换为 NumPy 数组
            image_array = tensor.detach().cpu().numpy()

            # 将像素值重新缩放到 0-255 范围内，确保为整数类型
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)

            # 从 NumPy 数组创建图像
            image = Image.fromarray(image_array.transpose(1, 2, 0))  # 调整维度顺序

            # 保存图像到指定路径，并指定文件扩展名
            image.save(output_path + ".png")
        
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                1,
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        
        # rgb_gt = batch["target"]["image"][0]
        rgb_gt = batch["target"]["image_HR"][0]
        
        output_context = self.decoder.forward(
            gaussians,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        # selected_gradient_mask, k_num = self.score_based_mask_calculation(batch,batch["context"]["score_maps"],batch["context"]["tao_high"] )
        selected_gradient_mask, k_num = self.score_based_mask_calculation(batch,batch["context"]["score_maps"])
        fine_color, gaussians_fine = self.rendering_fine(batch, gaussians, selected_gradient_mask, k_num, output_context)
        images_prob_fine = fine_color[0]
        images_context = output_context.color[0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")

            for index, color in zip(batch["target"]["index"][0], images_context):
                save_image(color, path / scene / f"render_context/{index:0>6}_context.png")

            for index, cont in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                save_image(cont, path / scene / f"context_input/{index:0>6}_context.png")

            for index, cont in zip(batch["context"]["index"][0], batch["context"]["image_HR"][0]):
                save_image(cont, path / scene / f"context_gt/{index:0>6}_context.png")
                
            for index, HR_gt in zip(batch["target"]["index"][0], rgb_gt):
                save_image(HR_gt, path / scene / f"HR_gt/{index:0>6}.png")
                
            for index, color in zip(batch["target"]["index"][0], images_prob_fine):
                save_image(color, path / scene / f"color/{index:0>6}_fine.png")
            
            for index, LR_input in zip(batch["target"]["index"][0], batch["target"]["image_LR"][0]):
                save_image(LR_input, path / scene / f"LR_input/{index:0>6}.png")

            for index, texture in zip(batch["context"]["index"][0], batch["context"]["score_maps"][0]):
                save_image(texture, path / scene / f"score_maps/{index:0>6}.png")
            
            for index, texture in zip(batch["target"]["index"][0], batch["target"]["Texture"][0]):
                save_image(texture, path / scene / f"target_texture/{index:0>6}.png")

            for index, Sobel in zip(batch["context"]["index"][0], batch["context"]["Texture"][0]):
                save_image(Sobel, path / scene / f"Sobel/{index:0>6}.png")
            
            # import pdb
            # pdb.set_trace()

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            
            coarse = compute_psnr(rgb_gt, images_prob).mean().item()
            fine = compute_psnr(rgb_gt, images_prob_fine).mean().item()

            if f"psnr_coarse" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr_coarse"] = []
            if f"psnr_fine" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr_fine"] = []

            self.test_step_outputs[f"psnr_coarse"].append(coarse)
            self.test_step_outputs[f"psnr_fine"].append(fine)
                       
            if coarse > fine:
                rgb = images_prob
            else:
                rgb = images_prob_fine

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            0,
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]
        
        # Picking gradient_based point first, then do rendering on fine image
        target_gt = batch["target"]["image"]

        # output_context = self.decoder.forward(
        #     gaussians_softmax,
        #     batch["context"]["extrinsics"],
        #     batch["context"]["intrinsics"],
        #     batch["context"]["near"],
        #     batch["context"]["far"],
        #     (h, w),
        #     depth_mode=self.train_cfg.depth_mode,
        # )

        # selected_gradient_mask, k_num = self.score_based_mask_calculation(batch,batch["context"]["score_maps"],0.3 )
        # output_softmax_fine, __ = self.rendering_fine(batch, gaussians_softmax, selected_gradient_mask, k_num, output_context)
        # rgb_softmax_fine = output_softmax_fine.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)


        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
        comparison_fine = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            # add_label(vcat(*rgb_softmax_fine), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison_fine",
            [prep_image(add_border(comparison_fine))],
            step=self.global_step,
            caption=batch["scene"],
        )

 

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
