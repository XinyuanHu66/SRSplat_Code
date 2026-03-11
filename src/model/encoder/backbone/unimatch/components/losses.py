import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from models.components.vgg_model import VGGFeatureExtractor
from models.components.perceptive_network import PerceptiveNet
from torchmetrics.functional.image import image_gradients


class Loss(nn.Module):
    """Loss Base class."""

    def __init__(
        self,
        weight: float = 1.0,
        enabled: bool = True,
        start_epoch: int = 0,
        end_epoch: int = -1,
        start_iter: int = 0,
        end_iter: int = -1,
    ) -> None:
        super(Loss, self).__init__()
        self.weight = weight
        self.enabled = enabled
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_iter = start_iter
        self.end_iter = end_iter

    def tb_name(self) -> list[str]:
        "Name used on tensorboard"

    def active(self, current_epoch: int, current_iter: int) -> bool:
        """Active if current is between start (including) and end (including) or end is -1."""
        if not self.enabled:
            return False
        if current_epoch < self.start_epoch:
            return False
        if current_epoch > self.end_epoch and self.end_epoch >= 0:
            return False
        if current_iter < self.start_iter:
            return False
        if current_iter > self.end_iter and self.end_iter >= 0:
            return False
        return True


class L1Loss(Loss):
    """L1 loss."""

    def __init__(self, **kwargs) -> None:
        super(L1Loss, self).__init__(**kwargs)
        self.loss = nn.L1Loss()

    def tb_name(self) -> list[str]:
        return ["l1_loss"]

    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, C, H, W] Input tensor.
        :param gt: [N, C, H, W] Ground-truth tensor.
        :return: [] The loss
        """
        return self.loss(x, gt)


class PerceptualLossMASA(Loss):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
            Default: False.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights: dict[str, float] = {"conv5_4": 1.0},
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        perceptual_weight: float = 1.0,
        style_weight: float = 0.0,
        norm_img: bool = False,
        criterion: str = "l1",
        **kwargs,
    ) -> None:
        super(PerceptualLossMASA, self).__init__(**kwargs)
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def tb_name(self) -> list[str]:
        return ["perceptual_loss"]

    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, C, H, W] Input tensor.
        :param gt: [N, C, H, W] Ground-truth tensor.
        :return: [] The loss
        """

        if self.norm_img:
            x = (x + 1.0) * 0.5
            gt = (gt + 1.0) * 0.5

        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="fro",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss

    def _gram_mat(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class PerceptualLossMARINER(Loss):
    def __init__(self, **kwargs) -> None:
        super(PerceptualLossMARINER, self).__init__(**kwargs)
        self.net = PerceptiveNet()

    def tb_name(self) -> list[str]:
        return ["perceptual_loss"]

    def forward(self, x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, C, H, W] Input tensor.
        :param gt: [N, C, H, W] Ground-truth tensor.
        :return: [] The loss
        """
        x_out = self.net(x)
        gt_out = self.net(gt)

        perceptive_loss = torch.tensor(0.0).float().to(x.device)
        for layer in gt_out:
            perceptive_loss += torch.mean((x_out[layer] - gt_out[layer]) ** 2)
        perceptive_loss /= len(gt_out)
        return perceptive_loss


class GANLoss(nn.Module):
    """Define GAN loss.
    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(
        self,
        gan_type: str,
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        loss_weight: float = 1.0,
    ) -> None:
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "wgan_softplus":
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """wgan loss.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input: torch.Tensor, target: bool) -> torch.Tensor:
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(
        self, input: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor | bool:
        """Get target label.
        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ["wgan", "wgan_softplus"]:
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(
        self, input: torch.Tensor, target_is_real: bool, is_disc: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == "hinge":
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class VGGStyleDiscriminator160(nn.Module):
    """VGG style discriminator with input size 160 x 160.
    It is used to train SRGAN and ESRGAN.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64):
        super(VGGStyleDiscriminator160, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 5 * 5, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(2) == 160 and x.size(3) == 160, (
            f"Input spatial size must be 160x160, " f"but received {x.size()}."
        )

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(
            self.bn0_1(self.conv0_1(feat))
        )  # output spatial size: (80, 80)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(
            self.bn1_1(self.conv1_1(feat))
        )  # output spatial size: (40, 40)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(
            self.bn2_1(self.conv2_1(feat))
        )  # output spatial size: (20, 20)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(
            self.bn3_1(self.conv3_1(feat))
        )  # output spatial size: (10, 10)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (5, 5)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


class AdversarialLoss(Loss):
    """
    Relative GAN
    """

    def __init__(
        self, gan_type: str = "RGAN", gan_k: int = 2, lr_dis: float = 1e-4, **kwargs
    ) -> None:
        """Careful when applying this loss in more than one model iteration. Works only if the parameter start_iter=<last_iteration> is given.
        Otherwise throws error that some tensor was changed inplace. Can only be used in one iteration.
        """

        super(AdversarialLoss, self).__init__(**kwargs)
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.discriminator = VGGStyleDiscriminator160(num_in_ch=3, num_feat=64)

        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), betas=(0, 0.9), eps=1e-8, lr=lr_dis
        )

        self.criterion_adv = GANLoss(gan_type="vanilla")

    def tb_name(self) -> list[str]:
        return ["adv_loss", "d_loss"]

    def set_requires_grad(
        self, nets: nn.Module | list[nn.Module], requires_grad: bool = False
    ) -> None:
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # D Loss
        for _ in range(self.gan_k):
            self.set_requires_grad(self.discriminator, True)
            self.optimizer.zero_grad()
            # real
            d_fake = self.discriminator(fake).detach()
            d_real = self.discriminator(real)
            d_real_loss = (
                self.criterion_adv(d_real - torch.mean(d_fake), True, is_disc=True)
                * 0.5
            )
            d_real_loss.backward()
            # fake
            d_fake = self.discriminator(fake.detach())
            d_fake_loss = (
                self.criterion_adv(
                    d_fake - torch.mean(d_real.detach()), False, is_disc=True
                )
                * 0.5
            )
            d_fake_loss.backward()
            loss_d = d_real_loss + d_fake_loss

            self.optimizer.step()

        # G Loss
        self.set_requires_grad(self.discriminator, False)
        d_real = self.discriminator(real).detach()
        d_fake = self.discriminator(fake)
        g_real_loss = (
            self.criterion_adv(d_real - torch.mean(d_fake), False, is_disc=False) * 0.5
        )
        g_fake_loss = (
            self.criterion_adv(d_fake - torch.mean(d_real), True, is_disc=False) * 0.5
        )
        loss_g = g_real_loss + g_fake_loss

        # Generator loss
        return loss_g, loss_d
