import torch
import torch.nn as nn
import torchvision.models as models


class PerceptiveNet(nn.Module):
    def __init__(
        self, layer_names: list[str] = ["relu1_1", "relu2_2", "relu3_3"]
    ) -> None:
        super(PerceptiveNet, self).__init__()

        vgg16_layers = list(
            models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.children()
        )
        vgg16_layer_names = [
            "conv1_1",
            "relu1_1",
            "conv1_2",
            "relu1_2",
            "pool1",
            "conv2_1",
            "relu2_1",
            "conv2_2",
            "relu2_2",
            "pool2",
            "conv3_1",
            "relu3_1",
            "conv3_2",
            "relu3_2",
            "conv3_3",
            "relu3_3",
            "pool3",
            "conv4_1",
            "relu4_1",
            "conv4_2",
            "relu4_2",
            "conv4_3",
            "relu4_3",
            "pool4",
            "conv5_1",
            "relu5_1",
            "conv5_2",
            "relu5_2",
            "conv5_3",
            "relu5_3",
            "pool5",
        ]

        self.blocks = []
        last_layer_idx = -1
        for layer_name in layer_names:
            current_layer_idx = vgg16_layer_names.index(layer_name)
            self.blocks.append(
                (
                    layer_name,
                    nn.Sequential(
                        *vgg16_layers[last_layer_idx + 1 : current_layer_idx + 1]
                    ),
                )
            )
            last_layer_idx = current_layer_idx
        self.block_seq = nn.Sequential(
            *map(lambda x: x[1], self.blocks)
        )  # Needed for state_dict.

        # Fix parameters.
        for param in self.block_seq.parameters():
            param.requires_grad = False

    def preprocess_input(self, batch: torch.Tensor) -> torch.Tensor:
        # Normalization constants.
        device = batch.device
        means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
        stds = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)

        # Normalize.
        batch = (batch - means.view(1, 3, 1, 1)) / stds.view(1, 3, 1, 1)

        return batch

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = self.preprocess_input(batch)

        # Forward pass.
        x = batch
        output = {}
        for layer_name, block in self.blocks:
            x = block(x)
            output[layer_name] = x

        return output
