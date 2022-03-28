from typing import List

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision


class Vgg13(nn.Module):
    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt
        self.max_disp = opt.max_disp
        self.num_in_ch = opt.num_in_ch

        vgg13 = torchvision.models.vgg13(pretrained=False)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

        # Encoder D
        self.stem_block_depth = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.stem_block_depth.apply(weights_init)

        self.downsample_2_d = vgg13.features[4:9]
        self.downsample_4_d = vgg13.features[9:14]
        self.downsample_8_d = vgg13.features[14:19]
        self.downsample_16_d = vgg13.features[19:24]

        # Encoder RGB
        self.stem_block_rgb = nn.Sequential(
            nn.Conv2d(self.num_in_ch - 1, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.stem_block_rgb.apply(weights_init)

        self.downsample_2_rgb = vgg13.features[4:9]
        self.downsample_4_rgb = vgg13.features[9:14]
        self.downsample_8_rgb = vgg13.features[14:19]
        self.downsample_16_rgb = vgg13.features[19:24]

        # Decoder
        self.upsample_16 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_16.apply(weights_init)

        self.upsample_8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_8.apply(weights_init)

        self.upsample_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_4.apply(weights_init)

        self.upsample_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.upsample_2.apply(weights_init)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, 1, 1),
        )
        self.final_conv.apply(weights_init)

        self.out_ch = 32 + 64

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Extract features using two VGG backbone:
            one for Image, the other for Depth
        Params:
        img: a tensor with shape BxCxHxW
        Returns:
            a list of features extracted from images
        """
        depth = batch[:, self.num_in_ch - 1 :, :, :]
        depth = depth / self.max_disp
        stem_block_d = self.stem_block_depth(depth)
        downsample_2_d = self.downsample_2_d(stem_block_d)
        downsample_4_d = self.downsample_4_d(downsample_2_d)
        downsample_8_d = self.downsample_8_d(downsample_4_d)
        downsample_16_d = self.downsample_16_d(downsample_8_d)

        rgb = batch[:, : self.num_in_ch - 1, :, :]
        rgb = rgb / 255.0
        stem_block_rgb = self.stem_block_rgb(rgb)
        downsample_2_rgb = self.downsample_2_rgb(stem_block_rgb)
        downsample_4_rgb = self.downsample_4_rgb(downsample_2_rgb)
        downsample_8_rgb = self.downsample_8_rgb(downsample_4_rgb)
        downsample_16_rgb = self.downsample_16_rgb(downsample_8_rgb)

        upsample_16 = self.upsample_16(downsample_16_rgb + downsample_16_d)
        upsample_8 = self.upsample_8(upsample_16 + downsample_8_rgb + downsample_8_d)
        upsample_4 = self.upsample_4(upsample_8 + downsample_4_rgb + downsample_4_d)
        upsample_2 = self.upsample_2(upsample_4 + downsample_2_rgb + downsample_2_d)

        upsample_1 = self.final_conv(upsample_2)
        return [upsample_2, upsample_1]
