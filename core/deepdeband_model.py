"""deepDeband U-Net (pix2pix) — ICIP 2022 / RaymondLZhou.

Architecture adapted from ComfyUI_Bit-Depth-Enhancer (MIT) which mirrors
pytorch-CycleGAN-and-pix2pix UnetGenerator used by deepDeband.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(
            input_nc,
            inner_nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False if norm_layer is not nn.Identity else True,
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False if norm_layer is not nn.Identity else True,
            )
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False if norm_layer is not nn.Identity else True,
            )
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, submodule=None, innermost=True, norm_layer=norm_layer
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout
            )
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, x):
        return self.model(x)


class DeepDebandModel(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        self.generator = UnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=8,
            ngf=ngf,
            norm_layer=nn.BatchNorm2d,
            use_dropout=False,
        )

    def forward(self, x):
        return self.generator(x)

    def load_pretrained(self, weights_path: str, strict: bool = False):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "generator" in checkpoint:
                state = checkpoint["generator"]
            else:
                state = checkpoint
        else:
            state = checkpoint

        # pix2pix checkpoints are often raw generator state_dict
        cleaned = {}
        for key, value in state.items():
            name = key
            if name.startswith("module."):
                name = name[7:]
            if name.startswith("generator."):
                name = name[len("generator.") :]
            cleaned[name] = value

        try:
            self.generator.load_state_dict(cleaned, strict=strict)
        except RuntimeError:
            self.generator.load_state_dict(state, strict=False)
        return self
