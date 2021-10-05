import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import cargan


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def get_padding(kernel_size, dilation=1):
    """Computes padding for non-strided dilated convolution"""
    return dilation * (kernel_size - 1) // 2


class GBlock(nn.Module):

    def __init__(self, input_dim, output_dim, upsample=1, kernel_size=3):
        super().__init__()

        ############################################################
        # Create first residual block consisting of conv1 and res1 #
        ############################################################
        
        self.conv1 = [nn.ReLU()]
        if upsample > 1:
            self.conv1 += [nn.Upsample(scale_factor=upsample)]
        self.conv1 += [
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                padding=get_padding(kernel_size)),
            nn.ReLU(),
            WNConv1d(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=3,
                padding=get_padding(kernel_size, 3))]

        self.res1 = [nn.Upsample(scale_factor=upsample)] if upsample > 1 else []
        self.res1 += [WNConv1d(input_dim, output_dim, kernel_size=1)]

        ####################################################
        # Create second residual block consisting of conv2 #
        ####################################################
        self.conv2 = [
            nn.ReLU(),
            WNConv1d(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=9,
                padding=get_padding(kernel_size, 9)),
            nn.ReLU(),
            WNConv1d(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=27,
                padding=get_padding(kernel_size, 27))]

        # Convert list of layers into nn.Sequential
        self.conv1 = nn.Sequential(*self.conv1)
        self.res1 = nn.Sequential(*self.res1)
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x) + self.res1(x)
        return x + self.conv2(x)


class Generator(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        input_size = cargan.NUM_FEATURES
        if cargan.CUMSUM:
            channels = 128
        self.gblocks = nn.Sequential(
            WNConv1d(input_size, channels, kernel_size=1),
            GBlock(channels, channels),
            GBlock(channels, channels),
            GBlock(channels, channels // 2, upsample=1 if cargan.CUMSUM else 4),
            GBlock(channels // 2, channels // 2, upsample=1 if cargan.CUMSUM else 4),
            GBlock(channels // 2, channels // 2, upsample=1 if cargan.CUMSUM else 4),
            GBlock(channels // 2, channels // 2),
            GBlock(channels // 2, channels // 4, upsample=1 if cargan.CUMSUM else 2),
            GBlock(channels // 4, channels // 4),
            GBlock(channels // 4, channels // 8, upsample=1 if cargan.CUMSUM else 2),
            GBlock(channels // 8, channels // 8))
        self.last_conv = nn.Sequential(
            nn.ReLU(),
            WNConv1d(channels // 8, 1, kernel_size=3, padding=1))
        if cargan.AUTOREGRESSIVE:
            self.ar_model = cargan.model.condition.Autoregressive()

    def forward(self, x, ar=None):
        if cargan.AUTOREGRESSIVE:
            ar_feats = self.ar_model(ar)
            ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, x.shape[2])
            x = torch.cat((x, ar_feats), dim=1)
        x = self.last_conv(self.gblocks(x))
        return torch.tanh(x)
