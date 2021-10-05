import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

import cargan


def NormedConv1d(*args, **kwargs):
    norm = kwargs.pop("norm", "weight_norm")
    if norm == "weight_norm":
        return weight_norm(nn.Conv1d(*args, **kwargs))
    elif norm == "spectral_norm":
        return spectral_norm(nn.Conv1d(*args, **kwargs))


def NormedConv2d(*args, **kwargs):
    norm = kwargs.pop("norm", "weight_norm")
    if norm == "weight_norm":
        return weight_norm(nn.Conv2d(*args, **kwargs))
    elif norm == "spectral_norm":
        return spectral_norm(nn.Conv2d(*args, **kwargs))


class DiscriminatorP(nn.Module):
    
    def __init__(self, period, norm="weight_norm"):
        super().__init__()
        self.layers = nn.ModuleList([
            NormedConv2d(cargan.NUM_DISCRIM_FEATURES, 32, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0), norm=norm),
            NormedConv2d(1024, 1024, (5, 1), 1, padding=(2, 0), norm=norm)])
        self.output = NormedConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))
        self.period = period

    def forward(self, x):
        # 1d to 2d
        x = F.pad(x, (0, self.period - x.shape[-1] % self.period), "reflect")
        x = x.view(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)
        fmaps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmaps.append(x)
        fmaps.append(self.output(x))
        return fmaps


class DiscriminatorS(torch.nn.Module):

    def __init__(self, norm="weight_norm"):
        super().__init__()

        self.layers = nn.ModuleList([
            NormedConv1d(cargan.NUM_DISCRIM_FEATURES, 128, 15, 1, padding=7, norm=norm),
            NormedConv1d(128, 128, 41, 2, groups=4, padding=20, norm=norm),
            NormedConv1d(128, 256, 41, 2, groups=16, padding=20, norm=norm),
            NormedConv1d(256, 512, 41, 4, groups=16, padding=20, norm=norm),
            NormedConv1d(512, 1024, 41, 4, groups=16, padding=20, norm=norm),
            NormedConv1d(1024, 1024, 41, 1, groups=16, padding=20, norm=norm),
            NormedConv1d(1024, 1024, 5, 1, padding=2, norm=norm)])
        self.output = NormedConv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            fmaps.append(x)
        fmaps.append(self.output(x))
        return fmaps


class Discriminator(nn.Module):

    def __init__(self, num_multi_pool=5, num_multi_scale=3):
        super().__init__()
        
        prime_ratios = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        self.multi_pooled_disc = nn.ModuleList([
            DiscriminatorP(prime_ratios[i])
            for i in range(num_multi_pool)])

        self.multi_scale_disc = nn.ModuleList( [
            DiscriminatorS(norm="spectral_norm" if i == 0 else "weight_norm")
            for i in range(num_multi_scale)])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        results = []
        
        for disc in self.multi_pooled_disc:
            results.append(disc(x))

        for disc in self.multi_scale_disc:
            results.append(disc(x))
            x = self.downsample(x)
            
        return results
