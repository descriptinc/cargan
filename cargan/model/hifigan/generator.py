import functools

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

import cargan


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        conv_fn = functools.partial(Conv1d, channels, channels, kernel_size, 1)

        # Don't use weight norm for cumsum experiment
        if cargan.CUMSUM:
            weight_norm = lambda x: x
        else:
            weight_norm = torch.nn.utils.weight_norm

        self.convs1 = nn.ModuleList([
            weight_norm(conv_fn(
                dilation=dilation[0],
                padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(conv_fn(
                dilation=dilation[1],
                padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(conv_fn(
                dilation=dilation[2],
                padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(conv_fn(
                dilation=1,
                padding=get_padding(kernel_size, 1))),
            weight_norm(conv_fn(
                dilation=1,
                padding=get_padding(kernel_size, 1))),
            weight_norm(conv_fn(
                dilation=1,
                padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()

        # Don't use weight norm for cumsum experiment
        if cargan.CUMSUM:
            weight_norm = lambda x: x
        else:
            weight_norm = torch.nn.utils.weight_norm

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(
        self,
        resblock='1',
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4]):
        super().__init__()
        if cargan.CUMSUM:
            upsample_rates = [1] * len(upsample_rates)
            upsample_initial_channel = 128
            upsample_kernel_sizes = [1] * len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(
                cargan.NUM_FEATURES,
                upsample_initial_channel,
                7,
                1,
                padding=3))
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if cargan.AUTOREGRESSIVE:
            self.ar_model = cargan.model.condition.Autoregressive()

    def forward(self, x, ar=None):
        if cargan.AUTOREGRESSIVE:
            ar_feats = self.ar_model(ar)
            ar_feats = ar_feats.unsqueeze(2).repeat(1, 1, x.shape[2])
            x = torch.cat((x, ar_feats), dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
