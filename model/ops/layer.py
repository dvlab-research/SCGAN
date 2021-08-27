import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm
import numpy as np
import math
from math import sqrt
from functools import partial
from functools import reduce
from util.func import _pair

from model.ops.norm import SPADE
from model.ops.norm import SpatialCondNorm
from model.sync_batchnorm import SynchronizedBatchNorm2d


class SpatialCondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=False, num_experts=3, spectral=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.conv_bases = nn.ModuleList()
        for _ in range(num_experts):
            base = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias) \
                if spectral is False else \
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                        groups, bias))
            # nn.init.normal_(base.weight, 0, 0.02)
            nn.init.xavier_normal_(base.weight, 0.02)
            self.conv_bases.append(base)

    # x: b x c x h x w
    # routing weights: b x experts x h x w
    # todo: solve it.
    def forward(self, x, routing_weights):
        # B, C, H, W = x.shape
        # b x out x h x w
        rk = routing_weights.size(1)
        if rk == self.num_experts:
            routing_weights = routing_weights.unsqueeze(2)
            out = [0] * self.num_experts
            for i in range(self.num_experts):
                out[i] = self.conv_bases[i](x)
                out[i] = out[i].unsqueeze(1)
                # out[i] = self.dropblock(out[i]).unsqueeze(1) if self.dropblock else out[i].unsqueeze(1)
                # if self.dropblock:
                #     print('loop ', out[i].shape)
                #
                #     print('loop2 ', out[i].shape)

            out = torch.cat(out, dim=1)
            # print('in local, ', out.shape, routing_weights.shape)
            out = out * routing_weights
            out = torch.sum(out, dim=1)
        else:
            # print('rk ', str(rk), ' n ', str(self.num_experts))
            b, _, h, w = routing_weights.shape
            out = [0] * self.num_experts
            for i in range(self.num_experts):
                out[i] = self.conv_bases[i](x)
            out = torch.cat(out, dim=1)
            # print('in local, ', out.shape, routing_weights.shape)
            out = out * routing_weights
            out = out.view(b, -1, self.num_experts, h, w)
            out = torch.sum(out, dim=2)
        return out

class Blur(nn.Module):
    def __init__(self):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        self.register_buffer('weight', weight)

    def forward(self, input):
        return F.conv2d(
            input,
            self.weight.repeat(input.shape[1], 1, 1, 1),
            padding=1,
            groups=input.shape[1],
        )

class IterativeGaussian(nn.Module):
    def __init__(self, win_size=7, channel=1, iters=3, sigma=1.5, relative=True):
        super(IterativeGaussian, self).__init__()
        self.win_size = win_size
        self.iters = iters
        self.channel = channel
        self.relative = relative
        self.sigma = sigma
        self.window = self.__create_win__(self.win_size, self.channel)
        self.eps = 1e-8

    def __gaussian__(self, win_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - win_size // 2) ** 2 / (2.0 * sigma ** 2)) for x in range(win_size)])
        return gauss

    def __create_win__(self, win_size, channel):
        _1D_win = self.__gaussian__(win_size, self.sigma).unsqueeze(1)
        _2D_win = _1D_win.mm(_1D_win.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_win.expand(channel, 1, win_size, win_size).contiguous()
        return window

    def forward(self, x):
        channel = x.size(1)

        if not (channel == self.channel and self.window.data.type() == x.data.type()):
            self.window = self.__create_win__(self.win_size, channel).type_as(x)
            self.channel = channel

        mask = x
        x1 = None
        for _ in range(self.iters):
            x1 = x
            x = F.conv2d(x, self.window, padding=self.win_size//2, groups=channel) * mask

        x = x / (x1+self.eps) if self.relative else x

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

'''
ResnetBlock used in SCGAN
'''
class SCResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        num_experts_conv = opt.num_experts_conv
        num_experts_norm = opt.num_experts_norm
        spectral = True if 'spectral' in opt.norm_G else False
        # create conv layers
        self.conv_0 = SpatialCondConv2d(in_channels=fin, out_channels=fmiddle, kernel_size=3,
                                        num_experts=num_experts_conv, spectral=spectral)
        self.conv_1 = SpatialCondConv2d(in_channels=fmiddle, out_channels=fout, kernel_size=3,
                                        num_experts=num_experts_conv,
                                        spectral=spectral)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(in_channels=fin, out_channels=fout,
                                                  kernel_size=3, padding=1, stride=1))
            scc_config_str = opt.norm_G.replace('spectral', '')
            t = scc_config_str.replace('scc', 'spade') if 'scc' in scc_config_str else scc_config_str
            t = 'spade' + t if 'spade' not in t else t
            self.norm_s = SPADE(t, fin, opt.semantic_nc)

        # define normalization layers
        scc_config_str = opt.norm_G.replace('spectral', '')
        self.norm_config = scc_config_str
        if 'scc' in scc_config_str or 'scs' in scc_config_str:
            scc_config_str = 'scc' + scc_config_str[3:]
            self.norm_0 = SpatialCondNorm(scc_config_str, fin, opt.semantic_nc, nClass=num_experts_norm)
            self.norm_1 = SpatialCondNorm(scc_config_str, fmiddle, opt.semantic_nc, nClass=num_experts_norm)
        elif 'spade' in scc_config_str:
            self.norm_0 = SPADE(scc_config_str, fin, opt.semantic_nc)
            self.norm_1 = SPADE(scc_config_str, fmiddle, opt.semantic_nc)
        else:
            if 'instance' in scc_config_str:
                self.norm_0 = nn.InstanceNorm2d(fin, affine=False)
                self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=False)
            elif 'batch' in scc_config_str:
                self.norm_0 = nn.BatchNorm2d(fin, affine=True)
                self.norm_1 = nn.BatchNorm2d(fmiddle, affine=True)
            elif 'sync_batch' in scc_config_str:
                self.norm_0 = SynchronizedBatchNorm2d(fin, affine=True)
                self.norm_1 = SynchronizedBatchNorm2d(fmiddle, affine=True)
            else:
                raise ValueError('normalization layer %s is not recognized' % scc_config_str)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, routing_weights, seg=None):
        if 'scc' not in self.norm_config and 'spade' not in self.norm_config:
            dx = self.conv_0(self.actvn(self.norm_0(x)), routing_weights[0])
            dx = self.conv_1(self.actvn(self.norm_1(dx)), routing_weights[0])
        elif 'scc' in self.norm_config or 'scs' in self.norm_config: # using scc norm
            dx = self.conv_0(self.actvn(self.norm_0(x, routing_weights[1])), routing_weights[0])
            dx = self.conv_1(self.actvn(self.norm_1(dx, routing_weights[1])), routing_weights[0])
        else:
            dx = self.conv_0(self.actvn(self.norm_0(x, seg)), routing_weights[0])
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg)), routing_weights[0])

        x_s = self.shortcut(x, seg=seg)

        out = x_s + dx

        return out

    def shortcut(self, x, seg=None, routing_weights=None):
        if self.learned_shortcut:
            if routing_weights is not None and seg is not None:
                x_s = self.conv_s(self.norm_s(x, routing_weights[1], seg))
            elif routing_weights is not None:
                x_s = self.conv_s(self.norm_s(x, routing_weights[1]))
            else:
                x_s = self.conv_s(self.norm_s(x)) if seg is None else self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
