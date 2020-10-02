# module.basic.py
# 
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.common import *


class Conv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(Conv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x


class Conv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(Conv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.Conv3d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x


class DeConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, out_pad=0, bn=True, relu=True):
        super(DeConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.ConvTranspose3d(ch_in, ch_out, kernel_size,
                                             stride, padding=pad, dilation=dilation, output_padding=out_pad)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class HorizontalCircularConv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(HorizontalCircularConv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.circ_pad = [pad, pad, 0, 0]
        self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size,
                                    stride, padding=[pad,0],
                                    dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = F.pad(x, self.circ_pad, mode='circular')
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class HorizontalCircularConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(HorizontalCircularConv3D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.circ_pad = [pad, pad, 0, 0, 0, 0]
        self.conv = torch.nn.Conv3d(ch_in, ch_out, kernel_size,
                                    stride, padding=[pad,pad,0],
                                    dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm3d(ch_out)

    def forward(self, x, residual=None):
        x = F.pad(x, self.circ_pad, mode='circular')
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if not residual is None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x

class UpsampleConv3D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(UpsampleConv3D, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear',
                                          align_corners=True)
        self.conv = Conv3D(ch_in, ch_out, kernel_size, stride, pad, dilation,
                           bn, relu)

    def forward(self, x, residual=None):
        x = self.upsample(x)
        x = self.conv(x, residual)
        return x