"""
Temporal Shift Module (TSM) and Gated Shift Module (GSM) for video understanding.

TSM adapted from: https://github.com/mit-han-lab/temporal-shift-module
MIT License, Copyright (c) 2021 MIT HAN Lab

GSM adapted from: https://github.com/swathikirans/GSM
BSD 2-Clause License, Copyright (c) 2019, FBK
"""

import math
import torch
import torch.nn as nn
import torchvision
import timm
from torch.cuda import FloatTensor as ftens


# --- TSM ---

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment, n_div, inplace=True):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using TSM, in-place shift...')
        print('=> Using TSM, fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div,
                       inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fold):
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


# --- GSM ---

class _GSM(nn.Module):
    def __init__(self, fPlane, num_segments=3):
        super(_GSM, self).__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()

    def lshift_zeroPad(self, x):
        return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)

    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)

    def forward(self, x):
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        assert shape[0] == self.fPlane
        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        x_bn = self.bn(x)
        x_bn_relu = self.relu(x_bn)
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)
        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]
        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        return y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)


# --- Shift insertion into backbone networks ---

class GatedShift(nn.Module):
    def __init__(self, net, n_segment, n_div):
        super(GatedShift, self).__init__()

        if isinstance(net, torchvision.models.resnet.BasicBlock):
            channels = net.conv1.in_channels
        elif isinstance(net, torchvision.ops.misc.ConvNormActivation):
            channels = net[0].in_channels
        elif isinstance(net, timm.layers.conv_bn_act.ConvBnAct):
            channels = net.conv.in_channels
        elif isinstance(net, nn.Conv2d):
            channels = net.in_channels
        else:
            raise NotImplementedError(type(net))

        self.fold_dim = math.ceil(channels // n_div / 4) * 4
        self.gsm = _GSM(self.fold_dim, n_segment)
        self.net = net
        self.n_segment = n_segment
        print('=> Using GSM, fold dim: {} / {}'.format(
            self.fold_dim, channels))

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:, :self.fold_dim, :, :] = self.gsm(x[:, :self.fold_dim, :, :])
        y[:, self.fold_dim:, :, :] = x[:, self.fold_dim:, :, :]
        return self.net(y)


def make_temporal_shift(net, clip_len, is_gsm=False):

    def _build_shift(net):
        if is_gsm:
            return GatedShift(net, n_segment=clip_len, n_div=4)
        else:
            return TemporalShift(net, n_segment=clip_len, n_div=8)

    if isinstance(net, torchvision.models.ResNet):
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = _build_shift(b.conv1)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1)
        net.layer2 = make_block_temporal(net.layer2)
        net.layer3 = make_block_temporal(net.layer3)
        net.layer4 = make_block_temporal(net.layer4)

    elif isinstance(net, timm.models.regnet.RegNet):
        n_round = 1

        def make_block_temporal(stage):
            blocks = list(stage.children())
            print('=> Processing stage with {} blocks residual'.format(
                len(blocks)))
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = _build_shift(b.conv1)

        make_block_temporal(net.s1)
        make_block_temporal(net.s2)
        make_block_temporal(net.s3)
        make_block_temporal(net.s4)

    elif isinstance(net, timm.models.convnext.ConvNeXt):
        n_round = 1

        def make_block_temporal(stage):
            blocks = list(stage.blocks)
            print('=> Processing stage with {} blocks residual'.format(
                len(blocks)))

            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv_dw = _build_shift(b.conv_dw)
            return nn.Sequential(*blocks)

        make_block_temporal(net.stages[0])
        make_block_temporal(net.stages[1])
        make_block_temporal(net.stages[2])
        make_block_temporal(net.stages[3])

    else:
        raise NotImplementedError('Unsupported architecture')
