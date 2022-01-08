"""
Hourglass network baseline from ...
    Hourglass network inserted in the pre-activated Resnet
    Use lr=0.01 for current version
    (c) YANG, Wei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HourglassNet", "HourglassEncoder"]


class AddCoords(nn.Module):
    def __init__(self, with_r=False, with_boundary=False):
        super().__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, boundary_map):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim, device=input_tensor.device).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim, device=input_tensor.device).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor)],
            dim=1,
        )

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and (boundary_map is not None):
            # B, 64(H), 64(W)
            boundary_map = boundary_map.view(boundary_map.shape[0], 1, boundary_map.shape[1], boundary_map.shape[2])
            boundary_channel = torch.clamp(boundary_map, 0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)

            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=True, with_boundary=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r, with_boundary=with_boundary)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        if with_boundary:
            in_size += 2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x, boundary_map=None):
        ret = self.addcoords(x, boundary_map)
        ret = self.conv(ret)
        return ret


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassEncoder(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block=Bottleneck, num_stacks=2, input_channels=3, num_blocks=4, num_classes=16):
        super(HourglassEncoder, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.coordconv1 = CoordConv(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes * 2, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        coordconv, hg, res, fc, score, fc_, score_, heatmap = [], [], [], [], [], [], [], []
        for i in range(num_stacks):
            if i == 0:
                coordconv.append(CoordConv(ch, ch, kernel_size=3, padding=1))
            else:
                coordconv.append(CoordConv(ch, ch, kernel_size=3, padding=1, with_boundary=True))
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
                # heatmap

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        self.coordconv = nn.ModuleList(coordconv)
        self.heatmap = nn.ModuleList()

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
            conv,
            bn,
            self.relu,
        )

    def forward(self, x):
        out = []
        x = x.float()
        x = self.coordconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        boundary_pred = None
        for i in range(self.num_stacks):
            y = self.coordconv[i](x, boundary_pred)
            y = self.hg[i](y)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            boundary_pred = score[:, -1]
            out.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out


class HourglassNet(nn.Module):
    def __init__(self, encoder: HourglassEncoder, head, box_coder):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.box_coder = box_coder

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features[-1])

    @classmethod
    def from_config(cls, config):
        from hydra.utils import instantiate

        encoder = instantiate(config.encoder)
        # if config.num_channels != 3:
        #     encoder = encoder.change_input_channels(config.num_channels)

        head = instantiate(config.head, channels=config.encoder.num_classes)
        box_coder = instantiate(config.box_coder, output_stride=4)

        return cls(
            encoder=encoder,
            head=head,
            box_coder=box_coder,
        )
