import torch.nn.functional
from omegaconf import DictConfig
from torch import nn, Tensor


class ParNetDownsampling(nn.Module):
    def __init__(self, input_channels, output_channels, groups=1):
        super().__init__()
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), groups=groups),
            nn.BatchNorm2d(output_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=groups,
            ),
            nn.BatchNorm2d(output_channels),
        )

        self.glob_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), groups=groups),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.avg_pool(x) + self.conv(x)
        y = y * self.glob_pool(x)
        return torch.nn.functional.silu(y)


class ParNetUpsampling(nn.Module):
    def __init__(self, input_channels, output_channels, groups=1):
        super().__init__()
        self.avg_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), groups=groups),
            nn.BatchNorm2d(output_channels),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=1,
                bias=False,
                groups=groups,
            ),
            nn.BatchNorm2d(output_channels),
        )

        self.glob_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), groups=groups),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.avg_pool(x) + self.conv(x)
        y = y * self.glob_pool(x)
        return torch.nn.functional.silu(y)


class ParNetFusion(nn.Module):
    def __init__(self, left_channels, right_channels, output_channels):
        super().__init__()
        self.left_norm = nn.BatchNorm2d(left_channels)
        self.right_norm = nn.BatchNorm2d(right_channels)
        self.shuffle = nn.Conv1d(left_channels + right_channels, output_channels, kernel_size=(1, 1))
        self.block = ParNetBlock(output_channels, output_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        z1 = torch.cat(
            [
                torch.nn.functional.interpolate(self.left_norm(x), size=y.size()[2:], align_corners=True, mode="bilinear"),
                self.right_norm(y),
            ],
            dim=1,
        )
        z2 = self.shuffle(z1)
        z3 = self.block(z2)
        return z3


class SSEBlock(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_channels)
        self.conv = nn.Conv1d(input_channels, input_channels, kernel_size=(1, 1))
        self._init_weights()

    def sse(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)  # Average pooling
        return self.conv(mean).sigmoid_()

    def forward(self, x: Tensor):
        x: Tensor = self.bn(x)
        return x * self.sse(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ParNetBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.sse = SSEBlock(input_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(input_channels),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv1(x) + self.conv3(x) + self.sse(x)
        return torch.nn.functional.silu(y, inplace=True)


__all__ = ["ParNet"]


class ParNet(nn.Module):
    def __init__(self, input_channels, encoder_blocks, decoder_blocks, head, box_coder):
        super().__init__()
        self.head = head
        self.box_coder = box_coder

        ch1, ch2, ch3, ch4 = encoder_blocks

        self.down1 = ParNetDownsampling(input_channels, ch1)
        self.down2 = ParNetDownsampling(ch1, ch2)
        self.down3 = ParNetDownsampling(ch2, ch3)
        self.down4 = ParNetDownsampling(ch3, ch4)

        self.fusion1 = ParNetFusion(ch4, ch3, decoder_blocks[1])
        self.fusion2 = ParNetFusion(decoder_blocks[1], ch2, decoder_blocks[0])

        self.stream1 = nn.Sequential(
            ParNetBlock(ch2, ch2),
            ParNetBlock(ch2, ch2),
            ParNetBlock(ch2, ch2),
            ParNetBlock(ch2, ch2),
        )

        self.stream2 = nn.Sequential(
            ParNetBlock(ch3, ch3),
            ParNetBlock(ch3, ch3),
            ParNetBlock(ch3, ch3),
            ParNetBlock(ch3, ch3),
            ParNetBlock(ch3, ch3),
        )

        self.stream3 = nn.Sequential(
            ParNetBlock(ch4, ch4),
            ParNetBlock(ch4, ch4),
            ParNetBlock(ch4, ch4),
            ParNetBlock(ch4, ch4),
            ParNetBlock(ch4, ch4),
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        stream3 = self.stream3(down4)
        stream2 = self.stream2(down3)
        stream1 = self.stream1(down2)

        fusion2 = self.fusion1(stream3, stream2)
        fusion1 = self.fusion2(fusion2, stream1)

        return self.head(fusion1)

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        head = instantiate(config.head, channels=config.decoder_blocks[0])
        box_coder = instantiate(
            config.box_coder,
            output_stride=4 // head.upsample_factor,
        )

        return cls(
            input_channels=config.num_channels,
            head=head,
            box_coder=box_coder,
            encoder_blocks=config.encoder_blocks,
            decoder_blocks=config.decoder_blocks,
        )


if __name__ == "__main__":
    from pytorch_toolbelt.utils.torch_utils import count_parameters

    net = ParNet(3).cuda()
    print(count_parameters(net))
    x = torch.randn((4, 3, 512, 512)).cuda()
    y = net(x)
    print(y.size())
