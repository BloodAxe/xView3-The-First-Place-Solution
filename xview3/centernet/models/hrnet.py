from omegaconf import DictConfig
from pytorch_toolbelt.modules import FPNFuseSum, FPNFuse
from torch import nn

__all__ = ["CenterNetHRNet"]


class CenterNetHRNet(nn.Module):
    def __init__(self, encoder, head, embedding_dim: int, box_coder):
        super().__init__()
        self.encoder = encoder
        self.box_coder = box_coder
        self.fuse = nn.Sequential(
            FPNFuse(mode="bilinear", align_corners=True), nn.Conv2d(sum(encoder.channels), embedding_dim, kernel_size=(1, 1))
        )
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        features_fused = self.fuse(features)
        return self.head(features_fused)

    @classmethod
    def from_config(cls, config: DictConfig):
        from hydra.utils import instantiate

        encoder = instantiate(config.encoder)
        if config.num_channels != 3:
            encoder = encoder.change_input_channels(config.num_channels)

        embedding_dim = config.embedding_dim

        head = instantiate(config.head, channels=embedding_dim)
        box_coder = instantiate(
            config.box_coder,
            output_stride=encoder.strides[0] // head.upsample_factor,
        )

        return cls(encoder=encoder, box_coder=box_coder, embedding_dim=embedding_dim, head=head)
