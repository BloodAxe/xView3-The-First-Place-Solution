import collections
import glob
import os

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn

from xview3 import *


def inspect_outputs(outputs):
    if torch.is_tensor(outputs):
        str_repr = f"{list(outputs.size())} mean={outputs.mean().item()} std={outputs.std().item()}"
    elif isinstance(outputs, collections.Mapping):
        str_repr = []
        for key, value in outputs.items():
            str_repr.append(f"\t{key} = {inspect_outputs(value)})")
        str_repr = "Mapping:\n" + "\n".join(str_repr)
    elif isinstance(outputs, collections.Iterable):
        str_repr = []
        for index, output in enumerate(outputs):
            str_repr.append(f"\t[{index}] = {inspect_outputs(output)})")
        str_repr = "List:\n" + "\n".join(str_repr)
    else:
        raise NotImplemented
    return str_repr


@torch.no_grad()
@torch.cuda.amp.autocast(True)
@pytest.mark.parametrize("config_filename", glob.glob("../configs/model/centernet/*.yaml"))
def test_all_centernet_models(config_filename):
    config = open(config_filename, "r").read()
    # Manually resolve interpolation
    num_channels = 2
    num_classes = 3
    config = config.replace("${dataset.num_classes}", str(num_classes))
    config = config.replace("${dataset.num_channels}", str(num_channels))
    config = OmegaConf.create(config)

    assert config.config.slug == fs.id_from_fname(os.path.dirname(config_filename)) + "_" + fs.id_from_fname(config_filename)

    net = get_detection_model(config).cuda().eval()

    print()
    print(config.config.slug)
    print(count_parameters(net, keys=("encoder", "decoder", "neck", "head"), human_friendly=True))
    input = torch.randn((2, num_channels, 512, 512)).cuda()
    output = net(input)

    for name, value in output.items():
        print(
            f"{name:30}",
            tuple(value.size()),
            f"{value.min().item():.2f}",
            f"{value.mean().item():.2f}",
            "+/-",
            f"{value.std().item():.2f}",
            f"{value.max().item():.2f}",
        )
    print()


@pytest.mark.parametrize("config_filename", glob.glob("../configs/loss/centernet/*.yaml"))
def test_centernet_losses(config_filename):
    config = open(config_filename, "r").read()
    config = OmegaConf.create(config)

    for criterion_cfg in config["losses"]:
        criterion = instantiate(criterion_cfg, **{"_convert_": "all"})
        assert config["slug"] == fs.id_from_fname(config_filename)
        assert isinstance(criterion["loss"], nn.Module)
