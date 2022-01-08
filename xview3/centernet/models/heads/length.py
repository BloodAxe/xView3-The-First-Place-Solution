import math

import torch
from torch import nn, Tensor

from xview3.constants import MIN_OBJECT_LENGTH_M, MAX_OBJECT_LENGTH_M, PIX_TO_M

__all__ = ["LengthParametrization"]


class LengthParametrization(nn.Module):
    def __init__(self, min_length=MIN_OBJECT_LENGTH_M / PIX_TO_M, max_length=MAX_OBJECT_LENGTH_M / PIX_TO_M):
        super().__init__()
        min_length = math.log(min_length)
        max_length = math.log(max_length)
        self.scale = float(max_length - min_length)
        self.bias = float(min_length)

    def forward(self, x: Tensor) -> Tensor:
        log_len = x.sigmoid() * self.scale + self.bias
        return log_len.exp()


if __name__ == "__main__":
    m = LengthParametrization()
    x = torch.tensor([-10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10])
    print(m(x))
