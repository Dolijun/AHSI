from torch import Tensor
import torch
from torch.nn import Module, Conv2d, Linear, GELU, Parameter, LayerNorm, Identity
from timm.models.layers import DropPath


class ConvNeXtBlock(Module):
    def __init__(self, dim, drop_path=0.1, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, 4 * dim)
        self.act = GELU()
        self.pwconv2 = Linear(4 * dim, dim)
        self.gamma = Parameter(layer_scale_init_value * torch.ones((dim)),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)

        return x

