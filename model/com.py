import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class DConv(nn.Module):
    def __init__(self, f_number,k=3, padding_mode='reflect') -> None:
        super().__init__()
        self.conv = nn.Conv2d(f_number, f_number, kernel_size=k, padding="same",groups=f_number, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x) 
class MLP(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x
## MLP1
class Mlp_conv(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features,
                 ffn_expansion_factor = 2,
                 bias = False,
                 
                ):
        super().__init__()
     
      
        hidden_features = int(in_features*ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(
            in_features, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
      
    def forward(self, x, training = False):
        x = self.project_in(x)
        x   = F.gelu(x)
        x =self.dwconv(x)
        x = self.project_out(x)
        return x



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
