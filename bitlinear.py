import torch
from torch import nn
import torch.nn.functional as F
from gemm_lowbit_extension import gemm_lowbit_kernel
__all__ = ["BitLinear"]

# def activation_quant(x):
#     scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
#     y = (x * scale).round().clamp_(-128, 127) / scale
#     return y

# def weight_quant(w):
#     scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
#     u = (w * scale).round().clamp_(-1, 1)
#     return u, scale

# class BitLinear(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True):
#         super(BitLinear, self).__init__(in_features, out_features, bias)
#         self.norm = nn.LayerNorm(in_features)
#         # 为量化权重和缩放因子初始化存储
#         self.register_buffer('quant_weight', None)
#         self.register_buffer('weight_scale', None)

#     def forward(self, x):
#         print(self.quant_weight)
#         w = self.weight
#         x_norm = self.norm(x)

#         # 使用量化函数并保存量化权重和缩放因子
#         x_quant = activation_quant(x_norm)
#         self.quant_weight, self.weight_scale = weight_quant(w)
#         quant_weight = self.quant_weight / self.weight_scale
#         # 通过.detach()实现直通估计（STE）
#         x_quant = x_norm + (x_quant - x_norm).detach()
#         w_quant = w + (quant_weight - w).detach()

#         y = F.linear(x_quant, w_quant, self.bias)
#         return y


def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale

# def weight_quant(w):
#     scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
#     u = (w * scale).round().clamp_(-1, 1)
#     return u, scale

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.norm = nn.LayerNorm(in_features)
        # 为量化权重和缩放因子初始化存储
        self.register_buffer('quant_weight', None)
        self.register_buffer('weight_scale', None)

    def forward(self, x):
        print(self.quant_weight)
        w = self.quant_weight
        w_scale = self.weight_scale
        x_norm = self.norm(x)

        # 使用量化函数并保存量化权重和缩放因子
        # x_quant = activation_quant(x_norm)
        # self.quant_weight, self.weight_scale = weight_quant(w)
        # quant_weight = self.quant_weight / self.weight_scale
        x_quant, x_scale = activation_quant(x_norm)
        y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale
        return y