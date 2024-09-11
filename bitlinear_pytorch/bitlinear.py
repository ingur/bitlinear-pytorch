import torch.nn as nn
from torch.nn import functional as F


def act_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp_(-128, 127) / scale


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale


class BitLinear(nn.Linear):
    """
    Implements the BitLinear layer as described in "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"

    :param in_features: Number of input features
    :param out_features: Number of output features
    :bias: If set to False, the layer will not learn an additive bias. Default: True
    :device: The device to use. Default: None
    :dtype: The datatype to use. Default: None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x):
        x_quant = x + (act_quant(x) - x).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_quant, w_quant, self.bias)
