import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def replace_linear_with_bitlinear(model: nn.Module, b: int = 8) -> None:
    """
    Replaces all linear layers in a model with BitLinear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, BitLinear(module.in_features, module.out_features, b))
        else:
            replace_linear_with_bitlinear(module)


class BitLinear(nn.Linear):
    """
    Implements a BitLinear layer as described in "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"

    :param in_features: Number of input features
    :param out_features: Number of output features
    :param b: Number of bits to use for activation quantization
    """

    def __init__(self, in_features: int, out_features: int, b: int = 8) -> None:
        super().__init__(in_features, out_features, False)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.bweight = torch.zeros_like(self.weight)
        self.Qb = 2 ** (b - 1) - 1
        self.eps = 1e-5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def bin_weights(self) -> None:
        gamma = self.weight.abs().mean() + self.eps
        w = torch.round(self.weight / gamma)
        self.bweight = torch.clamp(w, min=-1, max=1)

    def absmax_quantization(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = x.abs().max() + self.eps
        q = x * self.Qb / gamma
        q = torch.clamp(q, min=-self.Qb + self.eps, max=self.Qb - self.eps).round()
        return q, gamma

    def forward(self, input) -> torch.Tensor:
        self.bin_weights()
        x = self.norm(input)
        x_q, gamma = self.absmax_quantization(x)
        x_q = F.linear(x_q, BitLinear.ste(self.bweight, self.weight))
        beta = torch.linalg.norm(self.weight, ord=1)
        x = x_q * gamma * beta / self.Qb
        return x

    @staticmethod
    def ste(xb, xf):
        """
        Straight-Through Estimator (STE) function.
        Allows gradients to be passed through a non-differentiable operation during backpropagation by
        returning the gradient of a proxy operation. Uses quantized weights during forward pass, but
        the gradient of the full-precision weights during backpropagation.

        :param xb: Quantized weights
        :param xf: Full-precision weights
        :return: Quantized weights with the gradient of the full-precision weights
        """
        return xf + (xb - xf).detach()
