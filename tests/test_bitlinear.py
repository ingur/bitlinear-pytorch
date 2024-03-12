import torch

from bitlinear import BitLinear


def test_bitlinear():
    model = BitLinear(784, 256, 8)
    x = torch.randn(128, 784)
    y = model(x)
    assert y.shape == (128, 256)
