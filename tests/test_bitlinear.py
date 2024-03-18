import torch

from bitlinear_pytorch import BitLinear


def test_rand_activation():
    model = BitLinear(784, 256, 8)
    x = torch.randn(128, 784)
    y = model(x)
    assert y.shape == (128, 256)
