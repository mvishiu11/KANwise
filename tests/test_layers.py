import torch

from kanwise.kan import KANLinear


def test_kan_layer_initialization():
    """
    Test initialization of KANLayer.
    """
    input_dim = 10
    output_dim = 5
    layer = KANLinear(input_dim, output_dim)
    assert layer.in_features == input_dim
    assert layer.out_features == output_dim


def test_kan_layer_forward():
    """
    Test the forward pass of KANLayer.
    """
    input_dim = 10
    output_dim = 1
    spline_points = 4
    layer = KANLinear(input_dim, output_dim, spline_points)
    x = torch.randn(2, input_dim)  # Simulate a batch of 2 samples
    output = layer(x)
    assert output.shape == (2, output_dim)
