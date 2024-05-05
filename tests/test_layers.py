import torch

from kanwise.layers import KANLayer


def test_kan_layer_initialization():
    """
    Test initialization of KANLayer.
    """
    input_dim = 10
    output_dim = 5
    spline_points = 4
    layer = KANLayer(input_dim, output_dim, spline_points)
    assert layer.input_dim == input_dim
    assert layer.output_dim == output_dim
    assert layer.spline_points == spline_points
    assert layer.spline_coeffs.shape == (output_dim, input_dim, spline_points, 4)


def test_kan_layer_forward():
    """
    Test the forward pass of KANLayer.
    """
    input_dim = 10
    output_dim = 1
    spline_points = 4
    layer = KANLayer(input_dim, output_dim, spline_points)
    x = torch.randn(2, input_dim)  # Simulate a batch of 2 samples
    output = layer(x)
    assert output.shape == (2, output_dim)
