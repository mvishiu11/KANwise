import torch
import torch.nn as nn


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, spline_points=4):
        """
        Initialize the KAN Layer with spline-based activation functions.
        Each output node has a set of splines defined across the input dimensions.
        """
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_points = spline_points  # Should be at least 4 for cubic splines

        # Create parameters for the splines. Each spline needs at least 4 points (for cubic).
        # We store the coefficients of the cubic polynomial segments.
        self.spline_coeffs = nn.Parameter(
            torch.randn(output_dim, input_dim, spline_points, 4)
        )

    def cubic_spline(self, x, coeffs):
        """
        Compute the cubic spline for input x with given coefficients.
        :param x: Tensor, the input features for the spline, shape (batch_size,)
        :param coeffs: Tensor, the coefficients of the cubic polynomial, shape (4,)
        """
        # Expand coeffs to match the batch dimension if necessary
        coeffs = coeffs.unsqueeze(0).expand(x.size(0), -1, -1)
        # Compute cubic polynomial
        return torch.sum(
            coeffs[..., 0] * x.unsqueeze(-1) ** 3
            + coeffs[..., 1] * x.unsqueeze(-1) ** 2
            + coeffs[..., 2] * x.unsqueeze(-1)
            + coeffs[..., 3],
            dim=-1,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Handling single or multiple features
        num_features = x.shape[1]
        for i in range(self.output_dim):
            activation = torch.zeros(batch_size, device=x.device)
            for j in range(num_features):
                x_norm = (x[:, j] - x[:, j].min()) / (x[:, j].max() - x[:, j].min())
                activation += self.cubic_spline(x_norm, self.spline_coeffs[i, j])
            output[:, i] = activation

        return output
