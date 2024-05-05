import torch.nn as nn

from .layers import KANLayer


class KANModel(nn.Module):
    def __init__(self):
        """
        Initialize the KAN Model.
        """
        super(KANModel, self).__init__()
        self.layers = nn.Sequential(
            KANLayer(10, 20), nn.ReLU(), KANLayer(20, 10), nn.ReLU(), KANLayer(10, 1)
        )

    def forward(self, x):
        """
        Forward pass of the KAN Model.
        :param x: Tensor, input to the model
        """
        return self.layers(x)
