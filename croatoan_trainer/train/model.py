import torch
import torch.nn as nn
from torch.nn import Module


class BinarySimpleMLP(Module):
    """
    Simple MLP for binary problem
    (2 hidden layers; dropout layers; ReLU activation and
    Sigmoid activation at the end)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float
    ):
        """
        Args:
            `in_features` (int): Number of features for input
            `hidden_features (ine)`: Number of features for hidden layers
            `dropout` (float): Dropout for dropout layers
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)


class RegressionSimpleMLP(Module):
    """
    Simple MLP for regression problem
    (2 hidden layers; dropout layers; ReLU activation)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float
    ):
        """
        Args:
            `in_features` (int): Number of features for input
            `hidden_features (ine)`: Number of features for hidden layers
            `dropout` (float): Dropout for dropout layers
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)
