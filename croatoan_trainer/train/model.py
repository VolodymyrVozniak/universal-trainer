import torch
import torch.nn as nn
from torch.nn import Module

from ..constants import DEVICE


class BinarySimpleMLP(Module):
    """
    Simple MLP for binary problem
    (2 hidden layers; dropout layers; ReLU activation and
    Sigmoid activation at the end).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float
    ):
        """
        Args:
            `in_features` (int): Number of features for input.
            `hidden_features (ine)`: Number of features for hidden layers.
            `dropout` (float): Dropout for dropout layers.
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
        return self.model(data.to(DEVICE)).reshape(-1)


class RegressionSimpleMLP(Module):
    """
    Simple MLP for regression problem
    (2 hidden layers; dropout layers; ReLU activation).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float
    ):
        """
        Args:
            `in_features` (int): Number of features for input.
            `hidden_features` (int): Number of features for hidden layers.
            `dropout` (float): Dropout for dropout layers.
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
        return self.model(data.to(DEVICE)).reshape(-1)


class MulticlassSimpleMLP(Module):
    """
    Simple MLP for multiclassification problem
    (2 hidden layers; dropout layers; ReLU activation).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        output_features: int,
        dropout: float
    ):
        """
        Args:
            `in_features` (int): Number of features for input.
            `hidden_features` (int): Number of features for hidden layers.
            `output_features` (int): Number of features for output.
            `dropout` (float): Dropout for dropout layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, output_features)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.to(DEVICE))
