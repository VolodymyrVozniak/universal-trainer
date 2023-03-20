import os
from typing import List, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset

from ..utils.utils import load_pkl


class CroatoanDataset(Dataset):
    """
    Dataset that will be used for training.

    Attributes:
        `ids` (list): List with unique ids.
        `features` (dict): Dictionary with unique ids as keys
        and features as values.
        `targets` (dict): Dictionary with unique ids as keys
        and targets as values.

    Methods:
        `process_features(features)`: Processes features for one entry.
    """
    def __init__(
        self,
        ids: List[Union[int, str]],
        features: Dict[Union[int, str], List[float]],
        targets: Dict[Union[int, str], float]
    ):
        """
        Args:
            `ids` (list): List with unique ids.
            `features` (dict): Dictionary with unique ids as keys
            and features as values.
            `targets` (dict): Dictionary with unique ids as keys
            and targets as values.
        """
        self.ids = ids
        self.features = features
        self.targets = targets

    def process_features(self, features: List[float]) -> torch.Tensor:
        """
        Processes features for one entry.

        Args:
            features (list): Features for one entry.

        Returns:
            torch.Tensor: New features that will go to torch model.
        """
        return torch.Tensor(features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        id_ = self.ids[index]
        features = self.process_features(self.features[id_])
        return features, self.targets[id_]

    def __len__(self) -> int:
        return len(self.ids)


class OnTheFlyDataset(Dataset):
    """
    Dataset that will be used for training.

    Attributes:
        `ids` (list): List with unique ids.
        `features` (str): Path for folder with features saved to
        different `.pkl` files where file names are unique ids and
        file contents are features for particular unique id.
        `targets` (dict): Dictionary with unique ids as keys
        and targets as values.

    Methods:
        `process_features(features)`: Processes features for one entry.
    """
    def __init__(
        self,
        ids: List[Union[int, str]],
        features: str,
        targets: Dict[Union[int, str], float]
    ):
        """
        Args:
            `ids` (list): List with unique ids.
            `features` (dict): Path for folder with features saved to
            different `.pkl` files where file names are unique ids and
            file contents are features for particular unique id.
            `targets` (dict): Dictionary with unique ids as keys
            and targets as values.
        """
        self.ids = ids
        self.features = features
        self.targets = targets

    def process_features(self, features: List[float]) -> torch.Tensor:
        """
        Processes features for one entry.

        Args:
            features (list): Features for one entry.

        Returns:
            torch.Tensor: New features that will go to torch model.
        """
        return torch.Tensor(features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        id_ = self.ids[index]

        features_path = os.path.join(self.features, f"{id_}.pkl")
        try:
            features = load_pkl(features_path)
        except FileNotFoundError:
            raise ValueError(f"Unknown file format inside `{self.features}`! "
                             "You can use only `.pkl` files!")

        features = self.process_features(features)

        return features, self.targets[id_]

    def __len__(self) -> int:
        return len(self.ids)
