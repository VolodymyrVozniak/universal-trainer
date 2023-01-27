from typing import List, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset


class CroatoanDataset(Dataset):
    """
    Dataset that will be used for training. Usually uses data
    from `BinaryPreproc` or `RegressionPreproc` class
    """
    def __init__(
        self,
        ids: List[Union[int, str]],
        features: Dict[Union[int, str], List[float]],
        targets: Dict[Union[int, str], float]
    ):
        """
        Args:
            `ids`: List with unique ids
            `features`: Dict with unique ids as keys and features as values
            `targets`: Dict with unique ids as keys and targets as values
        """
        self.ids = ids
        self.features = features
        self.targets = targets

    def process_features(self, features: List[float]) -> torch.Tensor:
        """
        Processes features for one entry

        Args:
            features (list): Features for one entry

        Returns:
            torch.Tensor: new features that will go to torch model
        """
        return torch.Tensor(features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        id_ = self.ids[index]
        features = self.process_features(self.features[id_])
        return features, self.targets[id_]

    def __len__(self) -> int:
        return len(self.ids)
