import random
import pickle
import pkg_resources
from typing import Union, List, Any

import numpy as np
import torch
import torch.nn as nn

from ..constants import DEVICE


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def features_to_device(
    features: Union[List[torch.Tensor], torch.Tensor]
) -> Union[List[torch.Tensor], torch.Tensor]:
    try:
        if isinstance(features, list):
            return list(map(lambda x: x.to(DEVICE), features))
        else:
            return features.to(DEVICE)
    except Exception as error:
        raise ValueError(
            f"Moving features to `{DEVICE}` cannot be done with the "
            f"following error: {error}!"
        )


def has_batch_norm(model: torch.nn.Module) -> bool:
    bn_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, bn_layers):
            return True
    return False


def check_compile() -> bool:
    torch_version = pkg_resources.get_distribution("torch").version
    if int(torch_version.split(".")[0]) == 2:
        return True
    else:
        return False
