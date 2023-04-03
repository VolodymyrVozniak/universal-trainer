import random
import pickle

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def has_batch_norm(model: torch.nn.Module):
    bn_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    for module in model.modules():
        if isinstance(module, bn_layers):
            return True
    return False
