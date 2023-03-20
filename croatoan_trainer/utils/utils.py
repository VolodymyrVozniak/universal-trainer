import random
import pickle

import numpy as np
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
