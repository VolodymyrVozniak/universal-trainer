import logging

import torch


torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 785372

LOG_FILENAME = "logs_2.log"
LOG_MODE = "w"
LOGGER = logging.getLogger(__name__)
