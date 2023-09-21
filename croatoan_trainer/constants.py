import sys

import torch
from loguru import logger


torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 785372

LOG_MODE = "w"
LOG_FILENAME = "logs.log"
LOG_FORMAT = "<green>{time:DD.MM.YYYY HH:mm:ss}</green> | "\
             "<level>{level:<7}</level> | <level>{message}</level>"

logger.remove(0)
logger.add(sys.stderr, format=LOG_FORMAT)
