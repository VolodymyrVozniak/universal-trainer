import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_breast_cancer, fetch_california_housing

from croatoan_trainer.preprocess import BinaryPreproc, RegressionPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.metrics import get_metrics_binary, \
    get_metrics_regression


class CroatoanModel(Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
            # nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor):
        return self.model(data)


if __name__ == "__main__":
    data = fetch_california_housing()
    x = data['data']
    y = data['target']

    ids_to_targets = dict(zip(np.arange(len(y)), y))
    ids_to_features = dict(zip(np.arange(len(y)), x))

    preproc = RegressionPreproc(ids_to_targets)
    preproc.plot_targets(prepared=False)
    # preproc.prepare_targets(reverse=True)
    preproc.prepare_targets(log=False, quantiles=None)
    preproc.plot_targets(prepared=True)
    preproc.set_features(ids_to_features)
    preproc.random_split()
    preproc.plot_split_targets(prepared=False)
    preproc.plot_split_targets(prepared=True)
    info = preproc.get_split_info()

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=CroatoanDataset,
        loader_class=DataLoader,
        model_class=CroatoanModel,
        optimizer_class=Adam,
        criterion=nn.MSELoss(),
        get_metrics=get_metrics_regression,
        main_metric="r2",
        direction="maximize"
    )

    params = {
        "model": {
            "in_features": x.shape[1],
            "hidden_features": 20,
            "dropout": 0.25
        },
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 5*1e-5
        },
        "batch_size": 32,
        "epochs": 100
    }
    results, model_weights = trainer.train(params)
