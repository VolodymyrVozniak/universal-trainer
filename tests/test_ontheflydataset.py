import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_breast_cancer

from croatoan_trainer.preprocess import BinaryPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import OnTheFlyDataset
from croatoan_trainer.train.model import BinarySimpleMLP
from croatoan_trainer.train.metrics import get_metrics_binary


def test_ontheflydataset():
    data = load_breast_cancer()
    x = data['data']
    y = data['target']

    ids_to_features = "tests/data"
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = BinaryPreproc(ids_to_features, ids_to_targets)
    preproc.prepare_targets(reverse=True)
    preproc.random_split()

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=OnTheFlyDataset,
        loader_class=DataLoader,
        model_class=BinarySimpleMLP,
        optimizer_class=Adam,
        criterion=torch.nn.BCELoss(),
        get_metrics=get_metrics_binary,
        main_metric="f1",
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
        "batch_size": 32
    }
    results, model_weights = trainer.train(
        params=params,
        epochs=100,
        include_final=False
    )
