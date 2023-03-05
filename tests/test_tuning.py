import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_breast_cancer

from croatoan_trainer.preprocess import BinaryPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import BinarySimpleMLP
from croatoan_trainer.train.metrics import get_metrics_binary
from croatoan_trainer.tune import GridTuner, TPETuner, RandomTuner


def test_tuning():
    data = load_breast_cancer()
    x = data['data']
    y = data['target']

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = BinaryPreproc(ids_to_features, ids_to_targets)
    preproc.prepare_targets(reverse=True)
    preproc.random_split()

    tune_params_grid = {
        "model": {
            "in_features": ("constant", x.shape[1]),
            "hidden_features": ("categorical", (18, 22)),
            "dropout": ("categorical", (0.1, 0.5))
        },
        "optimizer": {
            "lr": ("constant", 1e-3),
            "weight_decay": ("constant", 5*1e-5)
        },
        "batch_size": ("categorical", (32, 64))
    }

    tune_params = {
        "model": {
            "in_features": ("constant", x.shape[1]),
            "hidden_features": ("int", (18, 22, 2, False)),
            "dropout": ("float", (0.1, 0.5, False))
        },
        "optimizer": {
            "lr": ("constant", 1e-3),
            "weight_decay": ("constant", 5*1e-5)
        },
        "batch_size": ("categorical", (32, 64))
    }

    tuner = GridTuner(
        params=tune_params_grid,
        storage=None,
        study_name="test_grid",
        direction="maximize",
        load_if_exists=False
    )

    tuner = RandomTuner(
        params=tune_params,
        storage=None,
        study_name="test_random",
        direction="maximize",
        load_if_exists=False
    )

    tuner = TPETuner(
        params=tune_params,
        storage=None,
        study_name="test_tpe",
        direction="maximize",
        load_if_exists=False
    )

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=CroatoanDataset,
        loader_class=DataLoader,
        model_class=BinarySimpleMLP,
        optimizer_class=Adam,
        criterion=torch.nn.BCELoss(),
        get_metrics=get_metrics_binary,
        main_metric="f1",
        direction="maximize"
    )

    params = trainer.tune(
        tuner=tuner,
        epochs=20,
        n_trials=2,
        timeout=None,
        early_stopping_rounds=None
    )

    results, model_weights = trainer.train(
        params=params,
        epochs=20,
        include_final=False
    )
    assert model_weights is None
