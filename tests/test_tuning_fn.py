import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_breast_cancer

from croatoan_trainer.preprocess import BinaryPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.metrics import get_metrics_binary
from croatoan_trainer.tune import TPETuner


class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        super(CustomModel, self).__init__()

        in_features = kwargs["in_features"]
        fc_layers = [nn.BatchNorm1d(in_features)]
        activation = getattr(nn, kwargs["activation"])()

        for i in range(kwargs["n_layers"]):
            out_features = kwargs[f"n_units_l{i}"]
            fc_layers.append(nn.Linear(in_features, out_features))

            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(kwargs[f"dropout_l{i}"]))

            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 1))
        fc_layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*fc_layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layers(data).reshape(-1)


def get_tune_params(trial: optuna.trial.Trial):
    model_params, optimizer_params = {}, {}

    model_params['in_features'] = 30  # some constant value

    model_params['activation'] = trial.suggest_categorical(
        'activation', ['ReLU', 'GELU', 'ELU', 'LeakyReLU']
    )

    n_layers = trial.suggest_int('n_layers', 2, 4)
    model_params['n_layers'] = n_layers

    for i in range(n_layers):
        n_units = trial.suggest_categorical(
            f'n_units_l{i}', (512, 1024, 2048)
        )
        dropout = trial.suggest_float(
            f'dropout_l{i}', 0.1, 0.5
        )
        model_params[f'n_units_l{i}'] = n_units
        model_params[f'dropout_l{i}'] = dropout

    optimizer_params['lr'] = trial.suggest_float(
        'lr', 1e-5, 1e-1, log=True
    )
    optimizer_params['weight_decay'] = trial.suggest_float(
        'weight_decay', 5e-5, 5e-3, log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size", (16, 32, 64, 128, 256, 512, 1024, 2048)
    )

    return model_params, optimizer_params, batch_size


def test_tuning():
    data = load_breast_cancer()
    x = data['data']
    y = data['target']

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = BinaryPreproc(ids_to_features, ids_to_targets)
    preproc.prepare_targets(reverse=True)
    preproc.random_split()

    tuner = TPETuner(
        params=get_tune_params,
        storage=None,
        study_name="test_tpe",
        direction="maximize",
        load_if_exists=False
    )

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=CroatoanDataset,
        loader_class=DataLoader,
        model_class=CustomModel,
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
