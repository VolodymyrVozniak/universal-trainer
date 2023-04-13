import operator
from typing import Any, List, Dict, Union, Callable

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from .dataset import CroatoanDataset
from .training import run_cv


class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str):
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


def objective(
    trial: optuna.trial.Trial,
    tune_params: Dict[str, Any],
    tune_epochs: int,
    cv_split: List[Dict[str, List[Union[int, str]]]],
    features: Union[str, Dict[Union[int, str], List[float]]],
    targets: Dict[Union[int, str], float],
    dataset_class: CroatoanDataset,
    loader_class: DataLoader,
    model_class: torch.nn.Module,
    optimizer_class: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    get_metrics: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
    main_metric: str,
    direction: str
) -> float:

    model_params = {}
    optimizer_params = {}

    def get_param_value(param, param_type, values):
        if param_type == "constant":
            return values
        elif param_type == "categorical":
            return trial.suggest_categorical(param, values)
        elif param_type == "float":
            return trial.suggest_float(
                name=param,
                low=values[0],
                high=values[1],
                log=values[2]
            )
        elif param_type == "int":
            return trial.suggest_int(
                name=param,
                low=values[0],
                high=values[1],
                step=values[2],
                log=values[3]
            )

    for param, (param_type, values) in tune_params["model"].items():
        model_params[param] = get_param_value(param, param_type, values)

    for param, (param_type, values) in tune_params["optimizer"].items():
        optimizer_params[param] = get_param_value(param, param_type, values)

    param_type, values = tune_params["batch_size"]
    batch_size = get_param_value("batch_size", param_type, values)

    params = {
        "model": model_params,
        "optimizer": optimizer_params,
        "batch_size": batch_size
    }

    results = run_cv(
        cv_split=cv_split,
        features=features,
        targets=targets,
        dataset_class=dataset_class,
        loader_class=loader_class,
        model_class=model_class,
        optimizer_class=optimizer_class,
        criterion=criterion,
        params=params,
        epochs=tune_epochs,
        get_metrics=get_metrics,
        main_metric=main_metric,
        direction=direction,
        include_epochs_pred=False
    )

    trial.set_user_attr("params", params)
    trial.set_user_attr("metrics", results["best_result"]["metrics"])

    return results["best_result"]["metrics"][main_metric]
