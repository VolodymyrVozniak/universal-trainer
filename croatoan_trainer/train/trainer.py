import logging
from typing import Dict, Callable, Any, Tuple, Union

import optuna
import torch
from torch.utils.data import DataLoader

from .dataset import CroatoanDataset
from .training import run_cv, run_test
from .tuning import EarlyStoppingCallback, objective
from ..preprocess.abstract import _Preproc
from ..tune.abstract import _Tuner

from ..constants import LOG_FILENAME, LOG_MODE, LOGGER


class Trainer():
    """
    A class used to train model on data from `BinaryPreproc`,
    `RegressionPreproc` or `MulticlassPreproc` class.

    Attributes:
        `preprocessed_data` (_Preproc):
            `BinaryPreproc`, `RegressionPreproc` or `MulticlassPreproc` class
            object with preprocessed data.
        `dataset_class` (CroatoanDataset):
            `CroatoanDataset` class or class that inherit `CroatoanDataset`.
        `loader_class` (torch.utils.data.DataLoader):
            Class for DataLoader.
        `model_class` (torch.nn.Module):
            Class for torch model.
        `optimizer_class` (torch.optim.Optimizer):
            Any class for optimizer from torch.
        `criterion` (torch.nn.modules.loss._Loss):
            Any loss from torch.
        `get_metrics` (callable):
            Function that takes two torch tensors (real and predicted values),
            computes some metrics and saves them to dict with metric's name as
            keys and metric's values as values.
        `main_metric` (str):
            Main metric (must be one of metrics defined in `get_metrics`
            function). This metric will be used to choose the best epoch on CV.
        `direction` (str):
            Direction in which we want to optimize `main_metric`. For example,
            `"minimize"` for regression problem and `mae` as `main_metric`
            or `"maximize"` for binary problem and `f1` as `main_metric`.
        `include_compile` (bool):
            Rather to use `model.compile()` if `torch` version >= `2.0.0`.

    Methods:
        `tune(tuner, epochs, n_trials, timeout, early_stopping_rounds)`:
            Tunes parameters defined in `tuner.params` using `tuner.study`.
            Returns best params that can be easily passed to `train()` method.
        `train(params, epochs, include_final, include_epochs_pred)`:
            Trains model. Trains on CV, chooses best epoch, trains on test
            with exactly that number of epochs and (optionally) trains
            final model on train + test data. Returns training results for
            each stage and model weights for final model if
            `include_final=True`, `None` otherwise.
        `print_logs()`:
            Prints training logs.
    """

    def __init__(
        self,
        preprocessed_data: _Preproc,
        dataset_class: CroatoanDataset,
        loader_class: DataLoader,
        model_class: torch.nn.Module,
        optimizer_class: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        get_metrics: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
        main_metric: str,
        direction: str,
        include_compile: bool = False
    ):
        """
        Args:
            `preprocessed_data` (_Preproc):
                `BinaryPreproc`, `RegressionPreproc` or `MulticlassPreproc`
                class object with preprocessed data.
            `dataset_class` (CroatoanDataset):
                `CroatoanDataset` class or class that inherits
                `CroatoanDataset`.
            `loader_class` (torch.utils.data.DataLoader):
                Class for DataLoader.
            `model_class` (torch.nn.Module):
                Class for torch model.
            `optimizer_class` (torch.optim.Optimizer):
                Any class for optimizer from torch.
            `criterion` (torch.nn.modules.loss._Loss):
                Any loss from torch.
            `get_metrics` (callable):
                Function that takes two torch tensors (real and predicted
                values), computes some metrics and saves them to dict with
                metric's name as keys and metric's values as values.
            `main_metric` (str):
                Main metric (must be one of metrics defined in `get_metrics`
                function). This metric will be used to choose
                the best epoch on CV.
            `direction` (str):
                Direction in which we want to optimize `main_metric`.
                For example, `"minimize"` for regression problem and `mae` as
                `main_metric` or `"maximize"` for binary problem and `f1` as
                `main_metric`.
            `include_compile` (bool):
                Rather to use `model.compile()` if `torch` version >= `2.0.0`.
        """
        self.preprocessed_data = preprocessed_data
        self.dataset_class = dataset_class
        self.loader_class = loader_class
        self.model_class = model_class
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.get_metrics = get_metrics
        self.main_metric = main_metric
        self.direction = direction
        self.include_compile = include_compile

    @staticmethod
    def _init_logs(file: bool, console: bool):
        if not file and not console:
            raise ValueError("Either `file` or `console` must be `True`!")

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        if file:
            # Due to incorrect logs in file in 'w' mode
            if LOG_MODE == "w":
                with open(LOG_FILENAME, "w") as file:
                    file.write("")
            f_handler = logging.FileHandler(LOG_FILENAME, LOG_MODE)
            f_handler.setFormatter(formatter)
        if console:
            c_handler = logging.StreamHandler()
            c_handler.setFormatter(formatter)

        LOGGER.setLevel(logging.INFO)
        LOGGER.addHandler(c_handler)
        LOGGER.addHandler(f_handler)

    def _check_split_epochs_pred(self, include_epochs_pred: bool):
        if not include_epochs_pred:
            if len(self.preprocessed_data.split["cv"]) > 1:
                raise ValueError(
                    "You can't set `include_epochs_pred` to `False` when "
                    "there are more than 1 fold in cv split!"
                )

    def print_logs(self):
        """Prints training logs."""
        with open("logs.log", "r") as file:
            file_content = file.read()
        print(file_content)

    def tune(
        self,
        tuner: _Tuner,
        epochs: int,
        n_trials: Union[int, None] = None,
        timeout: Union[int, None] = None,
        early_stopping_rounds: Union[int, None] = None
    ) -> Dict[str, Any]:
        """
        Tunes parameters defined in `tuner.params` using `tuner.study`.

        Args:
            `tuner` (_Tuner):
                `TunerTPE`, `TunerGrid` or `TunerRandom` class object.
            `epochs` (int):
                Number of epochs used to train each trial.
            `n_trials` (int):
                Number of trials. If this argument is `None`,
                there is no limitation on the number of trials. If `timeout`
                is also set to `None`, the study continues to create trials
                until it receives a termination signal such as Ctrl+C or
                SIGTERM. Default is `None`.
            `timeout` (int):
                Stop study after the given number of second(s).
                If this argument is set to `None`, the study is executed
                without time limitation. If `n_trials` is also set to `None`,
                the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
                Default is `None`.
            `early_stopping_rounds` (int):
                Number of iterations in optuna without improvements.
                If number number of iterations without improvements is larger
                than this argument optuna will finish optimization process.
                If set to `None` no early stopping will be used.
                Default is `None`.

        Returns:
            dict:
                Dictionary with best parameters with keys 'model', 'optimizer'
                and 'batch_size' that can be easily pass to `train()` method.
        """
        self._init_logs(file=False, console=True)

        if early_stopping_rounds:
            early_stopping = EarlyStoppingCallback(
                early_stopping_rounds=early_stopping_rounds,
                direction=self.direction
            )
            callbacks = [early_stopping]
        else:
            callbacks = None

        def func(trial: optuna.trial.Trial):
            return objective(
                trial=trial,
                tune_params=tuner.params,
                tune_epochs=epochs,
                cv_split=self.preprocessed_data.split["cv"],
                features=self.preprocessed_data.features,
                targets=self.preprocessed_data.targets,
                dataset_class=self.dataset_class,
                loader_class=self.loader_class,
                model_class=self.model_class,
                optimizer_class=self.optimizer_class,
                criterion=self.criterion,
                get_metrics=self.get_metrics,
                main_metric=self.main_metric,
                direction=self.direction,
                include_compile=self.include_compile
            )

        tuner.study.optimize(
            func=func,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks
        )

        return tuner.study.best_trial.user_attrs["params"]

    def train(
        self,
        params: Dict[str, Any],
        epochs: int,
        include_final: bool = True,
        include_epochs_pred: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]],
               Union[None, Dict[str, torch.Tensor]]]:
        """
        Trains model.
        Trains on CV, chooses best epoch,
        trains on test with exactly that number of epochs and
        (optionally) trains final model on train + test data.

        Args:
            `params` (dict):
                Params for model with keys: `model` (kwargs for
                `self.model_class`), `optimizer` (kwargs for
                `self.optimizer_class`) and `batch_size`.
            `epochs` (int):
                Number of epochs used to train CV.
            `include_final` (bool):
                Flag to train final model (meaning training on all data and
                check performance on all data to get a model, which can be
                used for inference). Default is `True`.
            `include_epochs_pred` (bool):
                Flag to include predictions for every sample on val and test
                sets on every epochs (can be a number or a vector).
                If `False`, keep predictions only on the best epoch.
                For large vectors as outputs and/or large amount of samples
                it is recommended to set this argument to `False`,
                because the final results may take a lot of memory.
                For now, you can set this parameter to `False` only for
                one-fold cv. Default is `True`.

        Returns:
            tuple:
                Dictionary with `cv`, `test` and `final` (optionally)
                as keys and dict with results for each stage as values
                (which contains lossses for each epoch inside `losses`,
                dict with metrics returned by `self.get_metrics` function
                for each epoch inside `metrics`, best epoch and best metrics
                inside `best_result`, training time inside `time`, list with
                unique ids inside `ids`, list with real values inside `true`
                and list with model outputs for each epoch inside `pred`)
                and model weights for final model if `include_final=True`,
                `None` otherwise.

        Raises:
            `ValueError`:
                If `include_epochs_pred` is set to `False` and there are more
                than one fold in cv.
        """
        self._check_split_epochs_pred(include_epochs_pred)

        self._init_logs(file=True, console=True)

        results = {}

        results["cv"] = run_cv(
            cv_split=self.preprocessed_data.split["cv"],
            features=self.preprocessed_data.features,
            targets=self.preprocessed_data.targets,
            dataset_class=self.dataset_class,
            loader_class=self.loader_class,
            model_class=self.model_class,
            criterion=self.criterion,
            optimizer_class=self.optimizer_class,
            params=params,
            epochs=epochs,
            get_metrics=self.get_metrics,
            main_metric=self.main_metric,
            direction=self.direction,
            include_compile=self.include_compile,
            include_epochs_pred=include_epochs_pred
        )

        epochs = results["cv"]["best_result"]["epoch"] + 1
        train_test = self.preprocessed_data.split["train_test"]
        results["test"], _ = run_test(
            train_test_split=train_test,
            features=self.preprocessed_data.features,
            targets=self.preprocessed_data.targets,
            dataset_class=self.dataset_class,
            loader_class=self.loader_class,
            model_class=self.model_class,
            criterion=self.criterion,
            optimizer_class=self.optimizer_class,
            params=params,
            epochs=epochs,
            get_metrics=self.get_metrics,
            main_metric=self.main_metric,
            direction=self.direction,
            include_compile=self.include_compile,
            include_epochs_pred=include_epochs_pred
        )

        if include_final:
            all_data = train_test["train"] + train_test["test"]
            train_test = {"train": all_data, "test": all_data}
            results["final"], model_weights = run_test(
                train_test_split=train_test,
                features=self.preprocessed_data.features,
                targets=self.preprocessed_data.targets,
                dataset_class=self.dataset_class,
                loader_class=self.loader_class,
                model_class=self.model_class,
                criterion=self.criterion,
                optimizer_class=self.optimizer_class,
                params=params,
                epochs=epochs,
                get_metrics=self.get_metrics,
                main_metric=self.main_metric,
                direction=self.direction,
                include_compile=self.include_compile,
                include_epochs_pred=include_epochs_pred
            )
        else:
            model_weights = None

        return results, model_weights
