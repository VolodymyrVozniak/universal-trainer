import logging
from typing import Dict, Callable, Any, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import CroatoanDataset
from .training import run_cv, run_test
from ..preprocess.preprocessor import _Preproc


class Trainer():
    """
    A class used to train model on data from
    `BinaryPreproc` or `RegressionPreproc` class

    Attributes:
        `preprocessed_data` (_Preproc): `BinaryPreproc` or `RegressionPreproc`
        class object with preprocessed data
        `dataset_class` (CroatoanDataset): `CroatoanDataset` class or class
        that inherit `CroatoanDataset`
        `loader_class` (torch.utils.data.Loader): Class for DataLoader
        `model_class` (torch.nn.Module): Class for torch model
        `optimizer_class` (torch.optim.Optimizer): Any class for optimizer
        from torch
        `criterion` (torch.nn.modules.loss._Loss): Any loss from torch
        `get_metrics` (callable): Function that takes two torch tensors
        (real and predicted targets), computes some metrics and saves them
        to dict with metric's name as keys and metric's values as values
        `main_metric` (str): Main metric (must be one of metrics defined in
        `get_metrics` function). This metric will be used to choose
        the best epoch on CV
        `direction` (str): Direction in which we want to optimize
        `main_metric`. For example, `"minimize"` for regression problem and
        `mae` as `main_metric` or `"maximize"` for binary problem
        and `f1` as `main_metrics`

    Methods:
        `train(params)`: Trains model. Trains on CV, chooses best epoch,
        then trains on test with exactly that number of epochs
        and then trains final model on train + test data and returns
        training results for each stage and model weights for final model
        `print_logs()`: Prints training logs
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
        direction: str
    ):
        """
        Args:
            `preprocessed_data` (_Preproc): `BinaryPreproc` or
            `RegressionPreproc` class object with preprocessed data
            `dataset_class` (CroatoanDataset): `CroatoanDataset` class or class
            that inherit `CroatoanDataset`
            `loader_class` (torch.utils.data.Loader): Class for DataLoader
            `model_class` (torch.nn.Module): Class for torch model
            `optimizer_class` (torch.optim.Optimizer): Any class for optimizer
            from torch
            `criterion` (torch.nn.modules.loss._Loss): Any loss from torch
            `get_metrics` (callable): Function that takes two torch tensors
            (real and predicted targets), computes some metrics and saves them
            to dict with metric's name as keys and metric's values as values
            `main_metric` (str): Main metric (must be one of metrics defined in
            `get_metrics` function). This metric will be used to choose
            the best epoch on CV
            `direction` (str): Direction in which we want to optimize
            `main_metric`. For example, `"minimize"` for regression problem and
            `mae` as `main_metric` or `"maximize"` for binary problem
            and `f1` as `main_metrics`
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

    @staticmethod
    def _init_logs(file: bool, console: bool):
        if not file and not console:
            raise ValueError("Either `file` or `console` must be `True`!")
        handlers = []
        if file:
            handlers.append(logging.FileHandler("logs.log", mode="w"))
        if console:
            handlers.append(logging.StreamHandler())
        logging.basicConfig(
            handlers=handlers,
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            force=True
        )

    def print_logs(self):
        """
        Prints training logs
        """
        with open("logs.log", "r") as file:
            file_content = file.read()
        print(file_content)

    def train(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
        """
        Trains model.
        Trains on CV, chooses best epoch, then trains on test with exactly
        that number of epochs and then trains final model on train + test data

        Args:
            `params` (dict): Params for model with keys: `model`
            (kwargs for `self.model_class`), `optimizer` (kwargs for
            `self.optimizer_class`), `batch_size` and `epochs`

        Returns:
            tuple: dictionary with `cv`, `test` and `final` as keys and
            dict with results for each stage as values and model weights
            for final model
        """
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
            get_metrics=self.get_metrics,
            main_metric=self.main_metric,
            direction=self.direction
        )

        params["epochs"] = results["cv"]["best_result"]["epoch"] + 1
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
            get_metrics=self.get_metrics,
            main_metric=self.main_metric,
            direction=self.direction
        )

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
            get_metrics=self.get_metrics,
            main_metric=self.main_metric,
            direction=self.direction
        )

        return results, model_weights
