import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Tuple, List, Dict, Union, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CroatoanDataset
from ..utils.utils import set_seed, features_to_device, has_batch_norm, \
    check_compile
from ..constants import DEVICE, SEED, LOGGER


def train_epoch(
    model: torch.nn.Module,
    loader_train: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    skip_one_bn: bool
) -> Tuple[torch.Tensor, torch.Tensor, float]:

    y_true, y_pred, losses = [], [], []
    model.train()

    for features, y in loader_train:
        features = features_to_device(features)
        y = y.float().to(DEVICE)

        # Skip 1 element in batch for BatchNorm layers
        if skip_one_bn:
            if len(y) == 1:
                LOGGER.warning("Skipped batch with 1 element!")
                continue

        optimizer.zero_grad()
        y_pred_loader = model(features)

        try:
            loss = criterion(y_pred_loader, y)
        except RuntimeError:
            y = y.long()
            loss = criterion(y_pred_loader, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item() * y.size(0))
        y_true.append(y)
        y_pred.append(y_pred_loader)

    y_true = torch.cat(y_true, dim=0).detach().cpu()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu()
    epoch_loss = sum(losses)/len(y_true)

    return y_true, y_pred, epoch_loss


def val_epoch(
    model: torch.nn.Module,
    loader_val: DataLoader,
    criterion: torch.nn.modules.loss._Loss
) -> Tuple[torch.Tensor, torch.Tensor, float]:

    y_true, y_pred, losses = [], [], []
    model.eval()
    state = torch.get_rng_state()

    for features, y in loader_val:
        features = features_to_device(features)
        y = y.float().to(DEVICE)

        with torch.no_grad():
            y_pred_loader = model(features)

            try:
                loss = criterion(y_pred_loader, y)
            except RuntimeError:
                y = y.long()
                loss = criterion(y_pred_loader, y)

        losses.append(loss.item() * y.size(0))
        y_true.append(y)
        y_pred.append(y_pred_loader)

    y_true = torch.cat(y_true, dim=0).detach().cpu()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu()
    epoch_loss = sum(losses)/len(y_true)

    torch.set_rng_state(state)

    return y_true, y_pred, epoch_loss


def train_val(
    model: torch.nn.Module,
    loaders: Tuple[DataLoader, DataLoader],
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    cv: bool,
    get_metrics: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
    main_metric: str,
    direction: str,
    include_compile: bool,
    include_epochs_pred: bool
) -> Tuple[Dict[str, List[float]],
           Dict[str, List[Dict[str, float]]],
           Dict[str, Union[float, Dict[str, float]]],
           Union[Dict[str, torch.Tensor], None],
           float,
           List[str],
           List[float],
           List[List[float]]]:

    start_time = time.time()
    stage = "val" if cv else "test"
    verbose = epochs // 5 if epochs // 5 > 0 else 1
    model = model.to(DEVICE)

    if include_compile:
        if check_compile():
            LOGGER.info("Compiled version of model will be used!")
            model = torch.compile(model)
        else:
            LOGGER.warning("Compiled version of model can't be used "
                           f"for `torch=={torch.__version__}`!")

    skip_one_bn = has_batch_norm(model)

    losses = defaultdict(list)
    metrics = defaultdict(list)

    model_weights = None
    pred = list()
    best_score = np.inf if direction == "minimize" else -np.inf

    log_template = "Epoch {ep:03d} train_loss: {t_loss:0.4f} "\
                   "{stage}_loss: {v_loss:0.4f} train_{metric}: "\
                   "{t_metric:0.4f} {stage}_{metric}: {v_metric:0.4f}"

    for epoch in range(epochs):
        y_true_train, y_pred_train, loss_train = \
            train_epoch(model, loaders[0], criterion, optimizer, skip_one_bn)
        y_true_val, y_pred_val, loss_val = \
            val_epoch(model, loaders[1], criterion)

        metrics_train = get_metrics(y_true_train, y_pred_train)
        metrics_val = get_metrics(y_true_val, y_pred_val)

        losses['train'].append(loss_train)
        losses[stage].append(loss_val)

        metrics['train'].append(metrics_train)
        metrics[stage].append(metrics_val)

        if include_epochs_pred:
            pred.append(y_pred_val.tolist())
        else:
            if stage == "val":
                if direction == "minimize":
                    if metrics_val[main_metric] < best_score:
                        best_score = metrics_val[main_metric]
                        pred = y_pred_val.tolist()
                elif direction == "maximize":
                    if metrics_val[main_metric] > best_score:
                        best_score = metrics_val[main_metric]
                        pred = y_pred_val.tolist()

        if epoch % verbose == 0 or epoch == epochs - 1:
            LOGGER.info(log_template.format(
                ep=epoch,
                metric=main_metric,
                stage=stage,
                t_loss=loss_train,
                v_loss=loss_val,
                t_metric=metrics_train[main_metric],
                v_metric=metrics_val[main_metric]
            ))

    if stage == "val":
        find_best_epoch = np.argmin if direction == "minimize" else np.argmax
        best_epoch = find_best_epoch(
            list(map(lambda x: metrics["val"][x][main_metric], range(epochs)))
        )
    elif stage == "test":
        best_epoch = epochs - 1
        model_weights = deepcopy(model.state_dict())
        if not include_epochs_pred:
            pred = y_pred_val.tolist()

    best_result = {
        "epoch": int(best_epoch),
        "metrics": metrics[stage][best_epoch]
    }

    LOGGER.info(f"Training is finished! Best epoch: {best_result['epoch']}. "
                f"Metrics: {best_result['metrics']}.")

    true = y_true_val.tolist()
    ids = loaders[1].dataset.ids
    end_time = time.time() - start_time

    return losses, metrics, best_result, model_weights, \
        end_time, ids, true, pred


def _get_mean_results_cv(results_per_fold, epochs, main_metric, direction):
    n_folds = len(results_per_fold)
    stages = ["train", "val"]
    metrics = list(results_per_fold[0]["metrics"]["train"][0].keys())

    mean_losses = defaultdict(lambda: np.zeros(epochs))
    mean_metrics = {}
    for stage in stages:
        mean_metrics[stage] = defaultdict(lambda: np.zeros(epochs))

    for fold in range(n_folds):
        rpf = results_per_fold[fold]
        for stage in stages:
            mean_losses[stage] += np.array(rpf['losses'][stage])
            for metric in metrics:
                mapping = map(lambda x: rpf['metrics'][stage][x][metric],
                              range(epochs))
                mean_metrics[stage][metric] += \
                    (np.array(list(mapping)) / n_folds)

    final_losses = {}
    final_metrics = defaultdict(list)
    for stage in stages:
        final_losses[stage] = list(mean_losses[stage] / n_folds)
        for epoch in range(epochs):
            mapping = map(lambda x: mean_metrics[stage][x][epoch], metrics)
            final_metrics[stage].append(dict(zip(metrics, mapping)))

    find_best_epoch = np.argmin if direction == "minimize" else np.argmax
    best_epoch = find_best_epoch(
        list(map(lambda x: final_metrics["val"][x][main_metric],
                 range(epochs)))
    )
    best_result = {
        "epoch": int(best_epoch),
        "metrics": final_metrics["val"][best_epoch]
    }

    LOGGER.info(f"Mean results for folds. Best epoch: {best_result['epoch']}."
                f" Metrics: {best_result['metrics']}.")

    return final_losses, final_metrics, best_result


def run_cv(
    cv_split: List[Dict[str, List[Union[int, str]]]],
    features: Union[str, Dict[Union[int, str], List[float]]],
    targets: Dict[Union[int, str], float],
    dataset_class: CroatoanDataset,
    loader_class: DataLoader,
    model_class: torch.nn.Module,
    optimizer_class: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    params: Dict[str, Any],
    epochs: int,
    get_metrics: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
    main_metric: str,
    direction: str,
    include_compile: bool,
    include_epochs_pred: bool
) -> Dict[str, Any]:

    LOGGER.info(f"'{DEVICE}' is being used!")
    LOGGER.info(f"Training with cv...")

    results_per_fold = []
    all_time = 0
    all_ids = []
    all_true = []
    all_pred = None
    concat_axis = 1 if include_epochs_pred else 0

    for i, ids in enumerate(cv_split):
        LOGGER.info(f"Fold {i} is being trained...")

        set_seed(SEED)

        dataset_train = dataset_class(ids["train"], features, targets)
        dataset_val = dataset_class(ids["val"], features, targets)

        loader_train = loader_class(
            dataset=dataset_train,
            batch_size=params["batch_size"],
            shuffle=True
        )
        loader_val = loader_class(
            dataset=dataset_val,
            batch_size=params["batch_size"],
            shuffle=False
        )

        model = model_class(**params["model"])
        optimizer = optimizer_class(model.parameters(), **params["optimizer"])

        losses, metrics, best_result, _, time, ids, true, pred = \
            train_val(
                model=model,
                loaders=(loader_train, loader_val),
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                cv=True,
                main_metric=main_metric,
                direction=direction,
                get_metrics=get_metrics,
                include_compile=include_compile,
                include_epochs_pred=include_epochs_pred
            )

        results = {}
        for var in ["losses", "metrics", "best_result",
                    "time", "ids", "true", "pred"]:
            results[var] = locals()[var]
        results_per_fold.append(results)

        all_time += time
        all_ids += ids
        all_true += true
        all_pred = np.array(pred) if i == 0 \
            else np.concatenate((all_pred, pred), axis=concat_axis)

    if len(cv_split) > 1:
        losses, metrics, best_result = \
            _get_mean_results_cv(
                results_per_fold=results_per_fold,
                epochs=epochs,
                main_metric=main_metric,
                direction=direction
            )

    all_results = {
        "losses": losses,
        "metrics": metrics,
        "best_result": best_result,
        "time": all_time,
        "ids": all_ids,
        "true": all_true,
        "pred": all_pred.tolist(),
        "results_per_fold": results_per_fold
    }

    if len(cv_split) == 1:
        all_results.pop("results_per_fold")

    return all_results


def run_test(
    train_test_split: Dict[str, List[Union[int, str]]],
    features: Union[str, Dict[Union[int, str], List[float]]],
    targets: Dict[Union[int, str], float],
    dataset_class: CroatoanDataset,
    loader_class: DataLoader,
    model_class: torch.nn.Module,
    optimizer_class: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    params: Dict[str, Any],
    epochs: int,
    get_metrics: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
    main_metric: str,
    direction: str,
    include_compile: bool,
    include_epochs_pred: bool
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:

    LOGGER.info(f"'{DEVICE}' is being used!")
    LOGGER.info(f"Training with test...")

    set_seed(SEED)

    dataset_train = dataset_class(
        ids=train_test_split["train"],
        features=features,
        targets=targets
    )
    dataset_test = dataset_class(
        ids=train_test_split["test"],
        features=features,
        targets=targets
    )

    loader_train = loader_class(
        dataset=dataset_train,
        batch_size=params["batch_size"],
        shuffle=True
    )
    loader_val = loader_class(
        dataset=dataset_test,
        batch_size=params["batch_size"],
        shuffle=False
    )

    model = model_class(**params["model"])
    optimizer = optimizer_class(model.parameters(), **params["optimizer"])

    losses, metrics, best_result, model_weights, time, ids, true, pred = \
        train_val(
            model=model,
            loaders=(loader_train, loader_val),
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            cv=False,
            main_metric=main_metric,
            direction=direction,
            get_metrics=get_metrics,
            include_compile=include_compile,
            include_epochs_pred=include_epochs_pred
        )

    results = {}
    for i in ["losses", "metrics", "best_result",
              "time", "ids", "true", "pred"]:
        results[i] = locals()[i]

    return results, model_weights
