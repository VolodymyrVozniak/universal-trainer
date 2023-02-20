import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_diabetes
from sklearn.metrics import median_absolute_error

from croatoan_trainer.preprocess import RegressionPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import RegressionSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_regression
from croatoan_trainer.analyze import RegressionAnalyzer


def test_regression():
    data = load_diabetes()
    x = data['data']
    y = data['target']

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = RegressionPreproc(ids_to_features, ids_to_targets)
    preproc.plot_targets(prepared=False)
    preproc.prepare_targets(log=False, quantiles=None)
    preproc.plot_targets(prepared=True)
    preproc.random_split()
    preproc.plot_split_targets(prepared=False)
    preproc.plot_split_targets(prepared=True)
    preproc.get_split_info()

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=CroatoanDataset,
        loader_class=DataLoader,
        model_class=RegressionSimpleMLP,
        optimizer_class=Adam,
        criterion=torch.nn.MSELoss(),
        get_metrics=get_metrics_regression,
        main_metric="mae",
        direction="minimize"
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
    assert model_weights is not None

    analyzer = RegressionAnalyzer(results)

    analyzer.get_stages()
    analyzer.get_metrics()
    analyzer.get_folds()
    analyzer.get_epochs("test")
    analyzer.get_time()
    analyzer.get_df_pred("cv")

    analyzer.get_df_metrics()
    metric = analyzer.get_metric_result("final", median_absolute_error)
    assert metric is not None

    analyzer.plot_all("test")
    analyzer.plot_pred_sample("final", 0)
    analyzer.plot_pred_per_epoch(
        stage="cv",
        epochs=range(9, analyzer.get_epochs("cv"), 10)
    )
    analyzer.plot_hist_per_epoch(
        stage="cv",
        epochs=range(9, analyzer.get_epochs("cv"), 10)
    )
    analyzer.plot_kde_per_epoch(
        stage="cv",
        epochs=range(9, analyzer.get_epochs("cv"), 10)
    )
