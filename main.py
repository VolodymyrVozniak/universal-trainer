import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from croatoan_trainer.preprocess import BinaryPreproc, RegressionPreproc, \
    MulticlassPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.analyze import BinaryAnalyzer, RegressionAnalyzer, \
    MulticlassAnalyzer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import BinarySimpleMLP, \
    RegressionSimpleMLP, MulticlassSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_binary, \
    get_metrics_regression, get_metrics_multiclass


if __name__ == "__main__":
    data = load_diabetes()
    x = data['data']
    y = data['target']

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = RegressionPreproc(ids_to_features, ids_to_targets)
    # preproc.plot_targets(prepared=False)
    # preproc.prepare_targets(reverse=True)
    preproc.prepare_targets(log=False, quantiles=None)
    # preproc.prepare_targets()
    # preproc.plot_targets(prepared=True)
    preproc.random_split()
    # preproc.plot_split_targets(prepared=False)
    # preproc.plot_split_targets(prepared=True)
    info = preproc.get_split_info()

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=CroatoanDataset,
        loader_class=DataLoader,
        model_class=RegressionSimpleMLP,
        optimizer_class=Adam,
        criterion=torch.nn.MSELoss(),
        get_metrics=get_metrics_regression,
        main_metric="r2",
        direction="maximize"
    )

    params = {
        "model": {
            "in_features": x.shape[1],
            "hidden_features": 20,
            # "output_features": 3,
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

    def postprocess_fn(model_output):
        return np.argmax(model_output, axis=1)

    analyzer = RegressionAnalyzer(results)
    analyzer.plot_hist_per_epoch("test", [0, 25, 36, 56])
    analyzer.plot_kde_per_epoch("test", [0, 25, 36, 56])
