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
    data = load_iris()
    x = data['data']
    y = data['target']

    ids_to_targets = dict(zip(np.arange(len(y)), y))
    ids_to_features = dict(zip(np.arange(len(y)), x))

    preproc = MulticlassPreproc(ids_to_targets)
    # preproc.plot_targets(prepared=False)
    # preproc.prepare_targets(reverse=True)
    # preproc.prepare_targets(log=False, quantiles=None)
    preproc.prepare_targets()
    # preproc.plot_targets(prepared=True)
    preproc.set_features(ids_to_features)
    preproc.random_split()
    # preproc.plot_split_targets(prepared=False)
    # preproc.plot_split_targets(prepared=True)
    info = preproc.get_split_info()

    trainer = Trainer(
        preprocessed_data=preproc,
        dataset_class=CroatoanDataset,
        loader_class=DataLoader,
        model_class=MulticlassSimpleMLP,
        optimizer_class=Adam,
        criterion=torch.nn.CrossEntropyLoss(),
        get_metrics=get_metrics_multiclass,
        main_metric="f1",
        direction="maximize"
    )

    params = {
        "model": {
            "in_features": x.shape[1],
            "hidden_features": 20,
            "output_features": 3,
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

    analyzer = MulticlassAnalyzer(results, postprocess_fn)
