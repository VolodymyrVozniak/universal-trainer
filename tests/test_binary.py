import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import jaccard_score

from croatoan_trainer.preprocess import BinaryPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import BinarySimpleMLP
from croatoan_trainer.train.metrics import get_metrics_binary
from croatoan_trainer.analyze import BinaryAnalyzer


def test_binary():
    data = load_breast_cancer()
    x = data['data']
    y = data['target']

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = BinaryPreproc(ids_to_features, ids_to_targets)
    preproc.plot_targets(prepared=False)
    preproc.prepare_targets(reverse=True)
    preproc.plot_targets(prepared=True)
    preproc.random_split()
    preproc.plot_split_targets(prepared=False)
    preproc.plot_split_targets(prepared=True)
    preproc.get_split_info()

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
    assert model_weights is None

    analyzer = BinaryAnalyzer(results)

    analyzer.get_stages()
    analyzer.get_metrics()
    analyzer.get_folds()
    analyzer.get_epochs("test")
    analyzer.get_time()
    analyzer.get_df_pred("cv")

    analyzer.get_df_metrics()
    analyzer.get_metric_result("cv", jaccard_score, True, zero_division=0)

    analyzer.print_classification_report("test")

    analyzer.plot_all("test")
    analyzer.plot_pred_sample("cv", 0)
    analyzer.plot_confusion_matrix_per_epoch(
        stage="cv",
        epochs=range(9, analyzer.get_epochs("cv"), 10)
    )
    analyzer.plot_pred_hist_per_epoch(
        stage="cv",
        epochs=range(9, analyzer.get_epochs("cv"), 10)
    )
