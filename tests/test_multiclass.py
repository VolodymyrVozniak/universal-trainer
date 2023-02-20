import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.datasets import load_iris
from sklearn.metrics import jaccard_score

from croatoan_trainer.preprocess import MulticlassPreproc
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import MulticlassSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_multiclass
from croatoan_trainer.analyze import MulticlassAnalyzer


def test_multiclass():
    data = load_iris()
    x = data['data']
    y = data['target']

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = MulticlassPreproc(ids_to_features, ids_to_targets)
    preproc.plot_targets(prepared=False)
    preproc.prepare_targets()
    preproc.plot_targets(prepared=True)
    preproc.random_split(n_folds=1, val_size=0.2)
    preproc.plot_split_targets(prepared=False)
    preproc.plot_split_targets(prepared=True)
    preproc.get_split_info()

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
    assert model_weights is not None

    def postprocess_fn(model_output):
        return np.argmax(model_output, axis=1)

    analyzer = MulticlassAnalyzer(results, postprocess_fn)

    analyzer.get_stages()
    analyzer.get_metrics()
    analyzer.get_folds()
    analyzer.get_epochs("test")
    analyzer.get_time()
    analyzer.get_df_pred("cv")

    analyzer.get_df_metrics()
    analyzer.get_metric_result("final", jaccard_score, average='macro')

    analyzer.print_classification_report("test")

    analyzer.plot_all("test")
    analyzer.plot_pred_sample("final", 0)
    analyzer.plot_confusion_matrix_per_epoch(
        stage="cv",
        epochs=range(9, analyzer.get_epochs("cv"), 10)
    )
