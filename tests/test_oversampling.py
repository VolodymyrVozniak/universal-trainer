import numpy as np
from sklearn.datasets import make_classification

from croatoan_trainer.preprocess import MulticlassPreproc


def test_oversampling():
    x, y = make_classification(
        n_samples=1000,
        n_classes=3,
        weights=[0.7, 0.1, 0.2],
        n_clusters_per_class=1,
        random_state=42
    )

    ids_to_features = dict(zip(np.arange(len(y)), x))
    ids_to_targets = dict(zip(np.arange(len(y)), y))

    preproc = MulticlassPreproc(ids_to_features, ids_to_targets)
    preproc.prepare_targets()
    preproc.plot_targets()
    preproc.random_split()
    preproc.plot_split_targets(prepared=False)
    preproc.plot_split_targets(prepared=True)
    preproc.oversampling(200)
    preproc.plot_split_targets(prepared=True)
    preproc.get_split_info()
