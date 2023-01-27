import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes

from croatoan_trainer.preprocess import BinaryPreproc, RegressionPreproc


data = load_breast_cancer()
x = data['data']
y = data['target']

ids_to_targets = dict(zip(np.arange(len(y)), y))
ids_to_features = dict(zip(np.arange(len(y)), x))

preproc = BinaryPreproc(ids_to_targets)
preproc.plot_targets(prepared=False)
preproc.prepare_targets(reverse=True)
# preproc.prepare_targets(log=False, quantiles=None)
preproc.plot_targets(prepared=True)
preproc.set_features(ids_to_features)
preproc.random_split()
preproc.plot_split_targets(prepared=False)
preproc.plot_split_targets(prepared=True)
info = preproc.get_split_info()
