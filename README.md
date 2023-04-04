![example workflow](https://github.com/VolodymyrVozniak/universal-trainer/actions/workflows/test.yml/badge.svg)

# Table of Contents
<ul>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#dependencies">Dependencies</a></li>
  <li><a href="#tutorials">Tutorials</a></li>
  <li>
    <a href="#usage">Usage</a>
    <ul>
      <li><a href="#preprocessing">Preprocessing<a></li>
      <li><a href="#training">Training</a></li>
      <li><a href="#tuning">Tuning</a></li>
      <li><a href="#analyzing">Analyzing</a></li>
    </ul>
  </li>
</ul>

</br>

# Installation

To install this repo as Python lib just run the following command:

```sh
pip install git+https://github.com/VolodymyrVozniak/universal-trainer
```

</br>

# Dependencies

```sh
torch>=1.13.0
scikit-learn>=1.2.0
pandas>=1.5.2
plotly>=5.7.0
optuna>=2.10.0
```

</br>

# Tutorials

1. For binary problem check [this tutorial](https://colab.research.google.com/drive/1s21Mn0ieNo5YJ4qLNDFBPEUTC874UfcK)
2. For regression problem check [this tutorial](https://colab.research.google.com/drive/1PA7bFGQRGazfSBhF8yQoAo7ocq0UGWi-)
3. For multiclassification problem check [this tutorial](https://colab.research.google.com/drive/1zW_I4JRRvCOoo5oNB1U3fWCfaJC8gB6s)

</br>

# Usage

## Preprocessing

There are 3 main classes for preprocessing:
* `BinaryPreproc` - for preprocessing binary data;
* `RegressionPreproc` - for preprocessing regression data;
* `MulticlassPreproc` - for preprocessing multiclassification data.

Preprocessing pipeline

[MAIN]

1. Prepare data for initializing preproc class.
    1. You will need a dictionary with unique ids as keys (unique ids can be just unique int numbers defined by `np.arange()` function) and features as values. You can somehow preprocess these features (one entry at a time) in Dataset class when train model, but if you want to use default `CroatoanDataset`, already define your feautures as final lists for each entry).
    2. Also, you will need a dictionary with unique ids as keys (these unique ids must match ids defined for features, meaning target with specific unique id must match features with this specific id) and targets as values.
2. Prepare targets for training.
    1. Plot input targets histogram (distribution) and define if we need to reverse targets (for binary problems) or log data and cut tails by quantiles (for regression problems).
    2. Prepare targets with already defined arguments.
    3. Plot prepared targets histogram (distribution) to check the difference and correctness.
3. Split data.
    1. Split data with random splitting type.
    2. For each splitting type you can get main info and plot targets histograms (distributions) for all sets and folds.

[EXTRA]

4. Oversampling (for binary and multiclassification problems only)
    1. Oversample each class label to reach `min_count` by adding extra ids to `self.split` for train.
5. Feature scaling.
    1. Scale features using scaler from sklearn (fit scaler on train data got from splitting, transform all features using this scaler and save the scaler to class attribute).

Examples

* Binary problem

```python
import numpy as np
from sklearn.datasets import load_breast_cancer

from croatoan_trainer.preprocess import BinaryPreproc


# Load example data
data = load_breast_cancer()
x = data['data']
y = data['target']

# Make dict with unique ids as keys and features as values
ids_to_features = dict(zip(np.arange(len(y)), x))

# Make dict with unique ids as keys and targets as values
ids_to_targets = dict(zip(np.arange(len(y)), y))

# Initialize preproc class
preproc = BinaryPreproc(ids_to_features, ids_to_targets)

# Plot input targets histogram
preproc.plot_targets(prepared=False)

# Define if we need to reverse our targets
preproc.prepare_targets(reverse=True)

# Plot prepared targets histogram
preproc.plot_targets(prepared=True)

# Split data
preproc.random_split(
    test_size=0.2,
    n_folds=5,
    val_size=None,
    seed=51983
)

# Plot input targets histograms
preproc.plot_split_targets(prepared=False)

# Plot prepared targets histograms
preproc.plot_split_targets(prepared=True)

# Get info about splitting
split_info = preproc.get_split_info()

# Scale features
preproc.scale_features("Standard")
```

For more details check [tutorial](https://colab.research.google.com/drive/1s21Mn0ieNo5YJ4qLNDFBPEUTC874UfcK)

<p align="right">(<a href="#top">back to top</a>)</p>

* Regression problem

```python
import numpy as np
from sklearn.datasets import load_diabetes

from croatoan_trainer.preprocess import RegressionPreproc


# Load example data
data = load_diabetes()
x = data['data']
y = data['target']

# Make dict with unique ids as keys and features as values
ids_to_features = dict(zip(np.arange(len(y)), x))

# Make dict with unique ids as keys and targets as values
ids_to_targets = dict(zip(np.arange(len(y)), y))

# Initialize preproc class
preproc = RegressionPreproc(ids_to_features, ids_to_targets)

# Plot input targets histogram
preproc.plot_targets(prepared=False)

# Define if we need to reverse our targets
preproc.prepare_targets(log=False, quantiles=None)

# Plot prepared targets histogram
preproc.plot_targets(prepared=True)

# Split data
preproc.random_split(
    test_size=0.2,
    n_folds=5,
    val_size=None,
    seed=51983
)

# Plot input targets histograms
preproc.plot_split_targets(prepared=False)

# Plot prepared targets histograms
preproc.plot_split_targets(prepared=True)

# Get info about splitting
split_info = preproc.get_split_info()
```

For more details check [tutorial](https://colab.research.google.com/drive/1PA7bFGQRGazfSBhF8yQoAo7ocq0UGWi-)

<p align="right">(<a href="#top">back to top</a>)</p>

* Multiclassification problem

```python
import numpy as np
from sklearn.datasets import load_iris

from croatoan_trainer.preprocess import MulticlassPreproc


# Load example data
data = load_iris()
x = data['data']
y = data['target']

# Make dict with unique ids as keys and features as values
ids_to_features = dict(zip(np.arange(len(y)), x))

# Make dict with unique ids as keys and targets as values
ids_to_targets = dict(zip(np.arange(len(y)), y))

# Initialize preproc class
preproc = MulticlassPreproc(ids_to_features, ids_to_targets)

# Plot input targets histogram
preproc.plot_targets(prepared=False)

# Define if we need to reverse our targets
preproc.prepare_targets()

# Plot prepared targets histogram
preproc.plot_targets(prepared=True)

# Split data
preproc.random_split(
    test_size=0.2,
    n_folds=5,
    val_size=None,
    seed=51983
)

# Plot input targets histograms
preproc.plot_split_targets(prepared=False)

# Plot prepared targets histograms
preproc.plot_split_targets(prepared=True)

# Get info about splitting
split_info = preproc.get_split_info()
```

For more details check [tutorial](https://colab.research.google.com/drive/1zW_I4JRRvCOoo5oNB1U3fWCfaJC8gB6s)

<p align="right">(<a href="#top">back to top</a>)</p>

## Training

There is 1 main class for training:
* `Trainer` - train binary, regression or multiclassification problem.

Training pipeline
1. Trains in CV mode (meaning trains model on train set of specific fold and checks model performance on val set of specific fold with passed value for epochs and gets avarage performance on each epoch by avaraging scores for all folds), chooses best epoch and saves all results (losses, metrics on each epoch for train and val sets, best result, training time, unique ids, true values and predicted values on each epoch for val set). Results on each fold are also saved.
2. Trains in test mode (meaning trains model on train set and checks model performance on test set with chosen number of epochs on the CV stage) and saves all results (losses, metrics on each epoch for train and test sets, best result, training time, unique ids, true values and predicted values on each epoch for test set).
3. Trains in final mode (meaning trains model on all data with chosen number of epochs on the CV stage) and saves all results (losses, metrics on each epoch for train and test sets, best result, training time, unique ids, true values and predicted values on each epoch for test set). Here train and test are the same: all possible data, but the metrics can differ, because train set is always shuffled, while test set isn't. You can skip this step py passing `include_final=False` when call the `train()` method.

Examples

* Binary problem

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import BinarySimpleMLP
from croatoan_trainer.train.metrics import get_metrics_binary


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
    "batch_size": 32,
}

results, model_weights = trainer.train(
    params=params,
    epochs=100,
    inlcude_final=True,
    include_epochs_pred=True
)
```

For more details check [tutorial](https://colab.research.google.com/drive/1s21Mn0ieNo5YJ4qLNDFBPEUTC874UfcK)

<p align="right">(<a href="#top">back to top</a>)</p>

* Regression problem

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import RegressionSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_regression


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
        "dropout": 0.25
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 5*1e-5
    },
    "batch_size": 32,
}

results, model_weights = trainer.train(
    params=params,
    epochs=100,
    inlcude_final=True,
    include_epochs_pred=True
)
```

For more details check [tutorial](https://colab.research.google.com/drive/1PA7bFGQRGazfSBhF8yQoAo7ocq0UGWi-)

<p align="right">(<a href="#top">back to top</a>)</p>

* Multiclassification problem

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import MulticlassSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_multiclass


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
}

results, model_weights = trainer.train(
    params=params,
    epochs=100,
    inlcude_final=True,
    include_epochs_pred=True
)
```

For more details check [tutorial](https://colab.research.google.com/drive/1zW_I4JRRvCOoo5oNB1U3fWCfaJC8gB6s)

<p align="right">(<a href="#top">back to top</a>)</p>

## Tuning

There are 3 main classes for defining tuner:
* `TPETuner` - for tuning parameters using TPE (Tree-structured Parzen Estimator) algorithm (optuna default). For more details check [this link](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler);
* `RandomTuner` - for tuning parameters using random sampling. For more details check [this link](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler);
* `GridTuner` - for tuning parameters using grid search. For more details check [this link](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html#optuna.samplers.GridSampler).

Tuning pipeline
1. Initialize tuner class with params for tuning.
2. Tune parameters using trainer class and `tune()` method.
3. Get best params and pass them to `train()` method.

Examples

* Binary problem + TPETuner

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from croatoan_trainer.tune import TPETuner
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import BinarySimpleMLP
from croatoan_trainer.train.metrics import get_metrics_binary


tune_params = {
    "model": {
        "in_features": ("constant", x.shape[1]),
        "hidden_features": ("int", (18, 22, 2, False)),
        "dropout": ("float", (0.1, 0.5, False))
    },
    "optimizer": {
        "lr": ("constant", 1e-3),
        "weight_decay": ("constant", 5*1e-5)
    },
    "batch_size": ("categorical", (32, 64))
}

tuner = TPETuner(
    params=tune_params,
    storage=None,
    study_name="binary",
    direction="maximize",
    load_if_exists=False
)

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

params = trainer.tune(
    tuner=tuner,
    epochs=20,
    n_trials=2,
    timeout=None,
    early_stopping_rounds=None
)

results, model_weights = trainer.train(
    params=params,
    epochs=100,
    include_final=True,
    include_epochs_pred=True
)
```

For more details check [tutorial](https://colab.research.google.com/drive/1s21Mn0ieNo5YJ4qLNDFBPEUTC874UfcK)

<p align="right">(<a href="#top">back to top</a>)</p>

* Regression problem + RandomTuner

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from croatoan_trainer.tune import RandomTuner
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import RegressionSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_regression


tune_params = {
    "model": {
        "in_features": ("constant", x.shape[1]),
        "hidden_features": ("int", (18, 22, 2, False)),
        "dropout": ("float", (0.1, 0.5, False))
    },
    "optimizer": {
        "lr": ("constant", 1e-3),
        "weight_decay": ("constant", 5*1e-5)
    },
    "batch_size": ("categorical", (32, 64))
}

tuner = RandomTuner(
    params=tune_params,
    storage=None,
    study_name="regression",
    direction="minimize",
    load_if_exists=False
)

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

params = trainer.tune(
    tuner=tuner,
    epochs=20,
    n_trials=2,
    timeout=None,
    early_stopping_rounds=None
)

results, model_weights = trainer.train(
    params=params,
    epochs=100,
    include_final=True,
    include_epochs_pred=True
)
```

For more details check [tutorial](https://colab.research.google.com/drive/1PA7bFGQRGazfSBhF8yQoAo7ocq0UGWi-)

<p align="right">(<a href="#top">back to top</a>)</p>

* Multiclassification problem + GridTuner

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from croatoan_trainer.tune import GridTuner
from croatoan_trainer.train import Trainer
from croatoan_trainer.train.dataset import CroatoanDataset
from croatoan_trainer.train.model import MulticlassSimpleMLP
from croatoan_trainer.train.metrics import get_metrics_multiclass


tune_params = {
    "model": {
        "in_features": ("constant", x.shape[1]),
        "hidden_features": ("categorical", (18, 20, 22)),
        "output_features": ("constant", 3),
        "dropout": ("categorical", (0.1, 0.25, 0.5)),
    },
    "optimizer": {
        "lr": ("constant", 1e-3),
        "weight_decay": ("constant", 5*1e-5)
    },
    "batch_size": ("categorical", (32, 64))
}

tuner = GridTuner(
    params=tune_params,
    storage=None,
    study_name="multiclass",
    direction="maximize",
    load_if_exists=False
)

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

params = trainer.tune(
    tuner=tuner,
    epochs=20,
    n_trials=2,
    timeout=None,
    early_stopping_rounds=None
)

results, model_weights = trainer.train(
    params=params,
    epochs=100,
    include_final=True,
    include_epochs_pred=True
)
```

For more details check [tutorial](https://colab.research.google.com/drive/1zW_I4JRRvCOoo5oNB1U3fWCfaJC8gB6s)

<p align="right">(<a href="#top">back to top</a>)</p>

## Analyzing

There are 3 main classes for analyzing:
* `BinaryAnalyzer` - for analyzing binary problem's training results;
* `RegressionAnalyzer` - for analyzing regression problem's training results;
* `MulticlassAnalyzer` - for analyzing multiclassification problem's training results.

Analyzing pipeline
1. Initialize analyze class with results got after training.
2. Plot different charts and analyze results.
3. Get dataframe with final metrics for each stage (`cv`, `test` or `final`).

<ins>**REMINDER!**</ins> The main stage is always `test`, not `final` (`test` is how your model performs on data that it didn't see; `final` is how your model performs on data that it used for training).

Examples

* Binary problem

```python
from croatoan_trainer.analyze import BinaryAnalyzer


analyzer = BinaryAnalyzer(results)

analyzer.plot_all("cv")
analyzer.plot_all("test")
analyzer.plot_all("final")

metrics = analyzer.get_df_metrics()
```

For more details check [tutorial](https://colab.research.google.com/drive/1s21Mn0ieNo5YJ4qLNDFBPEUTC874UfcK)

<p align="right">(<a href="#top">back to top</a>)</p>

* Regression problem

```python
from croatoan_trainer.analyze import RegressionAnalyzer


analyzer = RegressionAnalyzer(results)

analyzer.plot_all("cv")
analyzer.plot_all("test")
analyzer.plot_all("final")

metrics = analyzer.get_df_metrics()
```

For more details check [tutorial](https://colab.research.google.com/drive/1PA7bFGQRGazfSBhF8yQoAo7ocq0UGWi-)

<p align="right">(<a href="#top">back to top</a>)</p>

* Multiclassification problem

```python
from croatoan_trainer.analyze import MulticlassAnalyzer


def postprocess_fn(model_output):
    return np.argmax(model_output, axis=1)


analyzer = MulticlassAnalyzer(results, postprocess_fn)

analyzer.plot_all("cv")
analyzer.plot_all("test")
analyzer.plot_all("final")

metrics = analyzer.get_df_metrics()
```

For more details check [tutorial](https://colab.research.google.com/drive/1zW_I4JRRvCOoo5oNB1U3fWCfaJC8gB6s)

<p align="right">(<a href="#top">back to top</a>)</p>
