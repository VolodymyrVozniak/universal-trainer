from typing import List, Dict, Union

import pandas as pd

from .abstract import _Preproc


class MulticlassPreproc(_Preproc):
    """
    A class used to preprocess multiclassification data.

    Attributes:
        `features` (dict): Features for training.
        `df` (pd.DataFrame): Dataframe with unique ids,
        input and prepared targets.
        `targets` (dict): Prepared targets.
        `split` (dict): Prepared splits.
        `plotly_args` (dcit): Dict with args for plotly charts.

    Methods:
        `prepare_targets()`: Prepares targets.
        `plot_targets(prepared)`: Plots targets.
        `random_split(test_size, n_folds, val_size, seed)`: Splits data
        in random mode.
        `get_split_info()`: Gets split's info as dataframe.
        `plot_split_targets(prepared)`: Plots split targets.
        `set_plotly_args(**kwargs)`: Sets args for plotly charts.
    """

    def __init__(
        self,
        ids_to_features: Dict[Union[int, str], List[float]],
        ids_to_targets: Dict[Union[int, str], float]
    ):
        super().__init__(ids_to_features, ids_to_targets)

    # Work with targets

    def prepare_targets(self):
        """Prepares targets."""
        targets = self.df["Input Targets"].astype('float32')

        self.targets = dict(zip(self.df["ID"], targets))
        self.df["Prepared Targets"] = targets

        print("[INFO] Prepared targets were successfully saved "
              "to `self.targets`!")

    # Work with splits

    @staticmethod
    def _get_stratify(df: pd.DataFrame):
        return df["Prepared Targets"]
