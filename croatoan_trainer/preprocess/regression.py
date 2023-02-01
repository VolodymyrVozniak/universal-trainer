from typing import List, Dict, Union

import pandas as pd
import numpy as np

from .abstract import _Preproc


class RegressionPreproc(_Preproc):
    """
    A class used to preprocess regression data.

    Attributes:
        `features` (dict): Features for training.
        `df` (pd.DataFrame): Dataframe with unique ids,
        input and prepared targets.
        `targets` (dict): Prepared targets.
        `split` (dict): Prepared splits.
        `plotly_args` (dict): Dict with args for plotly charts.

    Methods:
        `prepare_targets(log, quantiles)`: Prepares targets.
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

    def prepare_targets(
        self,
        log: bool,
        quantiles: Union[None, List[float]] = None
    ):
        """
        Prepares targets.

        Args:
            `log` (bool): Flag to use natural log on data.
            Hint: it is useful to log data if we have a big range.
            `quantiles` (list): If specified cut tails with passed values
            (meaning interprate left quantile as min value,
            right quantile as max value and replace targets
            that don't belong to this range with these values).
            Hint: specify this parameter if there are any kind of
            outliers in the dataset (default is `None`).
        """
        targets = self.df["Input Targets"].astype('float32')

        if quantiles:
            print(f"[INFO] {quantiles} quantiles will be used!")
            min_ = targets.quantile(quantiles[0])
            max_ = targets.quantile(quantiles[1])
            print(f"[INFO] Left  quantile: {min_}")
            print(f"[INFO] Right quantile: {max_}")
            targets[targets < min_] = min_
            targets[targets > max_] = max_

        if log:
            print("[INFO] Natural log will be used!")
            targets = np.log(targets)

        self.targets = dict(zip(self.df["ID"], targets))
        self.df["Prepared Targets"] = targets

        print("[INFO] Prepared targets were successfully saved "
              "to `self.targets`!")

    # Work with splits

    @staticmethod
    def _get_stratify(df: pd.DataFrame):
        return pd.qcut(
            x=df["Prepared Targets"],
            q=20,
            labels=False,
            duplicates='drop'
        )
