from typing import List, Dict, Union

from .classification import _ClassificationPreproc


class BinaryPreproc(_ClassificationPreproc):
    """
    A class used to preprocess binary data.

    Attributes:
        `features` (dict): Features for training.
        `df` (pd.DataFrame): Dataframe with unique ids,
        input and prepared targets.
        `targets` (dict): Prepared targets.
        `split` (dict): Prepared splits.
        `scaler` (sklearn.scaler): Scaler from sklearn fitted on
        train data from `self.split` if self.scale_features()
        was called, None otherwise.
        `plotly_args` (dict): Dict with args for plotly charts.

    Methods:
        `prepare_targets(reverse)`: Prepares targets.
        `plot_targets(prepared)`: Plots targets.
        `random_split(test_size, n_folds, val_size, seed)`: Splits data
        in random mode.
        `get_split_info()`: Gets split's info as dataframe.
        `plot_split_targets(prepared)`: Plots split targets.
        `scale_features(scaler, **kwargs)`: Scale features using scaler
        from sklearn.
        `oversampling(min_count)`: Oversamples each class label to reach
        `min_count` by adding extra ids to `self.splt` for train.
        `set_plotly_args(**kwargs)`: Sets args for plotly charts.
    """

    def __init__(
        self,
        ids_to_features: Dict[Union[int, str], List[float]],
        ids_to_targets: Dict[Union[int, str], float]
    ):
        super().__init__(ids_to_features, ids_to_targets)

    def prepare_targets(self, reverse: bool):
        """
        Prepares targets.

        Args:
            `reverse` (bool): Flag to reverse targets.
            Hint: it is useful to have more samples of 0 class,
            because usually we are trying to optimize F1 metric.
        """
        targets = self.df["Input Targets"].astype('float32')

        if reverse:
            print("[INFO] Reverse targets will be used!")
            targets = 1 - targets

        self.targets = dict(zip(self.df["ID"], targets))
        self.df["Prepared Targets"] = targets

        print("[INFO] Prepared targets were successfully saved "
              "to `self.targets`!")
