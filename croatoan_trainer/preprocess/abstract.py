import os
from collections import defaultdict
from typing import List, Any, Dict, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
    RobustScaler, MaxAbsScaler

from ..base import _Base
from ..utils.plotting import plot_targets_hist, plot_split_targets_hist


class _Preproc(_Base):
    def __init__(
        self,
        ids_to_features: Union[str, Dict[Union[int, str], List[float]]],
        ids_to_targets: Dict[Union[int, str], float],
    ):
        """
        Args:
            `ids_to_features` ([dict; str]):
                Dict with unique ids as keys and features as values
                (that will be used in torch Dataset while training model)
                or path for folder with features saved to different `.pkl`
                files, where file names are unique ids and file contents
                are features for particular unique id.
            `ids_to_targets` (dict):
                Dict with unique ids as keys and input targets as values.

        Raises:
            `ValueError`:
                If there is wrong instance for `ids_to_features` argument.
                Only dict and str are allowed.
            `ValueError`:
                If there are duplicates in `ids_to_targets` or
                `ids_to_features` dicts keys.
            `ValueError`:
                If there are not the same unique ids in `ids_to_features`
                and `ids_to_targets` dicts and `ids_to_features` is str
                (meaning path for folder with features saved to different
                `.pkl` files).
        """
        self._check_ids(ids_to_features, ids_to_targets)

        self.features = ids_to_features
        self.df = self._make_df(ids_to_targets)

        self.targets = {}
        self.split = defaultdict(dict)
        self.scaler = None

        self.set_plotly_args(font_size=14, template="plotly_dark", bargap=0.2)

    @staticmethod
    def _checker(param: Any, fn_name: str):
        assert param, f"Run `{fn_name}()` method first!"

    @staticmethod
    def _check_ids(
        ids_to_features: Union[str, Dict[Union[int, str], List[float]]],
        ids_to_targets: Dict[Union[int, str], float],
    ):
        targets_ids = set(ids_to_targets.keys())

        if isinstance(ids_to_features, dict):
            features_ids = set(ids_to_features.keys())
        elif isinstance(ids_to_features, str):
            type_ = type(list(ids_to_targets.keys())[0])
            features_ids = set(
                [type_(i.split(".")[0]) for i in os.listdir(ids_to_features)]
            )
        else:
            raise ValueError(
                "Can't process features! Pass either dict with unique ids as "
                "keys and features as values or str path with unique ids as "
                "file names and features as file contents!"
            )

        if len(targets_ids - features_ids) > 0:
            print("[WARNING] Unique ids for `ids_to_features` and "
                  f"`ids_to_targets` are not the same ({len(targets_ids)} "
                  f"for `ids_to_targets` and {len(features_ids)} for "
                  "`ids_to_features`). Deleting unnecessary from "
                  "`ids_to_targets`...")
            unnecessary_ids = targets_ids - features_ids
            for id_ in unnecessary_ids:
                ids_to_targets.pop(id_)

        if len(features_ids - targets_ids) > 0:
            if isinstance(ids_to_features, dict):
                print("[WARNING] Unique ids for `ids_to_features` and "
                      f"`ids_to_targets` are not the same ({len(features_ids)}"
                      f" for `ids_to_features` and {len(targets_ids)} for "
                      "`ids_to_targets`). Deleting unnecessary from "
                      "`ids_to_features`...")
                unnecessary_ids = targets_ids - features_ids
                for id_ in unnecessary_ids:
                    ids_to_features.pop(id_)

            elif isinstance(ids_to_features, str):
                raise ValueError(
                    "Unique ids for `ids_to_features` and `ids_to_targets` "
                    f"are not the same ({len(features_ids)} for "
                    f"`ids_to_features` and {len(targets_ids)} for "
                    "`ids_to_targets`)."
                )

    @staticmethod
    def _make_df(
        ids_to_targets: Dict[Union[int, str], float]
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            data=ids_to_targets.items(),
            columns=["ID", "Input Targets"]
        )
        if df["ID"].nunique() != len(df):
            raise ValueError(
                "There are duplicates in unique ids! Please check it and "
                "assign new unique ids without duplicates!"
            )
        return df

    # Work with targets

    def plot_targets(self, prepared: bool = False):
        """
        Plots targets.

        Args:
            `prepared` (bool):
                Flag to plot prepared targets. If `True` plot prepared targets,
                plot input targets otherwise. Default is `False`.
        """
        if prepared:
            self._checker(self.targets, "prepare_targets")
            target_col = "Prepared Targets"
        else:
            target_col = "Input Targets"

        plot_targets_hist(
            df=self.df,
            target_column=target_col,
            plotly_args=self.plotly_args
        )

    # Work with splits

    @staticmethod
    def _check_split_params(**kwargs):
        if kwargs["test_size"] <= 0:
            raise ValueError("`test_size` must be > 0!")

        if kwargs["n_folds"] < 1:
            raise ValueError("`n_folds` must be >= 1!")

        if kwargs["n_folds"] == 1:
            if kwargs["val_size"] is None:
                raise ValueError("`val_size` must be specified "
                                 "if `n_folds` == `1`!")
            if kwargs["val_size"] <= 0:
                raise ValueError("`val_size` must be > 0 if `n_folds` == `1`!")

        if kwargs["n_folds"] > 1 and kwargs["val_size"] is not None:
            raise ValueError("`val_size` must be `None` if `n_folds` > 1 "
                             "or set `val_size` to `1` and use `val_size`")

    @staticmethod
    def _check_splits(
        train_test: Dict[str, List[Union[int, str]]],
        cv: List[Dict[str, List[Union[int, str]]]]
    ):
        def check_train_test(split, fold):
            stage = "test" if fold is None else "val"
            assert len(set(split["train"]) & set(split[stage])) == 0, \
                f"train and {stage} contain same ids! (Fold {fold})"

        check_train_test(train_test, fold=None)

        if len(cv) > 1:
            cv_tests = []
            for i, fold in enumerate(cv):
                check_train_test(fold, fold=i)
                cv_tests += fold["val"]

            checker_1 = len(set(train_test["train"]) - set(cv_tests)) == 0
            checker_2 = len(train_test["train"]) == len(cv_tests)
            assert checker_1 and checker_2, \
                f"train from train-test and vals from cv don't match!"

        else:
            cv_test = set(cv[0]["train"]) | set(cv[0]["val"])
            assert len(set(train_test["train"]) - set(cv_test)) == 0, \
                f"train from train-test and train-val from cv don't match!"

    def _split_intro(self, **kwargs):
        self._checker(self.targets, "prepare_targets")
        self._check_split_params(**kwargs)

    def _split_outro(
        self,
        train_test: Dict[str, List[Union[int, str]]],
        cv: List[Dict[str, List[Union[int, str]]]]
    ):
        self._check_splits(train_test, cv)

        self.split["train_test"] = train_test
        self.split["cv"] = cv

        print("[INFO] Train-test split was successfully saved to "
              "`self.split['train_test']`!\n"
              "[INFO] CV split was successfully saved to "
              "`self.split['cv']`!")

    def random_split(
        self,
        test_size: float = 0.2,
        n_folds: int = 5,
        val_size: Union[None, float] = None,
        seed: int = 51983
    ):
        """
        Splits data in random mode. Keeps only unique ids from
        `self.targets` keys after preparing targets in the correct sets
        (meaning one id only in train or test set for train-test
        and one id in train set or val set for CV).

        Args:
            `test_size` (float):
                Fraction of data for test. Default is `0.2`.
            `n_folds` (int):
                Number of CV folds. Can be `1` meaning validation set.
                If `1`, `val_size` must be specified. Default is `5`.
            `val_size` (float):
                Fraction of data for validation. Must be specified if
                `n_folds` == `1`. Default is `None`.
            `seed` (int):
                Seed for splittin. Default is `51983`.
        """
        params = locals()
        params.pop("self")
        self._split_intro(**params)

        train, test = train_test_split(
            self.df,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=self._get_stratify(self.df)
        )
        train_test = {
            "train": list(train["ID"]),
            "test": list(test["ID"])
        }

        if n_folds == 1:
            val_size = val_size * len(self.df) / len(train)
            train, val = train_test_split(
                train,
                test_size=val_size,
                shuffle=True,
                random_state=seed,
                stratify=self._get_stratify(train)
            )
            cv = [{"train": list(train["ID"]),
                   "val": list(val["ID"])}]
        else:
            splits = StratifiedKFold(n_splits=n_folds)
            y = self._get_stratify(train)
            cv = []
            for train_index, val_index in splits.split(train, y):
                fold = {
                    "train": list(train.iloc[train_index]["ID"]),
                    "val": list(train.iloc[val_index]["ID"])
                }
                cv.append(fold)

        self._split_outro(train_test, cv)

    def get_split_info(self) -> pd.DataFrame:
        """
        Gets split's info as dataframe.

        Returns:
            pd.DataFrame: Dataframe with split's info.
        """
        self._checker(self.split, "random_split")

        train_ids = self.split["train_test"]["train"]
        test_ids = self.split["train_test"]["test"]

        train_count, test_count = len(train_ids), len(test_ids)
        all_count = train_count + test_count
        counts = [all_count, train_count, test_count]
        percents = [1, train_count / all_count, test_count / all_count]
        info_indexes = ["All", "Train", "Test"]

        for i, fold in enumerate(self.split["cv"]):
            train_count, val_count = len(fold["train"]), len(fold["val"])
            counts += [train_count, val_count]
            percents += [train_count / all_count, val_count / all_count]
            info_indexes += [f"Train_{i}", f"Val_{i}"]

        info = pd.DataFrame(
            {"count": counts, "%": np.round(percents, 3) * 100},
            index=info_indexes
        )

        return info

    def plot_split_targets(self, prepared: bool = False):
        """
        Plots split targets.

        Args:
            `prepared` (bool):
                Flag to plot prepared targets. If `True` plot prepared targets,
                plot input targets otherwise. Default is `False`.
        """
        self._checker(self.split, "random_split")

        if prepared:
            target_col = "Prepared Targets"
            targets = self.targets
        else:
            target_col = "Input Targets"
            targets = dict(zip(self.df["ID"], self.df[target_col]))

        plot_split_targets_hist(
            split=self.split,
            targets=targets,
            id_column="ID",
            target_column=target_col,
            plotly_args=self.plotly_args
        )

    @staticmethod
    def _get_stratify():
        pass

    def scale_features(self, scaler: str, **kwargs):
        """
        Scale features using scaler from sklearn.
        Fit scaler on train data got from `random_split()` method,
        transform all features using this scaler, save them to
        `self.features` and save the scaler to `self.scaler` attribute.

        Args:
            `scaler` (str):
                Type of scaler to use from `sklearn`.
                Now you can either use 'Standard', 'MinMax', 'Robust',
                or 'MaxAbs'.
            `**kwargs`:
                Any additional parameters for scaler from `sklearn`.
        """
        self._checker(self.split, "random_split")

        available_scalers = ["Standard", "MinMax", "Robust", "MaxAbs"]
        if scaler not in available_scalers:
            raise ValueError(f"`{scaler}` is not supported! "
                             f"Choose one of {available_scalers}!")

        if scaler == "Standard":
            self.scaler = StandardScaler(**kwargs)
        elif scaler == "MinMax":
            self.scaler = MinMaxScaler(**kwargs)
        elif scaler == "Robust":
            self.scaler = RobustScaler(**kwargs)
        elif scaler == "MaxAbs":
            self.scaler = MaxAbsScaler(**kwargs)

        df = pd.DataFrame(self.features.values())
        df["ID"] = self.features.keys()

        del self.features

        train_ids = df[df["ID"].isin(
            self.split["train_test"]["train"]
        )]["ID"].to_numpy()
        X_train = df[df["ID"].isin(
            self.split["train_test"]["train"]
        )].drop(columns="ID")

        test_ids = df[df["ID"].isin(
            self.split["train_test"]["test"]
        )]["ID"].to_numpy()
        X_test = df[df["ID"].isin(
            self.split["train_test"]["test"]
        )].drop(columns="ID")

        del df

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        train_ids = np.concatenate((train_ids, test_ids)).tolist()
        X_train = np.concatenate((X_train, X_test)).tolist()

        self.features = dict(zip(train_ids, X_train))
