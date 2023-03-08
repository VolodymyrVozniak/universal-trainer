from typing import Dict, Union, List

import pandas as pd

from .abstract import _Preproc


class _ClassificationPreproc(_Preproc):
    def __init__(
        self,
        ids_to_features: Dict[Union[int, str], List[float]],
        ids_to_targets: Dict[Union[int, str], float]
    ):
        super().__init__(ids_to_features, ids_to_targets)

    def oversampling(self, min_count: int):
        """
        Oversamples each class label to reach `min_count`
        by adding extra ids to `self.split` for train.

        Args:
            `min_count`: each class label will have this
            minimum entries for training.
        """
        self._checker(self.split, "random_split")

        def get_new_train_df(train_df: pd.DataFrame) -> pd.DataFrame:

            counter = train_df["Prepared Targets"].value_counts().to_dict()

            new_df = pd.DataFrame()
            for label, count in counter.items():
                label_df = train_df[train_df["Prepared Targets"] == label]

                if count < min_count:
                    ratio = int(min_count / count) + 1
                    label_df = pd.concat([label_df for i in range(ratio)])
                    label_df = label_df.iloc[:min_count]

                new_df = pd.concat([new_df, label_df])

            return new_df

        df = self.df.copy()
        train_df = df[df["ID"].isin(self.split["train_test"]["train"])]
        new_train_df = get_new_train_df(train_df)
        self.split["train_test"]["train"] = new_train_df["ID"].tolist()

        for i, fold in enumerate(self.split["cv"]):
            train_df = new_train_df[new_train_df["ID"].isin(fold["train"])]
            self.split["cv"][i]["train"] = train_df["ID"].tolist()

    @staticmethod
    def _get_stratify(df: pd.DataFrame):
        return df["Prepared Targets"]
