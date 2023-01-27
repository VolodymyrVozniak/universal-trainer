import os
from collections import defaultdict
from typing import List, Any, Dict, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

from ..base import _Base
from ..utils.plotting import plot_targets_hist, plot_split_targets_hist


class _Preproc(_Base):
    def __init__(
        self,
        ids_to_targets: Dict[Any, float],
    ):
        """
        Args:
            `ids_to_targets` (dict): Dict with unique ids as keys
            and input targets as values
        """
        self.input_features = ids_to_targets

        self.targets = {}
        self.features = {}
        self.split = defaultdict(dict)

        self.set_plotly_args(font_size=14, template="plotly_dark")

    @staticmethod
    def _checker(param: Any, fn_name: str):
        assert param, f"Run `{fn_name}()` method first!"

    # Work with targets

    def plot_targets(self, preprocessed: bool = False):
        """
        Plots targets

        Args:
            `preprocessed` (bool): Flag to plot preprocessed targets.
            If `True` plot preprocessed targets, plot input targets otherwise
            (default is `False`)
        """
        if preprocessed:
            self._checker(self.targets, "prepare_targets")
            plot_targets_hist(
                targets=self.targets,
                info="Preprocessed Targets",
                plotly_args=self.plotly_args
            )
        else:
            plot_targets_hist(
                targets=self.input_targets,
                info="Input Targets",
                plotly_args=self.plotly_args
            )

    # Work with features

    def set_features(self, ids_to_features: Dict[Any, Any]):
        self.features = ids_to_features
        print("[INFO] Features were successfully saved to `self.features`!")

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
                                 "if `n_folds` == 1!")
            if kwargs["val_size"] <= 0:
                raise ValueError("`val_size` must be > 0 if `n_folds` == 1!")

        if kwargs["n_folds"] > 1 and kwargs["val_size"] is not None:
            raise ValueError("`val_size` must be None if `n_folds` > 1 "
                             "or set `val_size` to 1 and use `val_size`")

    @staticmethod
    def _check_splits(
        train_test: Dict[str, Union[str, float]],
        cv: List[Dict[str, Union[str, float]]]
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

    def _split_intro(self, split_type: str, **kwargs):
        self.get_targets()
        self._check_split_params(**kwargs)
        self._update_config(f"{split_type}_split", **kwargs)

    def _split_outro(
        self,
        split_type: str,
        train_test: Dict[str, Union[str, float]],
        cv: List[Dict[str, Union[str, float]]]
    ):
        self._check_splits(train_test, cv)

        self.split[split_type]["train_test"] = train_test
        self.split[split_type]["cv"] = cv

        print("[INFO] Train-test split was successfully saved to "
              f"`self.split['{split_type}']['train_test']`!\n"
              "[INFO] CV split was successfully saved to "
              f"`self.split['{split_type}']['cv']`!")

        info, similarity = self.split_info(split_type)

        self.split[split_type]["info"] = info
        self.split[split_type]["similarity"] = similarity

        print("[INFO] Info dataframe was successfully saved to "
              f"`self.split['{split_type}']['info']`!\n"
              "[INFO] Similarity dataframe was successfully saved to "
              f"`self.split['{split_type}']['similarity']`!")

    def random_split(
        self,
        test_size: float = 0.2,
        n_folds: int = 5,
        val_size: Union[None, float] = None,
        seed: int = 51983
    ):
        """
        Splits data in random mode meaning mols are not taken into account

        Args:
            `test_size` (float): Fraction of data for test (default is `0.2`)
            `n_folds` (int): Number of CV folds.
            Can be `1` meaning validation set.
            If `1` `val_size` must be specified (default is `5`)
            `val_size` (float): Fraction of data for validation.
            Must be specified if `n_folds` == `1` (default is `None`)
            `seed` (int): Seed for splitting (default is `51983`)
        """
        params = locals()
        params.pop("self")
        self._split_intro("random", **params)

        train, test = train_test_split(
            self.df,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=self._get_stratify(self.df)
        )
        train_test = {
            "train": list(train[self.unique]),
            "test": list(test[self.unique])
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
            cv = [{"train": list(train[self.unique]),
                   "val": list(val[self.unique])}]
        else:
            splits = StratifiedKFold(n_splits=n_folds)
            y = self._get_stratify(train)
            cv = []
            for train_index, val_index in splits.split(train, y):
                fold = {
                    "train": list(train.iloc[train_index][self.unique]),
                    "val": list(train.iloc[val_index][self.unique])
                }
                cv.append(fold)

        self._split_outro("random", train_test, cv)

    def scaffold_split(
        self,
        test_size: float = 0.2,
        n_folds: int = 5,
        val_size: Union[None, float] = None,
        seed: int = 51983
    ):
        """
        Splits data in scaffold mode mening mols are taken into account
        and similar mols will go to identical sets or folds

        Args:
            `test_size` (float): Fraction of data for test (default is `0.2`)
            `n_folds` (int): Number of CV folds.
            Can be `1` meaning validation set.
            If `1` `val_size` must be specified (default is `5`)
            `val_size` (float): Fraction of data for validation.
            Must be specified if `n_folds` == `1` (default is `None`)
            `seed` (int): Seed for splitting (default is `51983`)
        """
        params = locals()
        params.pop("self")
        self._split_intro("scaffold", **params)

        temp_sdf = "_temp.sdf"
        prop_to_sdf = set(self.df.columns) - {"ROMol", "_id"}
        PandasTools.WriteSDF(
            self.df,
            out=temp_sdf,
            idName="_id",
            properties=prop_to_sdf
        )

        featurizer = feat.CircularFingerprint(size=2048, radius=4)
        fp_splitter = splits.FingerprintSplitter()
        loader = data.SDFLoader(
            tasks=[self._get_target_col()],
            featurizer=featurizer,
            sanitize=True
        )
        dataset = loader.create_dataset(temp_sdf)

        if n_folds == 1:
            train, val, test = fp_splitter.split(
                dataset=dataset,
                frac_train=1 - test_size - val_size,
                frac_valid=val_size,
                frac_test=test_size,
                seed=seed
            )

            train_ids = list(self.df.iloc[train][self.unique])
            val_ids = list(self.df.iloc[val][self.unique])
            test_ids = list(self.df.iloc[test][self.unique])
            train_test = {"train": train_ids + val_ids, "test": test_ids}
            cv = [{"train": train_ids, "val": val_ids}]

        else:
            train_val, test_1, test_2 = fp_splitter.split(
                dataset=dataset,
                frac_train=1 - test_size,
                frac_valid=test_size,
                frac_test=0,
                seed=seed
            )
            test = test_1 + test_2

            train_val_ids = list(self.df.iloc[train_val][self.unique])
            test_ids = list(self.df.iloc[test][self.unique])
            train_test = {"train": train_val_ids, "test": test_ids}

            PandasTools.WriteSDF(
                self.df.iloc[train_val],
                out=temp_sdf,
                idName="_id",
                properties=prop_to_sdf
            )

            cv = []
            for i in tqdm(range(n_folds)):
                mols = PandasTools.LoadSDF(temp_sdf, idName="_id")
                dataset = loader.create_dataset(temp_sdf)

                if i == (n_folds - 1):
                    val = list(range(len(mols)))

                else:
                    val_size = 1 / (n_folds - i)
                    train, val_1, val_2 = fp_splitter.split(
                        dataset=dataset,
                        frac_train=1 - val_size,
                        frac_valid=val_size,
                        frac_test=0,
                        seed=seed
                    )
                    val = val_1 + val_2

                    PandasTools.WriteSDF(
                        df=mols.iloc[train],
                        out=temp_sdf,
                        idName="_id",
                        properties=prop_to_sdf
                    )

                val_ids = list(mols.iloc[val][self.unique])
                train_ids = list(set(train_val_ids) - set(val_ids))
                cv.append({"train": train_ids, "val": val_ids})

        os.remove(temp_sdf)
        self._split_outro("scaffold", train_test, cv)

    def split_info(
        self,
        split_type: str,
        radius: int = 3,
        n_bits: int = 2048
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets split's info and similarity as dataframes

        Args:
            `split_type` (str): Type of split
            `radius` (int): Radius for Morgan Fingerprints
            to compute similarity
            `n_bits` (int): Number of bits for Morgan Fingerprints
            to compute similarity

        Returns:
            tuple: tuple of dataframes (split's info and similarity)
        """
        self._checker(self.split[split_type], f"{split_type}_split")

        train_ids = self.split[split_type]["train_test"]["train"]
        test_ids = self.split[split_type]["train_test"]["test"]

        # For info df
        train_count, test_count = len(train_ids), len(test_ids)
        all_count = train_count + test_count
        counts = [all_count, train_count, test_count]
        percents = [1, train_count / all_count, test_count / all_count]
        info_indexes = ["All", "Train", "Test"]

        # For similarity df
        sim_kwargs = {"radius": radius, "n_bits": n_bits}
        print(f"[INFO] TanimotoSimilarity with MorganFingerprints "
              f"(radius = {radius} and n_bits = {n_bits}) will be used!")
        similarity = np.identity(2 + len(self.split[split_type]["cv"]))

        train_mols = self.df[self.df[self.unique].isin(train_ids)]["ROMol"]
        test_mols = self.df[self.df[self.unique].isin(test_ids)]["ROMol"]

        similarity[0][1] = compute_similarity(
            mols_1=list(train_mols),
            mols_2=list(test_mols),
            **sim_kwargs
        )
        similarity[1][0] = compute_similarity(
            mols_1=list(test_mols),
            mols_2=list(train_mols),
            **sim_kwargs
        )
        similarity_indexes = ["Train", "Test"]

        for i, fold in enumerate(self.split[split_type]["cv"]):
            train_ids, val_ids = fold["train"], fold["val"]

            # For info df
            train_count, val_count = len(fold["train"]), len(fold["val"])
            counts += [train_count, val_count]
            percents += [train_count / all_count, val_count / all_count]
            info_indexes += [f"Train_{i}", f"Val_{i}"]

            # For similarity df
            train_mols = self.df[self.df[self.unique].isin(train_ids)]["ROMol"]
            val_mols = self.df[self.df[self.unique].isin(val_ids)]["ROMol"]

            similarity[0][i+2] = compute_similarity(
                mols_1=list(train_mols),
                mols_2=list(val_mols),
                **sim_kwargs
            )
            similarity[i+2][0] = compute_similarity(
                mols_1=val_mols,
                mols_2=train_mols,
                **sim_kwargs
            )
            similarity[1][i+2] = compute_similarity(
                mols_1=test_mols,
                mols_2=val_mols,
                **sim_kwargs
            )
            similarity[i+2][1] = compute_similarity(
                mols_1=val_mols, mols_2=test_mols,
                **sim_kwargs
            )
            similarity_indexes += [f"Val_{i}"]

        info = pd.DataFrame(
            {"count": counts, "%": np.round(percents, 3) * 100},
            index=info_indexes
        )

        similarity = pd.DataFrame(
            similarity,
            columns=similarity_indexes,
            index=similarity_indexes
        )

        return info, similarity

    def print_split_info(self, split_type: str):
        """
        Prints split's info and similarity

        Args:
            `split_type` (str): Type of split
        """
        self._checker(self.split[split_type], f"{split_type}_split")
        print(self.split[split_type]["similarity"])
        print(self.split[split_type]["info"])

    def plot_split(self, split_type: str, target_col: str):
        """
        Plots split

        Args:
            `split_type` (str): Type of split
            `target_col` (str): Dataframe column to use for plotting.
            Pass 'target' to plot prepared targets
        """
        self._checker(self.split[split_type], f"{split_type}_split")
        plot_split_targets_hist(
            df=self.df,
            main_column=self.unique,
            target_column=target_col,
            train_test_split=self.split[split_type]["train_test"],
            cv_split=self.split[split_type]["cv"],
            split_type=split_type,
            plotly_args=self.plotly_args
        )

    @staticmethod
    def _get_stratify():
        pass

    @staticmethod
    def _get_target_col():
        return "target"

    # Get data

    def get_targets(self) -> Dict[str, float]:
        """
        Gets prepared targets

        Returns:
            dict: prepared targets
        """
        self._checker(self.targets, "prepare_targets")
        return self.targets

    def get_graphs(
        self
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Gets prepared graphs

        Returns:
            dict: prepared graphs
        """
        self._checker(self.graphs, "generate_graphs")
        return self.graphs

    def get_split(self, split_type: str) -> Dict[str, Any]:
        """
        Gets prepared split

        Args:
            `split_type` (str): Type of split

        Returns:
            dict: prepared split
        """
        self._checker(self.split[split_type], f"{split_type}_split")
        return self.split[split_type]

    def get_all(
        self,
        split_type: str
    ) -> Tuple[
        Dict[str, float],
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        Dict[str, Any]
    ]:
        """
        Gets prepared targets, graphs and split

        Args:
            `split_type` (str): Type of split

        Returns:
            tuple: tuple of dicts (targets, graphs and split)
        """
        targets = self.get_targets()
        graphs = self.get_graphs()
        split = self.get_split(split_type)
        return targets, graphs, split

    # Push data to S3

    def _push_file(
        self,
        data_version: str,
        local_path: str,
        upload_path: str,
        info: str,
        remove: bool
    ):
        # Add checking AWS credentials
        upload_path = f"data/{data_version}/{self.task}/" + upload_path
        S3.Bucket(BUCKET).upload_file(local_path, upload_path)
        print(f"[INFO] `{info}` was successfully uploaded to S3 "
              f"`{BUCKET}` bucket! S3 key: `{upload_path}`")
        if remove:
            os.remove(local_path)

    def push_sdf(self, data_version: str):
        """
        Pushes sdf file to S3

        Args:
            `data_version` (str): Data version (name of folder for S3)
        """
        self._push_file(
            data_version=data_version,
            local_path=self.sdf_path,
            upload_path="data.sdf",
            info=self.sdf_path,
            remove=False
        )

    def push_targets(self, data_version: str):
        """
        Pushes targets to S3 as json file

        Args:
            `data_version` (str): Data version (name of folder for S3)
        """
        save_json(self.get_targets(), "targets.json")
        self._push_file(
            data_version=data_version,
            local_path="targets.json",
            upload_path="targets.json",
            info="self.targets",
            remove=True
        )

    def push_graphs(self, data_version: str):
        """
        Pushes graphs to S3 as pkl file

        Args:
            `data_version` (str): Data version (name of folder for S3)
        """
        save_pkl(self.get_graphs(), "graphs.pkl")
        self._push_file(
            data_version=data_version,
            local_path="graphs.pkl",
            upload_path="graphs.pkl",
            info="self.graphs",
            remove=True
        )

    def push_split(self, split_type: str, data_version: str):
        """
        Pushes split to S3 as json file

        Args:
            `split_type` (str): Type of split
            `data_version` (str): Data version (name of folder for S3)
        """
        for key in ["train_test", "cv"]:
            save_json(self.get_split(split_type)[key], f"{key}.json")
            self._push_file(
                data_version=data_version,
                local_path=f"{key}.json",
                upload_path=f"split_{split_type}/{key}.json",
                info=f"self.split['{split_type}']['{key}']",
                remove=True
            )

        for key in ["info", "similarity"]:
            save_csv(self.get_split(split_type)[key], f"{key}.csv")
            self._push_file(
                data_version=data_version,
                local_path=f"{key}.csv",
                upload_path=f"split_{split_type}/{key}.csv",
                info=f"self.split['{split_type}']['{key}']",
                remove=True
            )

    def push_config(self, data_version: str):
        """
        Pushes config to S3 as json file

        Args:
            `data_version` (str): Data version (name of folder for S3)
        """
        save_json(self.config, "config.json")
        self._push_file(
            data_version=data_version,
            local_path="config.json",
            upload_path="config.json",
            info="self.config",
            remove=True
        )

    def push_all(self, data_version: str):
        """
        Pushes sdf file, targets, graphs, splits, config to S3

        Args:
            `data_version` (str): Data version (name of folder for S3)
        """
        self.push_sdf(data_version)
        self.push_targets(data_version)
        self.push_graphs(data_version)
        self.push_split("random", data_version)
        self.push_split("scaffold", data_version)
        self.push_config(data_version)
