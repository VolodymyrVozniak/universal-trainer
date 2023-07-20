from collections import defaultdict
from typing import List, Dict, Union, Callable, Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ..base import _Base


class _TrainAnalyzer(_Base):
    def __init__(
        self,
        results: Dict[str, Dict[str, Any]],
        postprocess_fn: Union[None, Callable[[List], List]]
    ):
        self.results = results
        self.postprocess_fn = postprocess_fn
        self._include_epochs_pred = True
        self.set_plotly_args(font_size=14, template="plotly_dark", bargap=0.2)
        # self.set_plotly_args(font_size=14, barmode="overlay", bargap=0.2)

    def _check_stage(self, stage: str):
        if stage not in self.get_stages():
            raise ValueError(f"`stage` must be in {self.get_stages()}!")

    def _check_metric(self, metric: str):
        if metric not in self.get_metrics() + ["loss"]:
            raise ValueError(f"`metric` must be in {self.get_metrics()}!")

    def _get_pred(self, stage: str, epoch: int) -> List[float]:
        if self.get_epochs(stage) == len(self.results[stage]["ids"]):
            if self._include_epochs_pred:
                print("[WARNING] In case of any errors set "
                      "`self._include_epochs_pred` attribute to `False` "
                      "if you have predictions only for the best epoch.")

        if (len(self.results[stage]["pred"]) == self.get_epochs(stage)) \
                and self._include_epochs_pred:
            try:
                pred = self.results[stage]["pred"][epoch]
            except IndexError:
                raise ValueError(f"There is no `{epoch}` epoch in "
                                 f"`{stage}` stage!")
        else:
            pred = self.results[stage]["pred"]

        if self.postprocess_fn:
            pred = self.postprocess_fn(pred)

        return pred

    def _plot(self, stage: str, metric: str, fold: Union[None, int] = None):
        self._check_stage(stage)
        self._check_metric(metric)

        if stage == "cv":
            if self.get_folds() == 1 and fold is not None:
                print("[WARNING] Can't use specific fold when there is only "
                      f"1 fold! Just `{stage}` scores will be used!")
                fold = None
            stages = ["train", "val"]
            results = self.results[stage] if fold is None \
                else self.results[stage]['results_per_fold'][fold]
        elif stage in ["test", "final"]:
            stages = ["train", "test"]
            results = self.results[stage]
            if fold is not None:
                print("[WARNING] Can't use specific fold when `stage` is "
                      f"`{stage}`! Just `{stage}` scores will be used!")
                fold = None

        tuples_list = []

        def process_task_stage(epoch):
            if metric == 'loss':
                value = results["losses"][stage_][epoch]
            else:
                value = results["metrics"][stage_][epoch][metric]
            return epoch, stage_, value

        epochs = len(results["losses"]["train"])
        for stage_ in stages:
            tuples_list += list(map(process_task_stage, range(epochs)))

        df = pd.DataFrame(tuples_list, columns=["epoch", "stage", "value"])

        metric = metric.capitalize()
        title = f"{metric} (stage: {stage}; fold: {fold})" \
            if fold is not None else f"{metric} (stage: {stage})"

        fig = px.line(df, x='epoch', y='value',
                      color='stage', markers=True)
        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=title,
            xaxis_title="Epoch",
            yaxis_title=metric
        )
        fig.show()

    def get_stages(self) -> List[str]:
        """
        Gets list of stages.

        Returns:
            list: List of stages.
        """
        return list(self.results.keys())

    def get_metrics(self) -> List[str]:
        """
        Gets list of metrics used in training.

        Returns:
            list: List of metrics.
        """
        return list(self.results["test"]["best_result"]["metrics"].keys())

    def get_folds(self) -> int:
        """
        Gets number of folds used in training.

        Returns:
            int: Number of folds.
        """
        cv_results = self.results["cv"]
        try:
            return len(cv_results["results_per_fold"])
        except KeyError:
            return 1

    def get_epochs(self, stage: str) -> int:
        """
        Gets number of epochs for stage.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.

        Returns:
            int: Number of epochs for stage.
        """
        self._check_stage(stage)
        try:
            return len(self.results[stage]["losses"]["train"])
        except KeyError:
            return 1

    def get_best_epoch(self, stage: str) -> int:
        """
        Gets number of best epoch for stage.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.

        Returns:
            int: Number of best epoch for stage.
        """
        self._check_stage(stage)
        return self.results[stage]["best_result"]["epoch"]

    def get_time(self) -> Dict[str, float]:
        """
        Gets train time in seconds for all stages.

        Returns:
            dict: Train time for all stages.
        """
        return {stage: self.results[stage]["time"]
                for stage in self.get_stages()}

    def get_df_pred(self, stage: str) -> pd.DataFrame:
        """
        Gets dataframe with predictions.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.

        Returns:
            pd.DataFrame: Columns: `['ID', 'True', 'Pred']`.
        """
        self._check_stage(stage)

        ids = self.results[stage]["ids"]
        true = self.results[stage]["true"]
        best_epoch = self.get_best_epoch(stage)
        pred = self._get_pred(stage, best_epoch)

        return pd.DataFrame({"ID": ids, "True": true, "Pred": pred})

    def plot_losses(
        self,
        stage: str,
        fold: Union[None, int] = None
    ):
        """
        Plots losses.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
            `fold` (int):
                Number of CV fold.
                Specify this parameter only when `stage` == `'cv'`.
                If not specified and `stage` == `'cv'` plots mean results
                for all CV folds. Default is `None`.
        """
        self._plot(stage, "loss", fold)

    def plot_metrics(
        self,
        stage: str,
        metrics: List[str],
        fold: Union[None, int] = None
    ):
        """
        Plots metrics.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
            `metrics` (list):
                List of metrics for plotting.
            `fold` (int):
                Number of CV fold.
                Specify this parameter only when `stage` == `'cv'`.
                If not specified and `stage` == `'cv'`, plots mean results
                for all CV folds. Default is `None`.
        """
        for metric in metrics:
            self._plot(stage, metric, fold)

    def plot_pred_sample(self, stage: str, id: Union[int, str]):
        """
        Plots predictions over epochs for specific entry.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
            `id` (str):
                Unique id to identify specific entry.
        """
        self._check_stage(stage)

        try:
            index = self.results[stage]['ids'].index(id)
        except ValueError:
            raise ValueError(f"There is no `{id}` id in `{stage}` stage!")
        true = self.results[stage]['true'][index]
        pred = self.results[stage]['pred']

        try:
            sample_pred = list(map(lambda x: x[index], pred))
        except (TypeError, IndexError):
            raise ValueError("`plot_pred_sample()` method is not available "
                             "when you have predictions only for the "
                             "best epoch!")
        if self.postprocess_fn:
            sample_pred = self.postprocess_fn(sample_pred)
        epochs = len(sample_pred)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(epochs)),
            y=sample_pred,
            mode='lines+markers',
            name='Pred'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(epochs)),
            y=[true for i in range(epochs)],
            mode='lines+markers',
            name='True',
            line_color='Yellow'
        ))

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Sample Prediction (stage: {stage}; ID: {id})",
            xaxis_title="Epoch",
            yaxis_title="Sample Prediction"
        )
        fig.show()

    def get_df_metrics(
        self,
        stages: List[str] = ["cv", "test"]
    ) -> pd.DataFrame:
        """
        Gets result dataframe with metrics.

        Args:
            `stages` (list):
                List of stages for final dataframe.
                Default is `["cv", "test"]`.

        Returns:
            pd.DataFrame:
                Dataframe with metrics.
        """
        for stage in stages:
            self._check_stage(stage)

        metrics = defaultdict(dict)
        for stage in stages:
            metrics[stage] = self.results[stage]["best_result"]["metrics"]

        for_df, index_values = [], []
        for metric in self.get_metrics():
            for stage, stage_metrics in metrics.items():
                for_df.append(stage_metrics[metric])
                index_values.append((metric, stage))

        columns = pd.MultiIndex.from_tuples(
            tuples=index_values,
            names=["metric", "stage"]
        )
        return pd.DataFrame([for_df], columns=columns)

    def get_metric_result(
        self,
        stage: str,
        metric: Callable[[List[float], List[float]], float],
        round: bool = True,
        **kwargs
    ) -> float:
        """
        Gets result for specific metric.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
            `metric` (callable):
                Function that takes `y_true` and `y_pred`
                in this order and gives float as output.
            `round` (bool):
                Flag to work with binary values (if `True`)
                or predictions (if `False`). Default is `True`.
            `**kwargs`:
                Extra arguments for `metric` function.

        Returns:
            float: Metric's result.
        """
        df = self.get_df_pred(stage)
        pred = np.round(df["Pred"]) if round else df["Pred"]
        return metric(df["True"], pred, **kwargs)
