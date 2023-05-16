from typing import List, Dict, Union, Callable, Any

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from .abstract import _TrainAnalyzer


class RegressionAnalyzer(_TrainAnalyzer):
    """
    A class used to analyze info about trained regression model.

    Attributes:
        `results` (dict): Dictionary with results per each stage
        after training. Keys are `cv`, `test` and `final`.
        Main keys for each stage are `losses`, `metrics`, `best_result`.
        `time`, `ids`, `true` and `pred`
        `postprocess_fn` (callable): Function that takes list with
        model outputs from `pred` key for each stage in `results`
        and somehow processes them.
        `plotly_args` (dict): Dict with args for plotly charts.

    Methods:
        `get_stages()`: Gets list of stages.
        `get_metrics()`: Gets list of metrics used in training.
        `get_folds()`: Gets number of folds used in training.
        `get_epochs(stage)`: Gets number of epochs for stage.
        `get_best_epoch(stage)`: Gets number of best epoch for stage.
        `get_time()`: Gets train time in seconds for all stages.
        `get_df_pred(stage)`: Gets dataframe with predictions.
        `get_df_metrics(stages)`: Gets dataframe with metrics.
        `get_metric_result(stage, metric, **kwargs)`: Gets result for metric.
        `plot_losses(stage, fold)`: Plots losses.
        `plot_metrics(stage, metrics, fold)`: Plots metrics.
        `plot_pred_sample(stage, id)`: Plots predictions over epochs
        for one unique id.
        `plot_pred(stage)`: Plots True-Predict dependency.
        `plot_hist(stage)`: Plots histogram for true and final predicted
        values.
        `plot_kde(stage)`: Plots kernel destiny estimation for true and
        final predicted values.
        `plot_residuals_hist(stage)`: Plots residuals histogram for true and
        final predicted values.
        `plot_all(stage)`: Plots main charts (losses, all metrics,
        true-predict dependency, histogram for true and final predicted values,
        kernel density estimation for true and final predicted values).
        `plot_pred_per_epoch(stage, epochs)`: Plots True-Predict dependency
        per epochs.
        `plot_hist_per_epoch(stage, epochs)`: Plots histograms for true and
        predicted values per epochs.
        `plot_kde_per_epoch(stage, epochs)`: Plots kernel destiny estimation
        for true and predicted values per epochs.
        `plot_residuals_hist_per_epoch(stage, epochs)`: Plots residuals
        histogram for true and predicted values per epochs.
        `set_plotly_args(**kwargs)`: Sets args for plotly charts.
    """

    def __init__(
        self,
        results: Dict[str, Dict[str, Any]],
        postprocess_fn: Union[None, Callable[[List], List]] = None
    ):
        """
        Args:
            `results` (dict): Dictionary with results per each stage
            after training. Keys are `cv`, `test` and `final` and
            values are dicts with results for each stage (which contain
            lossses for each epoch inside `losses`, dict with metrics
            for each epoch inside `metrics`, best epoch and best metrics
            inside `best_result`, training time inside `time`,
            list with unique ids inside `ids`, list with true values
            inside `true` and list with model outputs for each
            epoch inside `pred`).
            `postprocess_fn` (callable): Function that takes list with
            model outputs from `pred` key for each stage in `results`
            and somehow processes them (default is `None`).
        """
        super().__init__(results, postprocess_fn)

    def get_metric_result(
        self,
        stage: str,
        metric: Callable[[List[float], List[float]], float],
        **kwargs
    ) -> float:
        """
        Gets result for specific metric.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
            `metric` (callable): Function that takes `y_true` and `y_pred`
            in this order and gives float as output.
            `**kwargs`: Extra arguments for `metric` function.

        Returns:
            float: Metric's result.
        """
        return super().get_metric_result(stage, metric, False, **kwargs)

    def plot_pred_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots True-Predict dependency per epochs.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch).
        """

        df = self.get_df_pred(stage)

        best_epoch = self.get_best_epoch(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1

        subplot_titles = []
        for epoch in epochs:
            if epoch == best_epoch:
                subplot_titles.append(f"Best Epoch {epoch}")
            else:
                subplot_titles.append(f"Epoch {epoch}")

        min_ = df['True'].min()
        max_ = df['True'].max()

        fig = make_subplots(
            rows=rows,
            cols=cols,
            x_title="True Values",
            y_title="Predicted Values",
            subplot_titles=subplot_titles
        )

        for i, epoch in enumerate(epochs):
            pred = self._get_pred(stage, epoch)
            fig.add_trace(
                go.Scatter(
                    x=df["True"],
                    y=pred,
                    mode='markers',
                    text=df["ID"]
                ),
                row=i // cols + 1,
                col=i % cols + 1
            )
            fig.add_trace(
                go.Scatter(
                    x=np.linspace(min_, max_, num=100),
                    y=np.linspace(min_, max_, num=100),
                    mode='lines',
                    name='True = Pred',
                    line_color='Yellow'
                ),
                row=i // cols + 1,
                col=i % cols + 1
            )

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"True-Predict Dependency (stage: {stage})",
            showlegend=False
        )
        fig.show()

    def plot_hist_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots histograms for true and predicted values per epochs.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
            `epochs` (list): List with epochs for plotting.
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch).
        """
        df = self.get_df_pred(stage)

        best_epoch = self.get_best_epoch(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1

        subplot_titles = []
        for epoch in epochs:
            if epoch == best_epoch:
                subplot_titles.append(f"Best Epoch {epoch}")
            else:
                subplot_titles.append(f"Epoch {epoch}")

        fig = make_subplots(
            rows=rows,
            cols=cols,
            x_title="Target",
            y_title="Count",
            subplot_titles=subplot_titles
        )

        for i, epoch in enumerate(epochs):
            fig.add_trace(
                go.Histogram(x=df["True"], name=f"Epoch {epoch} True"),
                row=i // cols + 1,
                col=i % cols + 1
            )
            pred = self._get_pred(stage, epoch)
            fig.add_trace(
                go.Histogram(x=pred, name=f"Epoch {epoch} Pred"),
                row=i // cols + 1,
                col=i % cols + 1
            )

        fig.update_traces(opacity=0.75)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Target Histograms (stage: {stage})",
        )
        fig.show()

    def plot_kde_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots kernel density estimation for true and predicted values
        per epochs.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch).
        """
        df = self.get_df_pred(stage)
        hist_data = [df["True"]]
        group_labels = ["True"]

        best_epoch = self.get_best_epoch(stage)

        for epoch in epochs:
            pred = self._get_pred(stage, epoch)
            hist_data.append(pred)
            if epoch == best_epoch:
                group_labels.append(f"Best Epoch {epoch}")
            else:
                group_labels.append(f"Epoch {epoch}")

        fig = ff.create_distplot(
            hist_data=hist_data,
            group_labels=group_labels,
            show_rug=False,
            show_hist=False
        )

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Kernel Density Estimation (stage: {stage})",
            xaxis_title="Target",
            yaxis_title="Density"
        )
        fig.show()

    def plot_residuals_hist_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots residuals histogram for true and predicted values per epochs.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch).
        """
        df = self.get_df_pred(stage)

        best_epoch = self.get_best_epoch(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1

        subplot_titles = []
        for epoch in epochs:
            if epoch == best_epoch:
                subplot_titles.append(f"Best Epoch {epoch}")
            else:
                subplot_titles.append(f"Epoch {epoch}")

        fig = make_subplots(
            rows=rows,
            cols=cols,
            x_title="Residual",
            y_title="Count",
            subplot_titles=subplot_titles
        )

        for i, epoch in enumerate(epochs):
            pred = self._get_pred(stage, epoch)
            residuals = df["True"].values - np.array(pred)
            fig.add_trace(
                go.Histogram(x=residuals, name=f"Epoch {epoch}"),
                row=i // cols + 1,
                col=i % cols + 1
            )

        # fig.update_traces(nbinsx=100)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Residuals Histograms (stage: {stage})",
        )
        fig.show()

    def plot_pred(self, stage: str):
        """
        Plots True-Predict dependency.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        best_epoch = self.get_best_epoch(stage)
        self.plot_pred_per_epoch(stage, [best_epoch])

    def plot_hist(self, stage: str):
        """
        Plots histograms for true and final predicted values.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        best_epoch = self.get_best_epoch(stage)
        self.plot_hist_per_epoch(stage, [best_epoch])

    def plot_kde(self, stage: str):
        """
        Plots kernel density estimation for true and final predicted values.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        best_epoch = self.get_best_epoch(stage)
        self.plot_kde_per_epoch(stage, [best_epoch])

    def plot_residuals_hist(self, stage: str):
        """
        Plots residuals histogram for true and final predicted values.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        best_epoch = self.get_best_epoch(stage)
        self.plot_residuals_hist_per_epoch(stage, [best_epoch])

    def plot_all(self, stage):
        """
        Plots main charts (losses, all metrics, true-predict dependency,
        histogram for true and final predicted values,
        kernel density estimation for true and final predicted values).

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        self.plot_losses(stage)
        self.plot_metrics(stage, self.get_metrics())
        self.plot_pred(stage)
        self.plot_hist(stage)
        self.plot_kde(stage)
        self.plot_residuals_hist(stage)
