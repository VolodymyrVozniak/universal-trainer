from typing import List, Dict, Union, Callable, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score, auc, \
    precision_recall_curve

from .classification import _ClassificationAnalyzer


class BinaryAnalyzer(_ClassificationAnalyzer):
    """
    A class used to analyze info about trained binary model.

    Attributes:
        `results` (dict): Dictionary with results per each stage
        after training. Keys are `cv`, `test` and `final`.
        Main keys for each stage are `losses`, `metrics`, `best_result`
        `time`, `ids`, `true` and `pred`.
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
        `get_metric_result(stage, metric, round, **kwargs)`: Gets result
        for metric.
        `print_classification_report(stage, digits)`: Prints classification
        report.
        `plot_losses(stage, fold)`: Plots losses.
        `plot_metrics(stage, metrics, fold)`: Plots metrics.
        `plot_pred_sample(stage, id)`: Plots predictions over epochs
        for one unique id.
        `plot_confusion_matrix(stage)`: Plots confusion matrix.
        `plot_pred_hist(stage)`: Plots prediction histogram for best epoch.
        `plot_roc_auc(stage)`: Plots ROC-AUC chart.
        `plot_precision_recall_auc(stage)`: Plots Precision-Recall AUC chart.
        `plot_enrichment(stage)`: Plots enrichment chart.
        `plot_all(stage)`: Plots main charts (losses, all metrics,
        confusion matrix, prediction histogram for best epoch,
        ROC-AUC curve, Precision-Recall AUC curve, enrichment).
        `plot_confusion_matrix_per_epoch(stage, epochs)`: Plots confusion
        matrix per epochs.
        `plot_pred_hist_per_epoch(stage, epochs)`: Plots prediction histograms
        per epochs.
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
            and somehow processes them. For binary problem it is important
            to have probability of belonging to class 1 as final output.
            So, for example, if you have logits as your model output, define
            function that will convert your logits into probabilities
            (simple sigmoid function). If you have probabilities as your
            model output, keep this argument `None` and use default model
            outputs (default is `None`).
        """
        super().__init__(results, postprocess_fn)

    def plot_pred_hist_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots prediction histograms per epochs.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
            `epochs` (list): List with epochs for plotting.
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch).
        """
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
            x_title="Prediction",
            y_title="Count",
            subplot_titles=subplot_titles
        )

        for i, epoch in enumerate(epochs):
            try:
                pred = self.results[stage]['pred'][epoch]
                if self.postprocess_fn:
                    pred = self.postprocess_fn(pred)
            except IndexError:
                raise ValueError(f"There is no `{epoch}` epoch in "
                                 f"`{stage}` stage!")
            fig.add_trace(
                go.Histogram(x=pred, name=f"Epoch {epoch} Pred"),
                row=i // cols + 1,
                col=i % cols + 1
            )

        fig.update_layout(
            **self.plotly_args,
            title_text=f"Prediction Histograms (stage: {stage})",
        )
        fig.show()

    def plot_pred_hist(self, stage: str):
        """
        Plots prediction histogram for best epoch.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        best_epoch = self.get_best_epoch(stage)
        self.plot_pred_hist_per_epoch(stage, [best_epoch])

    def plot_roc_auc(self, stage: str):
        """
        Plots ROC-AUC chart.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        df = self.get_df_pred(stage)

        fpr, tpr, _ = roc_curve(df["True"], df["Pred"])
        rocauc = roc_auc_score(df["True"], df["Pred"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name="AUC=" + str(round(rocauc, 3)),
            mode='lines'
        ))

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"ROC-AUC (stage: {stage})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True
        )
        fig.show()

    def plot_precision_recall_auc(self, stage: str):
        """
        Plots Precision-Recall AUC chart.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        df = self.get_df_pred(stage)

        precision, recall, thresholds \
            = precision_recall_curve(df["True"],  df["Pred"])
        prauc = float(auc(recall, precision))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            name="AUC=" + str(round(prauc, 3)),
            mode='lines'
        ))

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Precision-Recall AUC (stage: {stage})",
            xaxis_title="Recall",
            yaxis_title="Precision",
            showlegend=True
        )
        fig.show()

    def plot_enrichment(self, stage: str):
        """
        Plots enrichment chart.

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        df = self.get_df_pred(stage)
        df = df.sort_values("Pred", ascending=False)

        n_pos = df["True"].value_counts()[1]
        n_all = df.shape[0]
        pairs = np.zeros((n_all, 3), dtype=np.float32)
        pos = 0
        for i in range(n_all):
            pos += df.iloc[i, :]["True"]
            pairs[i, :] = [(i+1) / n_all, pos / n_pos, (i+1) / n_pos]

        dfr = pd.DataFrame(pairs, columns=["frac_all", "frac_pos", "ideal"])
        dfr.loc[dfr["ideal"] > 1] = 1

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dfr["frac_all"],
            y=dfr["frac_pos"],
            name="Trained Model",
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=dfr["frac_all"],
            y=dfr["ideal"],
            name="Ideal Model",
            line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=dfr["frac_all"],
            y=dfr["frac_all"],
            name="Random Model",
            line=dict(dash="dash")
        ))

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Enrichment (stage: {stage})",
            xaxis_title="Fraction of samples",
            yaxis_title="Fraction of hits"
        )
        fig.show()

    def plot_all(self, stage):
        """
        Plots main charts (losses, all metrics, confusion matrix,
        prediction histogram for best epoch, ROC-AUC curve,
        Precision-Recall AUC curve, enrichment).

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        self.plot_losses(stage)
        self.plot_metrics(stage, self.get_metrics())
        self.plot_confusion_matrix(stage)
        self.plot_pred_hist(stage)
        self.plot_roc_auc(stage)
        self.plot_precision_recall_auc(stage)
        self.plot_enrichment(stage)
