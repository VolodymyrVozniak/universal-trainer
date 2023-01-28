from collections import defaultdict
from typing import List, Dict, Union, Callable, Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score, auc, \
    precision_recall_curve, confusion_matrix

from ..base import _Base


class _TrainAnalyzer(_Base):
    def __init__(
        self,
        results: Dict[str, Dict[str, Any]],
        postprocess_fn: Union[None, Callable[[List], List]]
    ):
        self.results = results
        self.postprocess_fn = postprocess_fn
        self.set_plotly_args(font_size=14, template="plotly_dark", bargap=0.2)
        # self.set_plotly_args(font_size=14, barmode="overlay", bargap=0.2)

    def _check_stage(self, stage: str):
        if stage not in self.get_stages():
            raise ValueError(f"`stage` must be in {self.get_stages()}!")

    def _check_metric(self, metric: str):
        if metric not in self.get_metrics() + ["loss"]:
            raise ValueError(f"`metric` must be in {self.get_metrics()}!")

    def _plot(self, stage: str, metric: str, fold: Union[None, int] = None):
        self._check_stage(stage)
        self._check_metric(metric)

        if stage == "cv":
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

    @staticmethod
    def get_stages() -> List[str]:
        """
        Gets list of stages

        Returns:
            list: list of stages
        """
        return ["cv", "test", "final"]

    def get_metrics(self) -> List[str]:
        """
        Gets list of metrics used in training

        Returns:
            list: list of metrics
        """
        return list(self.results["test"]["metrics"]["train"][0].keys())

    def get_folds(self) -> int:
        """
        Gets number of folds used in training

        Returns:
            int: number of folds
        """
        return len(self.results["cv"]["results_per_fold"])

    def get_epochs(self, stage) -> int:
        """
        Gets number of epochs for stage

        Args:
            `stage` (str): One of stage from `get_stages()` method

        Returns:
            int: number of epochs for stage
        """
        self._check_stage(stage)
        return len(self.results[stage]["losses"]["train"])

    def get_time(self) -> Dict[str, float]:
        """
        Gets train time in seconds for all stages

        Returns:
            dict: train time for all stages
        """
        return {stage: self.results[stage]["time"]
                for stage in self.get_stages()}

    def get_df_pred(self, stage: str) -> pd.DataFrame:
        """
        Gets dataframe with predictions

        Args:
            `stage` (str): One of stage from `get_stages()` method

        Returns:
            pd.DataFrame: columns: `['ID', 'True', 'Pred']`
        """
        self._check_stage(stage)

        ids = self.results[stage]["ids"]
        true = self.results[stage]["true"]
        best_epoch = self.results[stage]["best_result"]["epoch"]
        pred = self.results[stage]["pred"][best_epoch]

        if self.postprocess_fn:
            pred = self.postprocess_fn(pred)

        return pd.DataFrame({"ID": ids, "True": true, "Pred": pred})

    def plot_losses(
        self,
        stage: str,
        fold: Union[None, int] = None
    ):
        """
        Plots losses

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `fold` (int): Number of CV fold.
            Specify this parameter only when `stage` == `'cv'`.
            If not specified and `stage` == `'cv'` plots mean results
            for all CV folds (default is `None`)
        """
        self._plot(stage, "loss", fold)

    def plot_metrics(
        self,
        stage: str,
        metrics: List[str],
        fold: Union[None, int] = None
    ):
        """
        Plots metrics

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `metrics` (list): List of metrics for plotting
            `fold` (int): Number of CV fold.
            Specify this parameter only when `stage` == `'cv'`.
            If not specified and `stage` == `'cv'`, plots mean results
            for all CV folds (default is `None`)
        """
        for metric in metrics:
            self._plot(stage, metric, fold)

    def plot_pred_sample(self, stage: str, id: Union[int, str]):
        """
        Plots predictions over epochs for specific entry

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `id` (str): Unique id to identify specific entry
        """
        self._check_stage(stage)

        try:
            index = self.results[stage]['ids'].index(id)
        except ValueError:
            raise ValueError(f"There is no `{id}` id in `{stage}` stage!")
        true = self.results[stage]['true'][index]
        pred = self.results[stage]['pred']

        sample_pred = list(map(lambda x: x[index], pred))
        if self.postprocess_fn:
            sample_pred = self.postprocess_fn(sample_pred)
        epochs = len(sample_pred)

        df = pd.DataFrame({'epoch': range(epochs),
                           'sample_pred': sample_pred})
        df["stage"] = stage

        fig = px.line(df, x='epoch', y='sample_pred',
                      color='stage', markers=True)

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
        Gets result dataframe with metrics

        Args:
            `stages` (list): List of stages for final dataframe
            (default is `["cv", "test"]`)

        Returns:
            pd.DataFrame: dataframe with metrics
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


class BinaryAnalyzer(_TrainAnalyzer):
    """
    A class used to analyze info about trained binary model

    Attributes:
        `results` (dict): Dictionary with results per each stage
        after training. Keys are `cv`, `test` and `final`.
        Main keys for each stage are `losses`, `metrics`, `best_result`
        `time`, `ids`, `true` and `pred`
        `postprocess_fn` (callable): Function that takes list with
        model outputs from `pred` key for each stage in `results`
        and somehow processes them.

    Methods:
        `get_stages()`: Gets list of stages
        `get_metrics()`: Gets list of metrics used in training
        `get_folds()`: Gets number of folds used in training
        `get_epochs(stage)`: Gets number of epochs for stage
        `get_time()`: Gets train time in seconds for all stages
        `get_df_pred(stage)`: Gets dataframe with predictions
        `get_df_metrics(stages)`: Gets dataframe with metrics
        `get_metric_result(stage, metric, round, **kwargs)`: Gets result
        for metric
        `plot_losses(stage, fold)`: Plots losses
        `plot_metrics(stage, metrics, fold)`: Plots metrics
        `plot_pred_sample(stage, id)`: Plots predictions over epochs
        for one unique id
        `plot_confusion_matrix(stage)`: Plots confusion matrix
        `plot_confusion_matrix_per_epoch(stage, epochs)`: Plots confusion
        matrix per epochs
        `plot_roc_auc(stage)`: Plots ROC-AUC chart
        `plot_precision_recall_auc(stage)`: Plots Precision-Recall AUC chart
        `plot_enrichment(stage)`: Plots enrichment chart
        `plot_all(stage)`: Plots main charts (losses, all metrics,
        confusion matrix, ROC-AUC, Precision-Recall AUC, enrichment)
        `set_plotly_args(**kwargs)`: Sets args for plotly charts
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
            epoch inside `pred`)
            `postprocess_fn` (callable): Function that takes list with
            model outputs from `pred` key for each stage in `results`
            and somehow processes them. For binary problem it is important
            to have probability of belonging to class 1 as final output.
            So, for example, if you have logits as your model output, define
            function that will convert your logits into probabilities
            (simple sigmoid function). If you have probabilities as your
            model output, keep this argument `None` and use default model
            outputs (default is `None`)
        """
        super().__init__(results, postprocess_fn)

    def get_metric_result(
        self,
        stage: str,
        metric: Callable[[List[float], List[float]], float],
        round: bool = True,
        **kwargs
    ):
        """
        Gets result for specific metric

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `metric` (callable): Function that takes `y_true` and `y_pred`
            in this order and gives float as output
            `round` (bool): Flag to work with binary values (if `True`)
            or predictions (if `False`) (default is `True`)
            `**kwargs`: extra arguments for `metric` function
        """
        df = self.get_df_pred(stage)
        pred = np.round(df["Pred"]) if round else df["Pred"]
        return metric(df["True"], pred, **kwargs)

    def plot_confusion_matrix(self, stage: str):
        """
        Plots confusion matrix

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        df = self.get_df_pred(stage)
        df["Pred_"] = df["Pred"].round()

        cm = confusion_matrix(df["True"], df["Pred_"])

        axes = ["Class 0", "Class 1"]
        text = [[f"{cm[0][0]} (TN)", f"{cm[0][1]} (FP)"],
                [f"{cm[1][0]} (FN)", f"{cm[1][1]} (TP)"]]

        fig = ff.create_annotated_heatmap(
            cm,
            x=axes,
            y=axes,
            colorscale='Viridis',
            annotation_text=text
        )

        fig.update_layout(
            **self.plotly_args,
            title_text=f'Confusion Matrix (stage: {stage})'
        )
        fig.update_yaxes(categoryorder='array', categoryarray=axes[::-1])
        # fig['data'][0]['showscale'] = True
        fig.show()

    def plot_confusion_matrix_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots confusion matrix per epochs

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch)
        """
        df = self.get_df_pred(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1
        axes = ["Class 0", "Class 1"]
        annotations = []

        subplot_titles = []
        for epoch in epochs:
            subplot_titles.append(f"Epoch {epoch}")

        fig = make_subplots(
            rows=rows,
            cols=cols,
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
            cm = confusion_matrix(df["True"], pred.round())
            text = [[f"{cm[0][0]} (TN)", f"{cm[0][1]} (FP)"],
                    [f"{cm[1][0]} (FN)", f"{cm[1][1]} (TP)"]]
            fig_heatmap = ff.create_annotated_heatmap(
                cm,
                x=axes,
                y=axes,
                colorscale='Viridis',
                annotation_text=text
            )
            fig.add_trace(
                fig_heatmap.data[0],
                row=i // cols + 1,
                col=i % cols + 1
            )
            annotation_epoch = list(fig_heatmap.layout.annotations)
            for k in range(len(annotation_epoch)):
                if i > 0:
                    annotation_epoch[k]['xref'] = f'x{i+1}'
                    annotation_epoch[k]['yref'] = f'y{i+1}'
            annotations += annotation_epoch

        for annotation in annotations:
            fig.add_annotation(annotation)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Confusion Matrix Per Epoch (stage: {stage})"
        )
        fig.update_yaxes(categoryorder='array', categoryarray=axes[::-1])
        fig.show()

    def plot_roc_auc(self, stage: str):
        """
        Plots ROC-AUC chart

        Args:
            `stage` (str): One of stage from `get_stages()` method
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
        Plots Precision-Recall AUC chart

        Args:
            `stage` (str): One of stage from `get_stages()` method
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
        Plots enrichment chart

        Args:
            `stage` (str): One of stage from `get_stages()` method
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
        ROC-AUC, Precision-Recall AUC, enrichment)

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        self.plot_losses(stage)
        self.plot_metrics(stage, self.get_metrics())
        self.plot_confusion_matrix(stage)
        self.plot_roc_auc(stage)
        self.plot_precision_recall_auc(stage)
        self.plot_enrichment(stage)


class RegressionAnalyzer(_TrainAnalyzer):
    """
    A class used to analyze info about trained regression model

    Attributes:
        `results` (dict): Dictionary with results per each stage
        after training. Keys are `cv`, `test` and `final`.
        Main keys for each stage are `losses`, `metrics`, `best_result`
        `time`, `ids`, `true` and `pred`
        `postprocess_fn` (callable): Function that takes list with
        model outputs from `pred` key for each stage in `results`
        and somehow processes them.

    Methods:
        `get_stages()`: Gets list of stages
        `get_metrics()`: Gets list of metrics used in training
        `get_folds()`: Gets number of folds used in training
        `get_epochs(stage)`: Gets number of epochs for stage
        `get_time()`: Gets train time in seconds for all stages
        `get_df_pred(stage)`: Gets dataframe with predictions
        `get_df_metrics(stages)`: Gets dataframe with metrics
        `get_metric_result(stage, metric, **kwargs)`: Gets result for metric
        `plot_losses(stage, fold)`: Plots losses
        `plot_metrics(stage, metrics, fold)`: Plots metrics
        `plot_pred_sample(stage, id)`: Plots predictions over epochs
        for one unique id
        `plot_pred(stage)`: Plots True-Predict dependency
        `plot_hist(stage)`: Plots histogram for true and final predicted values
        `plot_kde(stage)`: Plots kernel destiny estimation for true and
        final predicted values
        `plot_hist_per_epoch(stage, epochs)`: Plots histograms for true and
        predicted values per epochs
        `plot_kde_per_epoch(stage, epochs)`: Plots kernel destiny estimation
        for true, final predicted values and predicted values per epochs
        `plot_all(stage)`: Plots main charts (losses, all metrics,
        true-predict dependency, histogram for true and final predicted values,
        kernel density estimation for true and final predicted values)
        `set_plotly_args(**kwargs)`: Sets args for plotly charts
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
            epoch inside `pred`)
            `postprocess_fn` (callable): Function that takes list with
            model outputs from `pred` key for each stage in `results`
            and somehow processes them.
        """
        super().__init__(results, postprocess_fn)

    def get_metric_result(
        self,
        stage: str,
        metric: Callable[[List[float], List[float]], float],
        **kwargs
    ):
        """
        Gets result for specific metric

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `metric` (callable): Function that takes `y_true` and `y_pred`
            in this order and gives float as output
            `**kwargs`: extra arguments for `metric` function
        """
        df = self.get_df_pred(stage)
        return metric(df["True"], df["Pred"], **kwargs)

    def plot_pred(self, stage: str):
        """
        Plots True-Predict dependency

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        df = self.get_df_pred(stage)

        fig = px.scatter(df, x='True', y='Pred', hover_data=["ID"])

        min_ = df['True'].min()
        max_ = df['True'].max()
        fig.add_trace(go.Scatter(
            x=np.linspace(min_, max_, num=100),
            y=np.linspace(min_, max_, num=100),
            mode='lines',
            name='True = Pred',
            line_color='Yellow'
        ))
        # fig.add_shape(type='line', x0=min_, y0=min_, x1=max_, y1=max_,
        #               line=dict(color='Yellow', width=3))

        fig.update_traces(line_width=3, marker_size=7)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"True-Predict Dependency (stage: {stage})",
            xaxis_title="True Values",
            yaxis_title="Predicted Values"
        )
        fig.show()

    def plot_hist(self, stage: str):
        """
        Plots histograms for true and final predicted values

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        df = self.get_df_pred(stage)

        fig = go.Figure()
        for el in ["True", "Pred"]:
            fig.add_trace(go.Histogram(x=df[el], name=el))

        fig.update_traces(opacity=0.75)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Target Histograms (stage: {stage})",
            xaxis_title="Target",
            yaxis_title="Count"
        )
        fig.show()

    def plot_kde(self, stage: str):
        """
        Plots kernel density estimation for true and predicted values

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        df = self.get_df_pred(stage)

        fig = ff.create_distplot(
            hist_data=[df["True"], df["Pred"]],
            group_labels=["True", "Pred"],
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

    def plot_hist_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots histograms for true and predicted values per epoch

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch)
        """
        df = self.get_df_pred(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1

        subplot_titles = []
        for epoch in epochs:
            subplot_titles.append(f"Epoch {epoch}")

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles
        )

        for i, epoch in enumerate(epochs):
            fig.add_trace(
                go.Histogram(x=df["True"], name=f"Epoch {epoch} True"),
                row=i // cols + 1,
                col=i % cols + 1
            )
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

        fig.update_traces(opacity=0.75)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Target Histograms Per Epoch (stage: {stage})",
            xaxis_title="Target",
            yaxis_title="Count"
        )
        fig.show()

    def plot_kde_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots kernel density estimation for true, final predicted
        values and predicted values per epoch

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch)
        """
        df = self.get_df_pred(stage)
        hist_data = [df["True"], df["Pred"]]
        group_labels = ["True", "Pred"]

        for epoch in epochs:
            try:
                pred = self.results[stage]['pred'][epoch]
                if self.postprocess_fn:
                    pred = self.postprocess_fn(pred)
            except IndexError:
                raise ValueError(f"There is no `{epoch}` epoch in "
                                 f"`{stage}` stage!")
            hist_data.append(pred)
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
            title_text=f"Kernel Density Estimation Per Epoch (stage: {stage})",
            xaxis_title="Target",
            yaxis_title="Density"
        )
        fig.show()

    def plot_all(self, stage):
        """
        Plots main charts (losses, all metrics, true-predict dependency,
        histogram for true and final predicted values,
        kernel density estimation for true and final predicted values)

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        self.plot_losses(stage)
        self.plot_metrics(stage, self.get_metrics())
        self.plot_pred(stage)
        self.plot_hist(stage)
        self.plot_kde(stage)


class MulticlassAnalyzer(_TrainAnalyzer):
    """
    A class used to analyze info about trained multiclassification model

    Attributes:
        `results` (dict): Dictionary with results per each stage
        after training. Keys are `cv`, `test` and `final`.
        Main keys for each stage are `losses`, `metrics`, `best_result`
        `time`, `ids`, `true` and `pred`
        `postprocess_fn` (callable): Function that takes list with
        model outputs from `pred` key for each stage in `results`
        and somehow processes them.

    Methods:
        `get_stages()`: Gets list of stages
        `get_metrics()`: Gets list of metrics used in training
        `get_folds()`: Gets number of folds used in training
        `get_epochs(stage)`: Gets number of epochs for stage
        `get_time()`: Gets train time in seconds for all stages
        `get_df_pred(stage)`: Gets dataframe with predictions
        `get_df_metrics(stages)`: Gets dataframe with metrics
        `get_metric_result(stage, metric, **kwargs)`: Gets result for metric
        `plot_losses(stage, fold)`: Plots losses
        `plot_metrics(stage, metrics, fold)`: Plots metrics
        `plot_pred_sample(stage, id)`: Plots predictions over epochs
        for one unique id
        `plot_confusion_matrix(stage)`: Plots confusion matrix
        `plot_confusion_matrix_per_epoch(stage, epochs)`: Plots confusion
        matrix per epochs
        `plot_all(stage)`: Plots main charts (losses, all metrics,
        confusion matrix)
        `set_plotly_args(**kwargs)`: Sets args for plotly charts
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
            epoch inside `pred`)
            `postprocess_fn` (callable): Function that takes list with
            model outputs from `pred` key for each stage in `results`
            and somehow processes them. For multiclassification problem
            it is important to have exact class as final output. So,
            for example, if you have list of logits as your model output,
            define function that will convert your logits into belonging
            to some class (just maximum of these logits)
            (default is `None`)
        """
        super().__init__(results, postprocess_fn)

    def get_metric_result(
        self,
        stage: str,
        metric: Callable[[List[float], List[float]], float],
        **kwargs
    ):
        """
        Gets result for specific metric

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `metric` (callable): Function that takes `y_true` and `y_pred`
            in this order and gives float as output
            `**kwargs`: extra arguments for `metric` function
        """
        df = self.get_df_pred(stage)
        return metric(df["True"], df["Pred"], **kwargs)

    def plot_confusion_matrix(self, stage: str):
        """
        Plots confusion matrix

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        df = self.get_df_pred(stage)
        df["Pred_"] = df["Pred"].round()

        cm = confusion_matrix(df["True"], df["Pred_"])
        axes = [f"Class {i}" for i in range(len(cm))]

        fig = ff.create_annotated_heatmap(
            cm,
            x=axes,
            y=axes,
            colorscale='Viridis',
            annotation_text=None
        )

        fig.update_layout(
            **self.plotly_args,
            title_text=f'Confusion Matrix (stage: {stage})'
        )
        fig.update_yaxes(categoryorder='array', categoryarray=axes[::-1])
        # fig['data'][0]['showscale'] = True
        fig.show()

    def plot_confusion_matrix_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots confusion matrix per epochs

        Args:
            `stage` (str): One of stage from `get_stages()` method
            `epochs` (list): List with epochs for plotting
            (epochs counter started from 0). Examples are `[0, 24, 49, 74, 99]`
            or `range(9, self.get_epochs("test"), 10)` (plot every 10th epoch)
        """
        df = self.get_df_pred(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1
        annotations = []

        subplot_titles = []
        for epoch in epochs:
            subplot_titles.append(f"Epoch {epoch}")

        fig = make_subplots(
            rows=rows,
            cols=cols,
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
            cm = confusion_matrix(df["True"], pred)
            axes = [f"Class {i}" for i in range(len(cm))]
            fig_heatmap = ff.create_annotated_heatmap(
                cm,
                x=axes,
                y=axes,
                colorscale='Viridis',
                annotation_text=None
            )
            fig.add_trace(
                fig_heatmap.data[0],
                row=i // cols + 1,
                col=i % cols + 1
            )
            annotation_epoch = list(fig_heatmap.layout.annotations)
            for k in range(len(annotation_epoch)):
                if i > 0:
                    annotation_epoch[k]['xref'] = f'x{i+1}'
                    annotation_epoch[k]['yref'] = f'y{i+1}'
            annotations += annotation_epoch

        for annotation in annotations:
            fig.add_annotation(annotation)
        fig.update_layout(
            **self.plotly_args,
            title_text=f"Confusion Matrix Per Epoch (stage: {stage})"
        )
        fig.update_yaxes(categoryorder='array', categoryarray=axes[::-1])
        fig.show()

    def plot_all(self, stage):
        """
        Plots main charts (losses, all metrics, confusion matrix)

        Args:
            `stage` (str): One of stage from `get_stages()` method
        """
        self.plot_losses(stage)
        self.plot_metrics(stage, self.get_metrics())
        self.plot_confusion_matrix(stage)
