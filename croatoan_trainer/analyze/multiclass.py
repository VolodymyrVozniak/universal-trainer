from typing import List, Dict, Union, Callable, Any

from .classification import _ClassificationAnalyzer


class MulticlassAnalyzer(_ClassificationAnalyzer):
    """
    A class used to analyze info about trained multiclassification model.

    Attributes:
        `results` (dict): Dictionary with results per each stage
        after training. Keys are `cv`, `test` and `final`.
        Main keys for each stage are `losses`, `metrics`, `best_result`
        `time`, `ids`, `true` and `pred`.
        `postprocess_fn` (callable): Function that takes list with
        model outputs from `pred` key for each stage in `results`
        and somehow processes them.

    Methods:
        `get_stages()`: Gets list of stages.
        `get_metrics()`: Gets list of metrics used in training.
        `get_folds()`: Gets number of folds used in training.
        `get_epochs(stage)`: Gets number of epochs for stage.
        `get_time()`: Gets train time in seconds for all stages.
        `get_df_pred(stage)`: Gets dataframe with predictions.
        `get_df_metrics(stages)`: Gets dataframe with metrics.
        `get_metric_result(stage, metric, **kwargs)`: Gets result for metric.
        `print_classification_report(stage, digits)`: Prints classification
        report.
        `plot_losses(stage, fold)`: Plots losses.
        `plot_metrics(stage, metrics, fold)`: Plots metrics.
        `plot_pred_sample(stage, id)`: Plots predictions over epochs
        for one unique id.
        `plot_confusion_matrix(stage)`: Plots confusion matrix.
        `plot_confusion_matrix_per_epoch(stage, epochs)`: Plots confusion
        matrix per epochs.
        `plot_all(stage)`: Plots main charts (losses, all metrics,
        confusion matrix).
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
            and somehow processes them. For multiclassification problem
            it is important to have exact class as final output. So,
            for example, if you have list of logits as your model output,
            define function that will convert your logits into belonging
            to some class (just maximum of these logits)
            (default is `None`).
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
        super().get_metric_result(stage, metric, False, **kwargs)

    def plot_all(self, stage):
        """
        Plots main charts (losses, all metrics, confusion matrix).

        Args:
            `stage` (str): One of stage from `get_stages()` method.
        """
        self.plot_losses(stage)
        self.plot_metrics(stage, self.get_metrics())
        self.plot_confusion_matrix(stage)
