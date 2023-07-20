from typing import List, Dict, Union, Callable, Any

import numpy as np
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report

from .abstract import _TrainAnalyzer


class _ClassificationAnalyzer(_TrainAnalyzer):
    def __init__(
        self,
        results: Dict[str, Dict[str, Any]],
        postprocess_fn: Union[None, Callable[[List], List]]
    ):
        super().__init__(results, postprocess_fn)

    def print_classification_report(self, stage: str, digits: int = 3):
        """
        Prints classification report using
        `sklearn.metrics.classification_report` function.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
            `digits` (int):
                Number of digits for formatting output
                floating point values. Default is `3`.
        """
        print(self.get_metric_result(
            stage=stage,
            metric=classification_report,
            digits=digits,
            zero_division=0
        ))

    def plot_confusion_matrix_per_epoch(self, stage: str, epochs: List[int]):
        """
        Plots confusion matrix per epochs.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
            `epochs` (list):
                List with epochs for plotting (epochs counter started from 0).
                Examples are `[0, 24, 49, 74, 99]`
                or `range(9, self.get_epochs("test"), 10)`
                (plot every 10th epoch).
        """
        df = self.get_df_pred(stage)

        best_epoch = self.get_best_epoch(stage)

        n_plots = len(epochs)
        cols = 4 if n_plots > 4 else n_plots
        rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1
        annotations = []

        subplot_titles = []
        for epoch in epochs:
            if epoch == best_epoch:
                subplot_titles.append(f"Best Epoch {epoch}")
            else:
                subplot_titles.append(f"Epoch {epoch}")

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles
        )

        for i, epoch in enumerate(epochs):
            pred = self._get_pred(stage, epoch)

            cm = confusion_matrix(df["True"], np.round(pred))
            x_axes = [f"Pred {i}" for i in range(len(cm))]
            y_axes = [f"True {i}" for i in range(len(cm))]

            fig_heatmap = ff.create_annotated_heatmap(
                cm,
                x=x_axes,
                y=y_axes,
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
            title_text=f"Confusion Matrix (stage: {stage})"
        )
        fig.update_yaxes(categoryorder='array', categoryarray=y_axes[::-1])
        # fig['data'][0]['showscale'] = True
        fig.show()

    def plot_confusion_matrix(self, stage: str):
        """
        Plots confusion matrix.

        Args:
            `stage` (str):
                One of stage from `get_stages()` method.
        """
        best_epoch = self.get_best_epoch(stage)
        self.plot_confusion_matrix_per_epoch(stage, [best_epoch])
