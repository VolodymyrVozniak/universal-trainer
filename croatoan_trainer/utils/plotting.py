from typing import Dict, List, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_targets_hist(
    df: pd.DataFrame,
    target_column: str,
    plotly_args: Dict[str, Any]
):
    df = df.copy()
    df[target_column] = df[target_column].astype("float32")
    fig = px.histogram(df, x=target_column)
    fig.update_layout(
        **plotly_args,
        title_text=f"{target_column} Histogram"
    )
    fig.show()


def plot_split_targets_hist(
    df: pd.DataFrame,
    main_column: str,
    target_column: str,
    train_test_split: Dict[str, Any],
    cv_split: List[Dict[str, Any]],
    plotly_args: Dict[str, Any]
):
    df = df.copy()
    df[target_column] = df[target_column].astype("float32")

    ids_train = train_test_split["train"]
    ids_test = train_test_split["test"]

    targets_train = df[df[main_column].isin(ids_train)][target_column]
    targets_test = df[df[main_column].isin(ids_test)][target_column]

    folds = len(cv_split)
    n_plots = folds * 2 + 2
    cols = 4
    rows = n_plots // cols if n_plots % cols == 0 else n_plots // cols + 1

    subplot_titles = ["Train", "Test"]
    for fold in range(folds):
        subplot_titles += [f"Train_{fold}", f"Val_{fold}"]

    fig = make_subplots(rows=rows, cols=cols,
                        x_title=target_column, y_title="Count",
                        subplot_titles=subplot_titles)

    fig.add_trace(
        go.Histogram(x=targets_train, name="Train"),
        row=1, col=1)

    fig.add_trace(
        go.Histogram(x=targets_test, name="Test"),
        row=1, col=2)

    i = 0
    for y in range(1, rows+1):
        for x in range(1, cols+1):
            if (x == 1 and y == 1) or (x == 2 and y == 1):
                continue
            stage = "train" if x % 2 == 1 else "val"
            try:
                ids = cv_split[i][stage]
            except IndexError:
                break
            targets = df[df[main_column].isin(ids)][target_column]
            fig.add_trace(
                go.Histogram(x=targets, name=f"{stage}_{i}".capitalize()),
                row=y, col=x
            )
            if x % 2 == 0:
                i += 1

    fig.update_layout(
        **plotly_args,
        title_text=f"{target_column} Histograms for Train-Test and CV Splits"
    )
    fig.show()
